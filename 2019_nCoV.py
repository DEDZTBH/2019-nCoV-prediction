import pandas as pd
import numpy as np
import torch
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

info_frame = pd.read_csv('data/2019-nCoV.csv')
info_frame.fillna(method='ffill', inplace=True)
info_frame.fillna(method='bfill', inplace=True)
init_date = datetime.strptime(info_frame['Date CST'][0], '%Y.%m.%d')
today_date = datetime.strptime(info_frame['Date CST'].iloc[-1], '%Y.%m.%d')
today_days = (today_date - init_date).days
info_frame['Date CST'] = info_frame['Date CST'].apply(lambda x: (datetime.strptime(x, '%Y.%m.%d') - init_date).days)
X = info_frame['Date CST'].to_numpy()
X = np.array(X, dtype=np.float32)
X = X.reshape(-1, 1)

Y_frame = info_frame.drop(['Date CST'], axis=1)
Ys = Y_frame.to_numpy().transpose()

device = torch.device("cpu")


class SigmoidRegression(torch.nn.Module):
    def __init__(self, a, b, k, x0):
        super(SigmoidRegression, self).__init__()

        self.a = torch.nn.Parameter((torch.ones(1) * a), requires_grad=True)
        self.b = torch.nn.Parameter((torch.ones(1) * b), requires_grad=True)
        self.k = torch.nn.Parameter((torch.ones(1) * k), requires_grad=True)
        self.x0 = torch.nn.Parameter((torch.ones(1) * x0), requires_grad=True)

        self.oa = torch.nn.Parameter((torch.ones(1) * a), requires_grad=False)
        self.ok = torch.nn.Parameter((torch.ones(1) * k), requires_grad=False)
        self.ob = torch.nn.Parameter((torch.ones(1) * b), requires_grad=False)
        self.ox0 = torch.nn.Parameter((torch.ones(1) * x0), requires_grad=False)

    def forward(self, x):
        ones = torch.ones(x.shape[0], 1).to(device)
        return (torch.sigmoid((x - ones * self.x0) * self.k) * self.a + ones * self.b).squeeze()


learningRate = 0.03
epochs = 100000


def trainModel(model, X, y):
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    last_loss = 0
    early_terminate_counter = 0
    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        inputs = torch.from_numpy(X).to(device)
        labels = torch.from_numpy(y).to(device)

        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels) \
               + 1e07 * (model.k - model.k) ** 2 \
               + 1e01 * (model.b - model.ob) ** 2 \
               + 3e04 * (model.x0 - model.ox0) ** 2
        # print(loss)
        # get gradients w.r.t to parameters
        loss.backward()
        model.k.grad /= 1e07
        model.b.grad /= 1e01
        model.x0.grad /= 3e04

        # update parameters
        optimizer.step()

        if epoch % 1000 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))

        if (last_loss - loss) ** 2 < 1e-7:
            early_terminate_counter += 1
        else:
            early_terminate_counter = 0

        if early_terminate_counter >= 100:
            print('[early terminate] epoch {}, loss {}'.format(epoch, loss.item()))
            break

        last_loss = loss


sModel = SigmoidRegression(Ys[0][-1] * 2, 27, 1, today_days + 1)
cModel = SigmoidRegression(Ys[1][-1] * 2, 41, 1, today_days + 1)
scModel = SigmoidRegression((Ys[0] / 2 + Ys[1])[-1] * 2, 55, 1, today_days + 1)
dModel = SigmoidRegression(Ys[2][-1] * 2, 0, 1, today_days + 1)

trainModel(sModel, X, np.array(Ys[0], dtype=np.float32))
trainModel(cModel, X, np.array(Ys[1], dtype=np.float32))
trainModel(scModel, X, np.array(Ys[0] / 2 + Ys[1], dtype=np.float32))
trainModel(dModel, X, np.array(Ys[2], dtype=np.float32))

spredicted = []
cpredicted = []
scP = []
dP = []

with torch.no_grad():  # we don't need gradients in the testing
    for i in np.arange(today_days + 30):
        spredicted.append(sModel(torch.from_numpy(np.array([i])).to(device)).cpu().data.numpy().item())
        cpredicted.append(cModel(torch.from_numpy(np.array([i])).to(device)).cpu().data.numpy().item())
        scP.append(scModel(torch.from_numpy(np.array([i])).to(device)).cpu().data.numpy().item())
        dP.append(dModel(torch.from_numpy(np.array([i])).to(device)).cpu().data.numpy().item())

plt.clf()
plt.rcParams['figure.figsize'] = [8, 14]

gs = plt.GridSpec(nrows=12, ncols=1)
axs = [None, None]
axs[0] = plt.subplot(gs[:5, :])
axs[1] = plt.subplot(gs[5:, :])

axs[0].title.set_text('Prediction as of {}'.format(str(today_date)))

collabel = ("Date", "Suspect", "Confirm", "Predicted Actual", "Death")
axs[1].axis('tight')
axs[1].axis('off')
the_table = axs[1].table(
    cellText=np.array([[str((init_date + timedelta(days=i)).date()) for i in range(today_days + 30)],
                       np.round(spredicted).astype(int),
                       np.round(cpredicted).astype(int),
                       np.round(scP).astype(int),
                       np.round(dP).astype(int)], dtype=str).transpose()
    [today_days + 1: today_days + 31],
    colLabels=collabel,
    cellLoc='center',
    loc='center')

axs[0].set_xlabel('Days after 2019-12-31')
axs[0].set_ylabel('Cases')

axs[0].plot(X, Ys[0], 'bo', label='Suspect', alpha=0.5)
axs[0].plot(np.arange(today_days + 30), spredicted, '--', label='Suspect Predictions', alpha=0.5)

axs[0].plot(X, Ys[1], 'yo', label='Confirm', alpha=0.5)
axs[0].plot(np.arange(today_days + 30), cpredicted, '-', label='Confirm Predictions', alpha=0.5)

axs[0].plot(X, Ys[0] / 2 + Ys[1], 'go', label='Suspect/2+Confirm', alpha=0.5)
axs[0].plot(np.arange(today_days + 30), scP, '-', label='Actual Predictions', alpha=0.5)

axs[0].plot(X, Ys[2], 'ro', label='Death', alpha=0.5)
axs[0].plot(np.arange(today_days + 30), dP, '-', label='Death Predictions', alpha=0.5)

axs[0].set_xlim(-2.25, today_days + 29)
axs[0].set_xticks(np.arange(today_days + 30, step=5))

axs[0].grid()
axs[0].legend(loc='best')

plt.savefig('prediction.png', dpi=300)
plt.show()
