import pandas as pd
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt

info_frame = pd.read_csv('data/2019-nCoV.csv')
info_frame.fillna(method='ffill', inplace=True)
info_frame.fillna(method='bfill', inplace=True)
init_date = datetime.strptime(info_frame['Date CST'][0], '%Y.%m.%d')
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
epochs = 30000

sModel = SigmoidRegression(2000, 27, 1, 27)
cModel = SigmoidRegression(2000, 41, 0.6, 27)
scModel = SigmoidRegression(4000, 55, 0.8, 27)
dModel = SigmoidRegression(1000, 0, 0.4, 27)


def trainModel(model, X, y):
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

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


trainModel(sModel, X, np.array(Ys[0], dtype=np.float32))
trainModel(cModel, X, np.array(Ys[1], dtype=np.float32))
trainModel(scModel, X, np.array(Ys[0] / 2 + Ys[1], dtype=np.float32))
trainModel(dModel, X, np.array(Ys[2], dtype=np.float32))

spredicted = []
cpredicted = []
scP = []
dP = []
with torch.no_grad():  # we don't need gradients in the testing
    for i in np.arange(60):
        spredicted.append(sModel(torch.from_numpy(np.array([i])).to(device)).cpu().data.numpy())
        cpredicted.append(cModel(torch.from_numpy(np.array([i])).to(device)).cpu().data.numpy())
        scP.append(scModel(torch.from_numpy(np.array([i])).to(device)).cpu().data.numpy())
        dP.append(dModel(torch.from_numpy(np.array([i])).to(device)).cpu().data.numpy())


plt.clf()
plt.plot(X, Ys[0], 'bo', label='Suspect', alpha=0.5)
plt.plot(np.arange(60), spredicted, '--', label='Suspect Predictions', alpha=0.5)

plt.plot(X, Ys[1], 'go', label='Confirm', alpha=0.5)
plt.plot(np.arange(60), cpredicted, '-', label='Confirm Predictions', alpha=0.5)

plt.plot(X, Ys[0] / 2 + Ys[1], 'ro', label='Suspect/2 + Confirm', alpha=0.5)
plt.plot(np.arange(60), scP, '-', label='Suspect/2 + Confirm Predictions', alpha=0.5)

plt.plot(X, Ys[2], 'yo', label='Death', alpha=0.5)
plt.plot(np.arange(60), dP, '-', label='Death Predictions', alpha=0.5)

plt.legend(loc='best')
plt.show()
