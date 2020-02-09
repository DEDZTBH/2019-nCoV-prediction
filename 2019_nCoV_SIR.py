import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import pandas as pd
from datetime import datetime, timedelta

info_frame = pd.read_csv('data/2019-nCoV.csv')
info_frame.fillna(method='ffill', inplace=True)
info_frame.fillna(method='bfill', inplace=True)
init_date = datetime.strptime(info_frame['Date CST'][0], '%Y-%m-%d')
today_date = datetime.strptime(info_frame['Date CST'].iloc[-1], '%Y-%m-%d')
today_days = (today_date - init_date).days
info_frame['Date CST'] = info_frame['Date CST'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - init_date).days)
X = info_frame['Date CST'].to_numpy()
X = np.array(X, dtype=np.float32)
X = X.reshape(-1, 1)

Y_frame = info_frame.drop(['Date CST'], axis=1)
Ys = Y_frame.to_numpy().transpose()

# Total population, N.
N = 90000000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 55, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.43, 6e-3
# A grid of time points (in days)
t = np.linspace(0, 365, 365)


# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T


def graph():
    plt.clf()
    plt.rcParams['figure.figsize'] = [8, 6]

    axs = [None, None]
    axs[0] = plt.subplot()

    axs[0].title.set_text(
        'Prediction as of {} CST (SIR model)'.format(str(today_date + timedelta(hours=23, minutes=59, seconds=59))))
    axs[0].set_xlabel('Days after 2019-12-31')
    axs[0].set_ylabel('Cases')

    axs[0].plot(X, Ys[0] + Ys[1], 'bo', label='Actual (100)', alpha=0.5)
    axs[0].plot(X, Ys[3], 'go', label='Recovered', alpha=0.5)
    axs[0].set_xlim(-2.25, today_days + 29)
    axs[0].set_xticks(np.arange(today_days + 30, step=5))

    axs[0].plot(t, I / 1000, 'r', alpha=0.5, lw=2, label='Infected')
    axs[0].plot(t, R / 1000, 'g', alpha=0.5, lw=2, label='Recovered (100)')

    axs[0].grid()
    axs[0].legend(loc='best')

    plt.savefig('predictionSIR.png', dpi=300)

    plt.show()


graph()
graph()