import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.integrate import odeint
import math

def sir(y, t, beta, gamma, rho):
    phi = tanh(rho, t, 43)
    ds = -beta*(1-phi)*y[0]*y[1]
    #s = -b(1-d)si
    di = beta*(1-phi)*y[0]*y[1] - gamma*y[1] - 0.03*y[1]
    #I = b(1-d)si - gI - mI
    dr = gamma*y[1]
    #R = gI
    dd = 0.03*y[1]
    return [ds, di, dr, dd]

def tanh(ym, t, tm):
    '''
    Sigmoid function that starts at 0 and plateaus at ym.
    The half way point between 0 and ym will be achieved at tm.
    '''
    return ym*(np.tanh(2. * ((t -tm))/tm) + 1) / 2.


if __name__ == "__main__":
    # Import raw data
    cd_data = pd.read_csv('bc_covid.csv')
    data_rows, data_cols = cd_data.shape
    real_t = np.linspace(0, data_rows-1, data_rows)

    # Model conditions
    y0 = [5071336, 1, 0, 0]
    t = real_t
    # t = np.linspace(0,120,121)
    # Retrieve calibrated parameters for model
    params = []
    with open('model_params.pkl', 'rb') as file:
        while True:
            try:
                params.append(pickle.load(file))
            except EOFError:
                break

    print('{}'.format(params))
    y = odeint(sir, y0, t, args=(params[0], params[1]))
    print('{}'.format(y))
    # plt.plot(t, y[:, 0], label='Susceptible')
    plt.plot(t, y[:, 1], '-b', label='Active Cases')
    plt.plot(t, y[:, 2], '-r', label='Recovered')
    plt.plot(t, y[:, 3], '-g', label='Deaths')
    plt.plot(real_t, cd_data['Cases'], '.b', label='Real Cases')
    plt.plot(real_t, cd_data['Recovered'], '.r', label='Real Recovery')
    plt.plot(real_t, cd_data['Dead'], '.g', label='Real Deaths')
    plt.xlabel('Time since Jan 28',{"fontsize":16})
    plt.ylabel('Cases',{"fontsize":16})
    plt.legend()
    plt.show()