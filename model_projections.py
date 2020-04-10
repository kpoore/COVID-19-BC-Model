import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.integrate import odeint
import math

def sir(y, t, beta, rho, nu, gamma):
    phi = tanh(0.0, rho, t, 43)
    ds = -beta*(1-phi)*y[0]*(y[1] + y[2])
    #S = -b(1-p)s(iu+ia)
    diu = beta*(1-phi)*y[0]*(y[1] + y[2]) - nu * y[1]
    #Iu = -b(1-p)s(iu+ia) - niu
    dia = nu * y[1] - gamma*y[2] - .008*y[2]
    #Ia = nIu - gIa - mIa
    dr = gamma*y[2]
    #R = gIa
    dd = 0.008*y[2]
    return [ds, diu, dia, dr, dd]

def tanh(y0, ym, t, tm):
    '''
    Sigmoid function that starts at 0 and plateaus at ym.
    The half way point between 0 and ym will be achieved at tm.
    '''
    return (ym - y0)*(np.tanh(2. * ((t -tm))/tm) + 1) / 2. + y0


if __name__ == "__main__":
    # Import raw data
    cd_data = pd.read_csv('bc_covid.csv')
    data_rows, data_cols = cd_data.shape
    real_t = np.linspace(0, data_rows-1, data_rows)

    # Model conditions
    y0 = [5071336, 0, 1, 0, 0]
    # t = real_t
    t = np.linspace(0,91,92)
    # Retrieve calibrated parameters for model
    params = []
    with open('model_params.pkl', 'rb') as file:
        while True:
            try:
                params.append(pickle.load(file))
            except EOFError:
                break

    print('{}'.format(params))
    y = odeint(sir, y0, t, args=(params[0], params[1], params[2], params[3]))
    print('{}'.format(y))
    # plt.plot(t, y[:, 0], label='Susceptible')
    plt.plot(t, y[:, 2], '-b', label='Active Cases')
    plt.plot(t, y[:, 3], '-r', label='Recovered')
    plt.plot(t, y[:, 4], '-g', label='Deaths')
    plt.plot(real_t, cd_data['Infected'], '.b', label='Real Cases')
    plt.plot(real_t, cd_data['Recovered'], '.r', label='Real Recovery')
    plt.plot(real_t, cd_data['Dead'], '.g', label='Real Deaths')
    plt.xlabel('Time since Jan 28',{"fontsize":16})
    plt.ylabel('Cases',{"fontsize":16})
    plt.legend()
    plt.show()

    plt.plot(t, np.sum(y[:, 1:], axis=1), label='Total Cases')
    plt.legend()
    plt.show()