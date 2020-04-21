import numpy as np
import pandas as pd
from scipy import integrate, interpolate
from scipy import optimize
import matplotlib.pyplot as plt
import pickle
from scipy.integrate import odeint
import math
import seaborn as sn


# t_data = np.linspace(0,48,49)
# collect data from CSV file
cda_data = pd.read_csv('BCCDC_COVID19_Dashboard_Case_Details.csv')
# data_rows, data_cols = cd_data.shape
t_data = np.linspace(0, 83, 84)
age_data = cda_data['Age_Group']

cd_data = pd.read_csv('bc_covid.csv')
inf_data = cd_data['Infected']
reco_data = cd_data['Recovered']
dead_data = cd_data['Dead']
contact_rate = pd.read_csv('contact_rate.csv', index_col=False, header=None).to_numpy()
C = contact_rate

def sir(y, t, params):
    '''
    Basic SIR model with a compartment for infected and unaware (asymptomatic)
    and a compartment for aware (symptomatic).
    Parameters include transmission rate, beta, social distancing factor, rho,
    transition rate from unaware to aware, nu, and recovery rate, gamma.
    A rate of 0.8% death rate is assumed as it matches local data.
    A sigmoid function is used for the social distancing rate as a way
    to "ramp up" social distancing in a matter of days without breaking
    the ODE.
    '''
    beta, rho, nu, gamma, mu0, mu1, mu2, mu3, mu4, mu5, mu6, mu7 = params
    mu = [mu0, mu1, mu2, mu3, mu4, mu5, mu6, mu7]
    nu = 5.5/365.
    gamma = 21/365.
    y = np.reshape(y, (5, 8))
    phi = tanh(0.0, rho, t, 43)
    ds = np.zeros(8)
    diu = np.zeros(8)
    dia = np.zeros(8)
    dr = np.zeros(8)
    dd = np.zeros(8)
    for i in range(len(C)):
        for j in range(len(C)):
            ds[i] += -beta*(1-phi)*C[i][j]*y[0][i]*(y[1][j] + y[2][j])
            #S = -b(1-p)s(iu+ia)
            diu[i] += beta*(1-phi)*C[i][j]*y[0][i]*(y[1][j] + y[2][j]) - nu * y[1][i]
            #Iu = -b(1-p)s(iu+ia) - niu
        dia[i] = nu * y[1][i] - gamma*y[2][i] - mu[i]*y[2][i]
        #Ia = nIu - gIa - mIa
        dr[i] = gamma*y[2][i]
        #R = gIa
        dd[i] = mu[i]*y[2][i]
    dy = np.array([ds, diu, dia, dr, dd])
    dy = dy.flatten()
    return dy


def tanh(y0, ym, t, tm):
    '''
    Sigmoid function that starts at 0 and plateaus at ym.
    The half way point between 0 and ym will be achieved at tm.
    '''
    return (ym - y0)*(np.tanh(2. * ((t -tm))/tm) + 1) / 2. + y0

def ls_opt(x, fit_p):
    '''
    Least Squares fitting for infectious cases
    '''
    f = lambda y,t: sir(y, t, fit_p)
    r = integrate.odeint(f, y0, x)
    # print('{}'.format(r))
    return r[:,2]


def ls_opt_p(x, fit_p):
    '''
    Least Squares fitting for infectious and recovered cases.
    '''
    f = lambda y,t: sir(y, t, fit_p)
    r = integrate.odeint(f, y0, x)
    # print('{}'.format(len(r)))
    sol = np.reshape(r, (84, 5, 8))
    inf = np.sum(sol[:,2,:], axis=1)
    rec = np.sum(sol[:,3,:], axis=1)
    dea = np.sum(sol[:,4,:], axis=1)
    return [inf, rec, dea]

def f_resid(p):
    inf, rec , death= ls_opt_p(t_data, p)
    '''
    Calculate residuals to measure the fitness of the parameters
    '''
    return (inf_data - inf)**2 + (reco_data - rec) + (dead_data - death)

if __name__ == "__main__":
    
    # Guess parameters
    param_g = [2.35401407e-08, 9.900000e-01,  6/365., 14/365., 0, 0, 0, 0, 0.04, 0.07, 0.1, 0.13]
    y_init = np.array([[468921, 521779, 690552, 717836, 646413, 723713, 659506, 642616],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0]])
    y0 = y_init.flatten()
    res = optimize.least_squares(f_resid, param_g, ftol=10**-9, method='trf',
                                bounds=([1e-10, 0, 4.0/365., 1/365., 0, 0, 0, 0, 0, 0, 0, 0],
                                [1e-7, 1, 14/365., 20/365, 0.1, 0.1, 0.1, 0.3, 0.3, 0.5, .5, .5]))
    # c = optimize.least_squares(f_resid, param_g, bounds=(0, [0.05, 0.6]))
    # (c,kvg) = optimize.curve_fit(f_resid, t_data, inf_data, p0=3.4*10**-4, bounds=([0,0,0], [1, 1, 1]))
    # (c,kvg) = optimize.curve_fit(f_resid, t_data, inf_data, p0=2.5*10**-8)
    print('Parameters are estimated at {}'.format(res.x))
    params = res.x
    # fit ODE results to interpolating spline just for fun
    xeval=np.linspace(min(t_data), max(t_data),30) 
    gls = interpolate.UnivariateSpline(xeval, ls_opt(xeval,params), k=3, s=0)

    # save variables in pickle for future use
    with open('model_params.pkl', 'wb') as file:
        for i in range(len(params)):
            pickle.dump(params[i], file)
    #pick a few more points for a very smooth curve, then plot 
    #   data and curve fit
    xeval=np.linspace(min(t_data), max(t_data),200)
    #Plot of the data as red dots and fit as blue line
    plt.plot(t_data, inf_data,'.r',xeval,gls(xeval),'-b')
    plt.xlabel('Time since Jan 28',{"fontsize":16})
    plt.ylabel('Total Cases',{"fontsize":16})
    plt.legend(('data','fit'),loc=0)
    plt.show()