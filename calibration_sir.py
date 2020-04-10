import numpy as np
from scipy import integrate, interpolate
from scipy import optimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pickle
import pandas as pd

# t_data = np.linspace(0,48,49)
cd_data = pd.read_csv('bc_covid.csv')
data_rows, data_cols = cd_data.shape
t_data = np.linspace(0, data_rows-1, data_rows)
inf_data = cd_data['Cases']
reco_data = cd_data['Recovered']

def sir(y, t, param):
    ds = -param[0]*(1-0.5)*y[0]*y[1]
    #s = -b(1-d)si
    di = param[0]*(1-0.5)*y[0]*y[1] - param[1]*y[1] - .03*y[1]
    #I = b(1-d)si - gI - mI
    dr = param[1]*y[1]
    #R = gI
    return [ds, di, dr]


def sir_integrate(x, y0, p):
    yn = integrate.odeint(sir, y0, x, args=(p))
    return yn


def ls_opt(x, fit_p):
    f = lambda y,t: sir(y, t, fit_p)
    r = integrate.odeint(f, y0, x)
    # print('{}'.format(r))
    return r[:,1]

def ls_opt_p(x, fit_p):
    f = lambda y,t: sir(y, t, fit_p)
    r = integrate.odeint(f, y0, x)
    # print('{}'.format(r))
    return [r[:,1], r[:,2]]

def f_resid(p):
    inf, rec = ls_opt_p(t_data, p)
    return np.sqrt((inf_data - inf)**2 + (reco_data - rec)**2)

if __name__ == "__main__":
    
    # Guess parameters
    param_g = [3.3507*10**-8, 0.0569384]
    y0 = [5071336, 1, 0]
    res = optimize.least_squares(f_resid, param_g, bounds=(0, [.1, .6]))
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