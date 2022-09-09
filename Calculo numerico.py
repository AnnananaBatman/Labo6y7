# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 13:23:37 2022

@author: luo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'qt5')
plt.rcParams.update({'font.size': 14})

#%%

def dx_dt(t,x):
    return 2*x-5*np.sin(t)

def dx2_d2t(t,x):
    return x

def solucion_x(t):
    return np.exp(t)
    
def funcion_exacta(h, tf, x0):
    func = np.zeros(len(t))
    func[0] = x0
    for i in range(0, len(t) - 1):
        func[i+1] = solucion_x(t[i+1])
    return t, func

def metodo_euler(h, tf, x0):
    t = np.arange(0, tf, h)
    x_euler = np.zeros(len(t))
    func = np.zeros(len(t))
    func[0] = x0
    x_euler[0] = x0
    for i in range(0, len(t) - 1):
        x_euler[i + 1] = x_euler[i] + h*dx_dt(t[i], x_euler[i])
        #func[i+1] = y_exacta(t[i+1])
    return t, x_euler

def metodo_taylor(h, tf, x0):
    t = np.arange(0, tf, h)
    x_taylor = np.zeros(len(t))
    x_taylor[0] = x0
    for i in range(0, len(t) - 1):
        x_taylor[i + 1] = x_taylor[i] + h*dx_dt(t[i], x_taylor[i]) + (h**2/2)*dx2_d2t(t[i], x_taylor[i])
    return t, x_taylor
    
#%%

hs = [0.1, 0.0625, 0.05, 0.025, 0.01]
df_elist = []

for i in hs:
    t, x_taylor = metodo_euler(i, 1.01, 1)
    t, x_euler = metodo_taylor(i, 1.01, 1)
    t, x_exacta = funcion_exacta(i, 1.01, 1)
    temp_e = abs(x_taylor-x_euler)
    data = np.array((t, temp_e, np.full(len(t),i))).T
    temp_df = pd.DataFrame(data=data, columns=['t', 'tau_i', 'h'])
    df_elist.append(temp_df)
    plt.figure()
    plt.plot(t, x_taylor, 'bo--', label='Taylor')
    plt.plot(t, x_euler, 'g', label='Euler')
    plt.plot(t, x_exacta, 'r--', label='Exacta')
    plt.title('Aproximacion para ODE - h={}'.format(i))
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()

#%%

#veo el error de truncado

plt.figure()
for i in range(len(df_elist)):
    temp_df = df_elist[i]
    t = np.array((df_elist[i].iloc[:,0]))
    tau = np.array((df_elist[i].iloc[:,1]))
    h = df_elist[i].iloc[:,2][0]
    plt.plot(t, tau, label='h={}'.format(h))
    plt.xlabel('t')
    plt.ylabel('$\tau$')
    plt.title('Error Taylor vs Euler')
    plt.grid()
    plt.legend(loc='upper left')
plt.show()

#%%

tau_1 = []

for i in range(len(df_elist)):
    temp_df = df_elist[i]
    t = np.array((df_elist[i].iloc[:,0]))
    tau = np.array((df_elist[i].iloc[:,1]))
    h = df_elist[i].iloc[:,2][0]
    temp_tau = tau[t>0.99]
    tau_1.append(temp_tau)

#%%

plt.figure()
plt.plot(tau_1, np.log(tau_1))
plt.xlabel('tau')
plt.ylabel('log(tau)')
plt.grid()
plt.show()

#seguramente algo de certeza


#%%