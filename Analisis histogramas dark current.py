# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 17:07:22 2022

@author: luo
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython
from scipy.optimize import curve_fit
import csv
get_ipython().run_line_magic('matplotlib', 'qt5')
plt.rcParams.update({'font.size': 14})

#%%

temp = [-10, -5, 0, 5, 10, 15]
exp = [1, 5, 10, 20, 40, 60, 80, 100, 120, 160, 200, 250, 300]

temperatura = -10

df_list = []
for i in range(len(exp)):
    file_path =r'D:\Imagenes Labo6\dark_2-30V_51G_12_07_22\Histogram_means\mean_hist_{}'.format(temperatura)+'_{}s.csv'.format(exp[i]) 
    temp_df_his0 = pd.read_csv(file_path)
    df_list.append(temp_df_his0)

#%%

def plot_histograma(x, i, temp, lab):
    x_label = x[i].columns[2]
    y_label = x[i].columns[1]
    x_data = np.array((x[i][x_label]))
    y_data = np.array((x[i][y_label]))
    ax.set_title('Histograma - Dark current- {}C'.format(temp) + ' - {}s'.format(exp[i]),fontsize=16)
    ax.set_ylabel('Frecuencia',fontsize=14)
    ax.set_xlabel('Intensidad [escala gris]',fontsize=14)
    ax.bar(x = x_data, height = y_data, width=np.diff(x_data)[0], label = lab)
    ax.set_yscale('log')
    ax.legend(loc = 'upper right')

def stats(x, i):
    x_label = x[i].columns[2]
    y_label = x[i].columns[1]
    x_data = np.array((x[i][x_label]))
    y_data = np.array((x[i][y_label]))
    y_mean = sum(x_data*y_data)/sum(y_data)
    probs = y_data/sum(y_data)
    y_std = np.sqrt(sum(probs*(x_data-y_mean)**2))
    return y_mean, y_std

#%%
fig, ax = plt.subplots()
plot_histograma(df_list,5, temperatura, '{}s'.format(exp[5]))
ax.axvline(x= ym[5],zorder=2,color='r',label='mean')
ax.axvline(x = ym[5]+yd[5],zorder=2,color='orange',label='std', linestyle='dashed')
ax.axvline(x = ym[5]-yd[5],zorder=2,color='orange', linestyle='dashed')
ax.legend(loc = 'upper right')

#%%

for i in range(len(df_list)):
    fig, ax = plt.subplots()
    plot_histograma(df_list,i, temperatura, '{}s'.format(exp[i]))
    plt.show()
    
#%%

px_qty = sum(np.array((df_list[0]['Frequency'])))

print('la cantidad de pixeles son', round(px_qty,0))

#%%

fig,ax = plt.subplots()
tiempos = [11, 5, 1]
exp_tiempos = [exp[tiempos[0]], exp[tiempos[1]], exp[tiempos[2]]]
for i in range(len(tiempos)):
    #plot_histograma(df_list,tiempos[i], temperatura, '{}s'.format(exp_tiempos[i]))
    plot_histograma(df_list,tiempos[i], temperatura, '$\mu_d$\n $\sigma_d$')

#%%

#quito el primer y ultimo dato de cada df que parece algo de error (siempre hay algun pixel saturado)
for i in range(len(df_list)):
    df_list[i]=df_list[i][0:254]

#%%

ym = []
yd = []

for i in range(len(df_list)):
    #fig, ax = plt.subplots()
    #plot_histograma(df_list,i, 0, '{}s'.format(exp[i]))
    #plt.show()
    temp_ym, temp_yd = stats(df_list,i)
    ym.append(temp_ym)
    yd.append(temp_yd)
    
print(ym,yd)


#%%
df_to_save = pd.DataFrame(np.array((exp, ym, yd)).transpose())
df_to_save.to_csv(r'D:\Imagenes Labo6\dark_2-30V_51G_12_07_22\Histogram_means\mean&std_hist_{}C.csv'.format(temperatura))

#%%

ymedian = []

for i in range(len(df_list)):
    max_value = df_list[i][df_list[i]['Count']==max(df_list[i][1:251]['Count'])]
    temp_ymedian = max_value['Value'].values[0]
    ymedian.append(temp_ymedian)

#%%

for i in range(len(df_list)):
    fig, ax = plt.subplots()
    plot_histograma(df_list,i)
    ax.axvline(x= ymedian[i],zorder=2,color='r')
    plt.show()

#%%

N = [1.6, 2.8, 2, 4, 5.6, 8]
#qty = cte*area*tiempo = a*(tiempo/N^2)
#a = flujo*pi*f^2
pN = np.linspace(1.6,8,100)

def qty(x,a):
    return a*(40/x**2)

popt, pcov = curve_fit(qty, N, ym, maxfev= 5000)
perr = np.sqrt(np.diag(pcov)) # errores de 1 sigma

#%%

def mu_d(x, a, b):
    return a*x+b


#%%
temperatura=0
temp_data0 = pd.read_csv(r'D:\Imagenes Labo6\nuevas_dark_51G_04_08_22\mean&std_hist_15C.csv')
temp_data1 = pd.read_csv(r'D:\Imagenes Labo6\dark_2-30V_51G_12_07_22\Histogram_means\mean&std_hist_0C.csv')
temp_data2 = pd.read_csv(r'D:\Imagenes Labo6\dark_2-30V_51G_12_07_22\Histogram_means\mean&std_hist_10C.csv')
exp = np.array((temp_data0.iloc[:,1]))
ym = np.array((temp_data0.iloc[:,2]))
exp1 = np.array((temp_data1.iloc[:,1]))
ym1 = np.array((temp_data1.iloc[:,2]))
exp2 = np.array((temp_data2.iloc[:,1]))
ym2 = np.array((temp_data2.iloc[:,2]))

plt.figure()
plt.scatter(exp, ym,zorder= 1,label='-10C')
plt.show()


#%%

exp = exp[:len(exp)-5]
ym = ym[:len(ym)-5]
popt, pcov = curve_fit(mu_d, exp, ym, maxfev= 5000)
perr = np.sqrt(np.diag(pcov)) # errores de 1 sigma


px = np.linspace(0, 300, 1000)
py = mu_d(px, *popt)

def R2(yd,yf):
    residuals = yd- yf
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yd-np.mean(yd))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

coef_det=R2(ym,mu_d(exp, popt[0],popt[1]))
print("coef de determinacion I es",coef_det)
   

#%%

plt.figure()
plt.scatter(exp, ym,zorder= 1,label='-10C')
plt.plot(px,py, label = 'Ajuste teórico -10C', color='orange')
#plt.plot(exp1, ym1,zorder= 1,label='0C')
#plt.plot(exp2, ym2,zorder= 1, label='10C')
#plt.plot(pN, qty(pN,popt), zorder=2, color='orange')
plt.xlabel('Tiempo de exposición [s]')
#plt.ylabel('Promedio intensidad por px ($\mu_d$)')
plt.ylabel('SNR: $(\mu_y - \mu_{y,dark})/\sigma^2$')
#plt.title('Dark current - {}C - variación tiempo de exposición'.format(temperatura))
plt.title('Dark current - variación tiempo de exposición')
plt.legend(loc='upper left')
plt.show()

#print('La cte es' ,popt,'+-', perr)
#print('El flujo es', popt/(np.pi*35**2), '+-', perr/(np.pi*35**2))

#%%

folderpath = r'D:\Imagenes Labo6\dark_2-30V_51G_12_07_22\Histogram_means\\'

#guardo los datos en un archivo txt
file=open(folderpath + 'Analisis_mu_d.txt','a')
file.write('%10.6f' % (popt[0])+',')
file.write('%10.6f' % (popt[1])+',')
file.write('%10.6f' % (perr[0])+',')
file.write('%10.6f' % (perr[1])+',')
file.write('%10.6f' % (coef_det)+'\n')
file.close()

#%%

df_all_list = []

for i in range(len(n)):
    temp_data = pd.read_csv(r'D:\Imagenes Labo6\dark_2-30V_51G_12_07_22\Histogram_means\mean&std_hist_{}C.csv'.format(n[i]))
    temp_data.drop(['Unnamed: 0'],axis=1)
    temp_data.columns = ['Unnamed', 'Exp Time', 'Mean GV', 'std GV']
    temp_data['Temp'] = np.full(len(temp_data),n[i])
    df_all_list.append(temp_data)

df_all = pd.concat(df_all_list)
df_list_exp = [g for _,g in df_all.groupby('Exp Time')]

#%%

def plot_TempvsGV(x, i, texp):
    x_label = x[i].columns[4]
    y_label = x[i].columns[2]
    x_data = np.array((x[i][x_label]))
    y_data = np.array((x[i][y_label]))
    plt.title('Dark current - {}s - variación temperatura'.format(texp),fontsize=16)
    plt.ylabel('Promedio Gray Value',fontsize=14)
    plt.xlabel('Temperatura (C)',fontsize=14)
    plt.scatter(x_data, y_data, zorder=1)
    
for i in range(len(df_list_exp)):
    plt.figure()
    plot_TempvsGV(df_list_exp, i, int(np.array((df_list_exp[i]['Exp Time']))[0]))
    plt.show()



'''
observaciones:
    donde hay variacion de tiempo de exp: ver que a 300s siempre esta mas bajo, puede ser mi culpa porque siempre lo puse como primero, probarde medir eso de nuevo la vez que viene
    donde hay var de temp: ver que no siempre tiene sentido, solamente en ciertos rangos hay sentido, probar con otro cable usb si es que hay
'''



