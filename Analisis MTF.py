# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 19:11:42 2022

@author: luo
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from scipy import ndimage,signal
from scipy.signal import savgol_filter,argrelextrema
from scipy.optimize import curve_fit
import pandas as pd
from scipy.fft import fft, fftfreq, irfft

get_ipython().run_line_magic('matplotlib', 'qt5')
plt.rcParams.update({'font.size': 14})

#%%
# create empty list
df_borde_list = []
df_sine_list = []
df_sieme_list = []
df_list = [df_borde_list, df_sine_list, df_sieme_list]


n = [1.4, 2, 2.8, 4, 8]


# append datasets into the list
for i in range(len(n)):
    path_borde = r"D:\Imagenes Labo6\variacion diaf\f{}_borde.csv".format(n[i])
    path_sine = r"D:\Imagenes Labo6\variacion diaf\f{}_seno.csv".format(n[i])
    #path_sieme = r"D:\Imagenes Labo6\variacion diaf\f{}_siemens_centro.csv".format(n[i])
    temp_df_1 = pd.read_csv(path_borde)
    df_borde_list.append(temp_df_1)
    temp_df_2 = pd.read_csv(path_sine)
    temp_df_2['Gray_Value']=np.flip(np.array((temp_df_2['Gray_Value'])))
    df_sine_list.append(temp_df_2)
    #temp_df_3 = pd.read_csv(path_sieme)
    #temp_df_3['Gray_Value']=np.flip(np.array((temp_df_3['Gray_Value'])))
    #df_sieme_list.append(temp_df_3)

#%%

#funciones
def plot_perfil(x, i, title, lab):
    x_label = x[i].columns[0]
    y_label = x[i].columns[1]
    x_data = np.array((x[i][x_label]))
    temp_ydata = np.array((x[i][y_label]))
    y_data = temp_ydata
    #y_data = (temp_ydata-min(temp_ydata))/(max(temp_ydata)-min(temp_ydata))
    plt.plot(x_data, y_data, label=lab)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


def fourier_transf(x, i, c):
    x_label = x[i].columns[0]
    y_label = x[i].columns[1]
    x_data = np.array((x[i][x_label]))
    y_data = np.array((x[i][y_label]))
    y_fft = abs(fft(y_data))
    y_fft = (y_fft-min(y_fft))/(max(y_fft)-min(y_fft))
    #fig, axes = plt.subplots(nrows=2, ncols=1)
    #axes[0].plot(x_data, y_data,color='cornflowerblue')
    #axes[0].set_xlabel(x_label)
    #axes[0].set_ylabel(y_label)
    #axes[0].set_title(title_x)
    #fig.subplots_adjust(hspace=0.5)
    plt.plot(x_data,y_fft,color=c, label='variación Lfocal 35mm')
    plt.plot(-1*x_data,y_fft,color=c)
    #plt.title(title_freq) 
    #plt.xlabel(x_label)
    #xlim(-10,10) 
    #show()
    #return y_fft

def derivada_discreta(x, i):
    x_label = x[i].columns[0]
    y_label = x[i].columns[1]
    x_data = np.array((x[i][x_label]))
    y_data = np.array((x[i][y_label]))
    y_data = y_data-y_data[0]
    y_der = []
    for m in range(-1, len(x[i])-1):
        temp_y = (y_data[m+1]-y_data[m])/(2*(x_data[m+1]-x_data[m]))
        y_der.append(temp_y)
    return np.array((y_der))

def derivada_vs_fft(x, y_derivada, i, title):
    x_label = x[i].columns[0]
    y_label = x[i].columns[1]
    x_data = np.array((x[i][x_label]))
    y_data = np.array((x[i][y_label]))
    y_der = y_derivada[i]
    y_der = (y_der-min(y_der))/(max(y_der)-min(y_der))
    y_fft = abs(fft(y_data))
    y_fft = (y_fft-min(y_fft))/(max(y_fft)-min(y_fft))
    plt.figure()
    plt.plot(x_data, y_der,color='orange', label='Derivada discreta')
    plt.plot((x_data+x_data[y_der == max(y_der)]),y_fft,color='g', label='FFT')
    plt.plot(-1*(x_data-x_data[y_der == max(y_der)]),y_fft,color='g')
    plt.title(title) 
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper right')
    plt.show()

   
def plot_mtf(x, i, title, lab):
    x_label = x[i].columns[0]
    y_label = x[i].columns[1]
    x_data = np.array((x[i][x_label]))
    y_data = np.array((x[i][y_label]))
    y_fft = abs(fft(y_data))
    py_mtf, px_mtf = fft(y_fft), fftfreq(len(x_data), np.mean(np.diff(x_data)))
    py_mtf = (py_mtf-min(py_mtf))/(max(py_mtf)-min(py_mtf)) #normalizo los valores de mtf
    x_mtf = px_mtf[(px_mtf>= 0) & (py_mtf <= max(py_mtf))].real #tomo la parte real de fft
    y_mtf = py_mtf[(px_mtf>= 0) & (py_mtf <= max(py_mtf))].real
    plt.plot(x_mtf, y_mtf, label=lab)
    plt.title(title)
    plt.xlabel('Frecuencia [lp/mm]')
    return x_mtf, y_mtf


#%%

#Veo los datos originales
for j in range(len(n)):
    plt.figure()
    plot_perfil(df_list[0], j, 'Perfil borde - 25mm - variacion {}'.format(n[j]), 'apertura f/{}'.format(n[j]))
    plt.show()

#%%

#Figura de comparacion de algunos perfiles

num = [0,2,4]

plt.figure()
for k in num:
    plot_perfil(df_list[0], k, '', 'apertura f/{}'.format(n[k]))
plt.legend(loc='upper left')
plt.xlabel('Posición [mm]')
plt.ylabel('Intensidad [escala gris]')
plt.title('Perfil borde - 25mm - 2.36V - G51')
plt.show()

#%%

#ANALISIS DE BORDE
#Figura de todos los mtf
xdata_mtf = []
ydata_mtf = []
df_mtf_list = []
xc = []

plt.figure()
for k in range(len(n)):
    temp_x_mtf, temp_y_mtf = plot_mtf(df_list[0], k, '', 'apertura f/{}'.format(n[k]))
    temp_xc= min(temp_x_mtf.real[temp_y_mtf.real < (max(temp_y_mtf.real)*0.1)])
    xc.append(temp_xc)
plt.legend(loc='upper right')
plt.xlabel('Frecuencia [lp/mm]')
plt.ylabel('Transferencia de contraste')
plt.title('MTF borde - 25mm - variación diafragma - 2.36V - G51')
plt.show()

Res = 1/(2*np.array((xc)))*1000 #micrometro

print('Las configuraciones son ', n)
print('Las resoluciones son ', Res, 'micrometro')

#%%

#Derivada vs fft para linespread
der_dis = []
for j in range(len(n)):
    temp_der = derivada_discreta(df_list[0], j)
    der_dis.append(temp_der)

for k in range(len(n)):
    derivada_vs_fft(df_list[0], der_dis, k, 'Linespread - derivada vs fft - #{}'.format(n[k]))
    
puntos = plt.ginput(2)
puntos = np.array((puntos))
print('el ancho de banda es aprox', np.diff(puntos[:,0]), 'micrometro')

#%%

#Funciones para ajustar mtf, no sirve mucho porque son para sistemas de lentes muy ideales, y aca tenemos muchas imperfecciones

def mtf(x, xi_c, a):
    f = 25
    #xi_c = D/(lamb*f)
    phi = 1/(np.cos(x/xi_c))
    return 2*a*(phi-np.cos(phi)*np.sin(phi))/(np.pi)

def mtf_eff(x,xi_c):
    return (abs((np.sin(x/xi_c))/(x/xi_c)))**3

popt, pcov = curve_fit(mtf_eff, x_mtf, y_mtf, maxfev= 5000)
perr = np.sqrt(np.diag(pcov)) # errores de 1 sigma

px = np.linspace(min(x_mtf), max(x_mtf),500)

plt.figure()
plt.plot( x_mtf, y_mtf , label='Datos')
plt.plot(px, mtf_eff(px,*popt),label='Modelo MTF')
plt.xlabel('Frecuencia [lp/mm]')
plt.title('Perfil MFT - Lfocal 2')
plt.legend(loc='upper right')
plt.show()


#%%
#ANALISIS DEL SENO

from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_max(x,i ,f,title, lab):
    x_label = x[i].columns[0]
    y_label = x[i].columns[1]
    x_data = np.array((x[i][x_label]))
    y_data = np.array((x[i][y_label]))/max(np.array((x[i][y_label])))
    #y_data = np.array((x[i][y_label]))
    filtered = lowess(y_data,x_data, frac=f, it=0)
    filtered_x, filtered_y=filtered[:,0], filtered[:,1]
    #x_max, y_max=x_temp[argrelextrema(y_temp, np.greater)[0]], y_temp[argrelextrema(y_temp, np.greater)[0]]
    x_max, y_max=filtered_x[argrelextrema(filtered_y, np.greater)[0]], filtered_y[argrelextrema(filtered_y, np.greater)[0]]
    x_max = x_max[y_max >0.2]
    y_max = y_max[y_max >0.2]
    plt.plot(x_data, y_data,zorder=1)
    plt.plot(filtered_x,filtered_y,zorder=2, color='orange', label=lab)
    plt.scatter(x_max, y_max, color='r', zorder = 3) 
    plt.xlabel(x_label) 
    plt.ylabel(y_label)
    plt.title(title) 
    y_max_norm = y_max/max(y_max)
    return x_max, y_max, y_max_norm

#%%

xdata_max = []
ydata_max = []
ydata_max_norm = []
xc_sine = []
popt_sine = []
perr_sine = []


for k in range(len(n)):
    plt.figure()
    temp_x_max, temp_y_max, temp_y_max_norm = plot_max(df_list[1], k, 0.0178, '','#{}'.format(n[k]))
    #popt0, pcov0 = curve_fit(mtf_eff, temp_x_max, temp_y_max_norm, maxfev= 5000)
    #perr0 = np.diag(pcov0)
    #px0 = np.linspace(min(temp_x_max), 15,500)
    #py0 = mtf_eff(px0,*popt0)
    #temp_xc_sine= max(temp_x_max[temp_y_max <= (max(temp_y_max)*0.6)])
    #xc_sine.append(temp_xc_sine)
    #popt_sine.append(popt0)
    #perr_sine.append(perr0)
    xdata_max.append(temp_x_max)
    ydata_max.append(temp_y_max)
    ydata_max_norm.append(temp_y_max_norm)
    plt.legend(loc='upper right')
    plt.xlabel('Posición [mm]')
    plt.ylabel('Intensidad [escala gris]')
    plt.title('Perfil Siemens - 25mm - variación diafragma - 2.36V - G51')
    plt.show()

Res_sine = 1/(2*np.array((xc_sine)))*1000

print('Las configuraciones son ', n)
print('La resolucion del patron sinusoidal es de ', Res_sine, 'micrometro')

#%%

def mtf_eff(f,f0):
    return 1/(1+abs(f/f0)**2)

plt.figure()
for i in (0,2,4):
    #pxdata_max = xdata_max[i][2:len(xdata_max[0])]
    #pydata_max_norm = ydata_max_norm[i][2:len(xdata_max[0])]
    #plt.scatter(pxdata_max[i],pydata_max_norm[i])
    plt.plot(xdata_max[i],ydata_max_norm[i],label = 'apertura f/{}'.format(n[i]))
    px0 = np.linspace(min(xdata_max[i]), 15,500)
    #py0 = mtf_eff(px0,*popt_sine[i])
    #plt.plot(px0,py0, label = 'variación #{}'.format(n[i]))
    #temp_xc_sine_mtf= min(px0[py0 <= (max(py0)*0.1)])
    #xc_sine_mtf.append(temp_xc_sine_mtf)
plt.legend(loc='upper right')
plt.xlabel('Frecuencia [lp/mm]')
plt.ylabel('Transferencia de contraste')
plt.title('Perfil MTF barras - 25mm - variación diafragma')
plt.show()

#%%


#%% 
i=1
plt.figure()
plt.scatter(xdata_max[i],ydata_max[i],color='orange')
px0 = np.linspace(min(xdata_max[i]), 15,500)
py0 = mtf_eff(px0,*popt_sine[i])
plt.plot(px0,py0, label = 'variación #{}'.format(n[i]))
plt.legend(loc='upper right')
plt.xlabel('[1/mm]')
plt.ylabel('MTF')
plt.title('Ajuste MTF sine - 25mm - variación diafragma')
plt.show()

#%%

folderpath = r'C:\Users\luo\OneDrive\文档\Labo 6y7\15_06_22\10_47_48\diafragmas_2.34V_Ganancia50\\'
folderpath_sine = r'C:\Users\luo\OneDrive\文档\Labo 6y7\15_06_22\10_47_48\diafragmas_2.34V_Ganancia50\\'

#guardo los datos en un archivo txt
file=open(folderpath + 'Analisis_res.txt','a')
for i in range(len(Res)-1):
    file.write('%10.6f' % (Res[i])+',')
file.write('%10.6f' % (Res[len(Res)-1])+'\n')
file.close()

#%%
file_sine=open(folderpath_sine + 'Analisis_res_sine.txt','a')
for i in range(len(Res_sine)-1):
    file.write('%10.6f' % (Res_sine[i])+',')
file.write('%10.6f' % (Res_sine[len(Res_sine)-1])+'\n')
file_sine.close()

file_sine0=open(folderpath_sine + 'Analisis_xic_sine.txt','a')
for i in range(len(popt_sine)-1):
    file.write('%10.6f' % (popt_sine[i])+',')
file.write('%10.6f' % (popt_sine[len(popt_sine)-1])+'\n')
file_sine0.close()



