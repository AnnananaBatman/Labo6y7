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
import statsmodels.api as sm
from scipy.optimize import curve_fit
from decimal import Decimal
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.sandbox.regression.predstd import wls_prediction_std
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

n = [ "25s_26cms"]

n = [1.4, 2, 2.8, 4, 8]

n = [1.4, 1.4, 2.8, 8, 8]

m = [1.6, 1.6, 1.6, 1.6, 1.6]

#path_sine = r"D:\Imagenes Labo6\22_06_22\11_28_15\f1.6_"+ n[i]+"_sine.csv"
# append datasets into the list
for i in range(len(n)):
    path_borde = r"D:\Imagenes Labo6\variacion diaf\f{}_borde_ldV.csv".format(n[i])
    #path_borde = r"D:\Imagenes Labo6\22_06_22\11_28_15\f1.6_"+ n[i]+"_edge1.csv"
    path_sine = r"D:\Imagenes Labo6\variacion diaf\f{}_seno_centro1.csv".format(n[i])
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

def plot_perfil(x, i, title):
    x_label = x[i].columns[0]
    y_label = x[i].columns[1]
    x_data = np.array((x[i][x_label]))
    temp_ydata = np.array((x[i][y_label]))
    y_data = temp_ydata
    #y_data = (temp_ydata-min(temp_ydata))/(max(temp_ydata)-min(temp_ydata))
    plt.plot(x_data, y_data, label='apertura f/{}'.format(n[i]))
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

   
def plot_mtf(x, i, title):
    x_label = x[i].columns[0]
    y_label = x[i].columns[1]
    x_data = np.array((x[i][x_label]))
    y_data = np.array((x[i][y_label]))
    y_fft = abs(fft(y_data))
    #plt.figure(1)
    #plt.plot(x_freqs,y_fft)
    #plt.title(title_freq)
    #ptos = plt.ginput(2)
    #ptos = np.array(ptos)
    #plt.figure(1)
    #plt.plot(x_freqs,y_fft)
    #plt.scatter(ptos[0][0],ptos[0][1], color = 'r')
    #plt.scatter(ptos[1][0],ptos[1][1], color = 'r')
    #plt.title(title_freq)
    #band = fft(y_fft[(y_fft>ptos[1][0]) & (y_fft< ptos[1][1])])
    py_mtf, px_mtf = fft(y_fft), fftfreq(len(x_data), np.mean(np.diff(x_data)))
    py_mtf = (py_mtf-min(py_mtf))/(max(py_mtf)-min(py_mtf))
    x_mtf = px_mtf[(px_mtf>= 0) & (py_mtf <= max(py_mtf))].real
    y_mtf = py_mtf[(px_mtf>= 0) & (py_mtf <= max(py_mtf))].real
    plt.plot(x_mtf, y_mtf, label='Apertura f/{}'.format(n[i]))
    plt.title(title)
    #plt.xlim(-5,55)
    plt.xlabel('Frecuencia [ciclo/mm]')
    return x_mtf, y_mtf


#%%

#Veo los datos originales
for j in range(len(n)):
    plt.figure()
    plot_perfil(df_list[0], j, 'Perfil borde - 25mm - variacion {}'.format(n[j]))
    plt.show()

#%%

#Figura de comparacion de todos los perfiles
plt.figure()
for k in range(len(n)):
    plot_perfil(df_list[0], k, '')
plt.legend(loc='upper left')
plt.xlabel('Posición [mm]')
plt.ylabel('Intensidad [escala gris]')
plt.title('Perfil borde - 25mm - 2.36V - G51')
plt.show()

#%%

plt.figure()
plot_perfil(df_list[1],0,'')
plot_perfil(df_list[1],2,'')
plot_perfil(df_list[1],4,'')
#plot_perfil(df_list[2],4,'')
plt.legend(loc='upper right')
plt.xlabel('Posición [mm]')
plt.ylabel('Intensidad [escala gris]')
plt.title('Perfil barras - 25mm - 2.36V - G51')
plt.show()

#%%
xder = np.array((df_list[0][0].iloc[:,0]))/21
yder = derivada_discreta(df_list[0],2)

plt.figure()
plot_perfil(df_list[0],2,'')
plt.plot(xder, yder,label='LSF ESF')
plt.legend(loc='upper left')
plt.xlabel('Posición [mm]')
plt.ylabel('Intensidad [escala gris]')
plt.title('Perfil borde - 25mm - f/2.8 - 2.36V - G51')
plt.show()

#%%

plt.figure()
fourier_transf(df_list[0], 1,'g')
fourier_transf(df_list[1], 0,'r')
plt.legend(loc='upper left')
plt.xlabel('Posición [mm]')
plt.ylabel('Gray scale')
plt.title('Perfil linespread - 25mm vs 35mm - 2.36V - G51')
plt.show()

#%%

#COmparacion tiempo de exposicion vs intensidad del perfil de borde
ymean_bright = []
for i in range(len(n)):
    xdata = np.array((df_list[0][i]['Distance_(mm)']))
    ydata = np.array((df_list[0][i]['Gray_Value']))
    y_bright = ydata[xdata > 1.6]
    ymean_bright.append(np.mean(y_bright))

times = np.array((13,25,19,48,100))
plt.figure()
plt.scatter(times, ymean_bright)
plt.xlabel('Tiempo de exposicion [s]')
plt.ylabel('Promedio de la intensidad - lado blanco')
plt.show()

z = np.polyfit(times, ymean_bright, 3)


#%%

#Figura de todos los mtf
xdata_mtf = []
ydata_mtf = []
df_mtf_list = []
xc = []

plt.figure()
for k in range(len(n)):
    temp_x_mtf, temp_y_mtf = plot_mtf(df_list[0], k, '')
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

plt.figure()
plt.scatter(n,Res)
plt.xlabel('# Apertura Diafragma')
plt.ylabel('Resolución (μm)')
plt.title('Resolución espacial - 25mm - variación diafragma - 2.36V - G51')
plt.show()


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
print('el ancho de banda es aprox', 100, 'micrometro')


#%%
#Lo que entendi yo es que hay que ver donde llega el limite inferior de mtf
#En este caso es de una frecuencia espacial de aprox 45, o sea 45/mm
#Por lo que la resolucion debe ser el orden de 1/frecuencia espacial que es aprox 20 micrones

x_mtf, y_mtf = plot_mtf(df_list[0], 2, 'MTF borde - 25mm {}'.format(n[1]))
x_mtf1, y_mtf1 = plot_mtf(df_list[1], 0, 'MTF borde - 25mm {}'.format(m[0]))

plt.figure()
plt.plot( x_mtf, y_mtf , label='señal original f/2.8')
plt.plot( x_mtf1, y_mtf1 , color='orange',label='señal filtrada f/2.8')
plt.scatter(x_mtf,y_mtf, color='r', label='mínimos')
plt.legend(loc='upper right')
plt.ylabel('Intensidad [escala gris]')
plt.xlabel('Frecuencia [lp/mm]')
plt.title('Perfil borde - MFT - 25mm vs 35mm')
plt.show()

#%%

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
plt.xlabel('Frecuencia [1/mm]')
plt.title('Perfil MFT - Lfocal 2')
plt.legend(loc='upper right')
plt.show()


#%%

der_dis = []
for j in range(len(n)):
    fourier_transf(df_list[1], j, 'b')
    
#%%

#NO LO TERMINE USANDO MUCHO ESTE
#Todos los perfiles y todos los mft por separado
for j in range(len(n)):
    #Veo la funcion linespread, que es la derivada de los bordes (o lo equivalente en este caso: fft)
    fourier_transf(df_list[0], j, 'Perfil borde - 25mm - diafragma {}'.format(n[j]), 'FFT borde - 25mm - Lfocal {}'.format(n[j]))
    #Veo la fft de linespread, que deberia devolver mft
    plt.figure()
    plot_mtf(df_list[0], j,'Perfil MFT - diafragma {}'.format(n[j]))
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
    #temp_xc_sine= max(temp_x_max[temp_y_max_norm <= (max(temp_y_max_norm)*0.6)])
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

#puntos = plt.ginput(2)
#puntos = np.array((puntos))

temp_x_max, temp_y_max, temp_y_max_norm = plot_max(df_list[1], 4, 0.03, '','#{}'.format(n[4]))
#%%

ydata_max_norm[0] = np.delete(ydata_max_norm[0], 3)
xdata_max[0] = np.delete(xdata_max[0], 3)
ydata_max_norm[4] = np.concatenate([ydata_max_norm[4], np.array((0.18776, 0.19056))])
xdata_max[4] = np.concatenate([xdata_max[4], np.array((10.5254, 10.72))])

xc_sine_mtf = []

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

#Si hay que introducir los max manualmente

'''
puntos = plt.ginput(6)
puntos = np.array((puntos))
max_loc_y, max_loc_x = np.concatenate((y_max, puntos[:,1])), np.concatenate((x_max, puntos[:,0]))
max_loc_y, max_loc_x = np.array((sorted(list(max_loc_y)))), np.array((sorted(list(max_loc_x))))

plt.figure()
plt.plot(x_temp, y_temp)
plt.plot(df_list[1][0]['Distance_(mm)'],filtered_y)
plt.scatter(max_loc_x, max_loc_y)
'''

#%%

folderpath = r'C:\Users\luo\OneDrive\文档\Labo 6y7\15_06_22\10_47_48\diafragmas_2.34V_Ganancia50\\'
folderpath_sine = r'C:\Users\luo\OneDrive\文档\Labo 6y7\15_06_22\10_47_48\diafragmas_2.34V_Ganancia50\\'

#guardo los datos en un archivo txt
file=open(folderpath + 'Analisis_res.txt','a')
file.write('%10.6f' % (Res[0])+',')
file.write('%10.6f' % (Res[1])+',')
file.write('%10.6f' % (Res[2])+',')
file.write('%10.6f' % (Res[3])+',')
file.write('%10.6f' % (Res[4])+'\n')
file.close()

#%%
file_sine=open(folderpath_sine + 'Analisis_res_sine.txt','a')
file_sine.write('%10.6f' % (Res_sine[0])+',')
file_sine.write('%10.6f' % (Res_sine[1])+',')
file_sine.write('%10.6f' % (Res_sine[2])+',')
file_sine.write('%10.6f' % (Res_sine[3])+',')
file_sine.write('%10.6f' % (Res_sine[4])+'\n')
file_sine.close()

file_sine0=open(folderpath_sine + 'Analisis_xic_sine.txt','a')
file_sine0.write('%10.6f' % (popt_sine[0])+',')
file_sine0.write('%10.6f' % (popt_sine[1])+',')
file_sine0.write('%10.6f' % (popt_sine[2])+',')
file_sine0.write('%10.6f' % (popt_sine[3])+',')
file_sine0.write('%10.6f' % (popt_sine[4])+'\n')
file_sine0.close()

#%%

data_sine = np.loadtxt(folderpath_sine + 'Analisis_res_sine.txt',dtype=float,delimiter = ',',skiprows= 0)
data_borde = np.loadtxt(folderpath + 'Analisis_res.txt',dtype=float,delimiter = ',',skiprows= 0)

plt.figure()
plt.scatter(times, data_borde[0], label='Centro',color='g')
plt.scatter(times, data_borde[1], label='LU',color='cornflowerblue')
plt.scatter(times, data_borde[2], label='RD',color='orange')
plt.legend(loc='upper right')
plt.xlabel('Tiempo de exposicion [s]')
plt.ylabel('Resolucion [μs]')
plt.title('Comparacion de resoluciones en diferentes areas')
plt.show()

plt.figure()
plt.scatter(times, data_sine[0], label='Centro',color='g')
plt.scatter(times, data_sine[1], label='LD',color='cornflowerblue')
plt.scatter(times, data_sine[2], label='RU',color='orange')
plt.legend(loc='upper right')
plt.xlabel('Tiempo de exposicion [s]')
plt.ylabel('Resolucion [μs]')
plt.title('Comparacion de resoluciones en diferentes areas')
plt.show()



