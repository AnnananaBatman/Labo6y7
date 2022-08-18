# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 20:39:35 2022

@author: luo
"""
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
import cv2
from scipy.optimize import curve_fit

get_ipython().run_line_magic('matplotlib', 'qt5')
plt.rcParams.update({'font.size': 14})

#%%

path = r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\dark_51G_-15c'
with open(r"D:\Imagenes Labo6\light_2-31V_51G_20_07_22\dark_51G_-15c\output_dark.txt", "w") as a:
    for path, subdirs, files in os.walk(path):
        for filename in files:
            f = os.path.join(path, filename)
            a.write(str(f) + os.linesep) 

tiempos = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300]

#%%

file_path = r"D:\Imagenes Labo6\light_2-31V_51G_20_07_22\dark_51G_-15c\output_dark.txt"
temp_df = pd.read_csv(file_path)
temp_df_1 = temp_df[temp_df[temp_df.columns[0]].str.contains('0000')]
temp_df_csv = temp_df[temp_df[temp_df.columns[0]].str.contains('txt')]
temp_df_1.to_csv(r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\dark_51G_-15c\filenames_dark_0000.csv', header=False, index=False)
temp_df_csv.to_csv(r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\dark_51G_-15c\filenames_dark_csv.csv', header=False, index=False)

#%%

file_path =r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\dark_51G_-15c\filenames_dark_0000.csv'
file_path_csv =r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\dark_51G_-15c\filenames_dark_csv.csv'
temp_df = pd.read_csv(file_path, header=None)
temp_df_csv = pd.read_csv(file_path_csv, header=None)
for l in range(len(temp_df)):
    src_path = temp_df[0][l]
    src_path_csv = temp_df_csv[0][l]
    char1, char2 = 'c\\', 's\\'
    exp_time = float(src_path_csv[src_path_csv.find(char1)+2 : src_path_csv.find(char2)])
    dst_path = r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\dark_51G_-15c\imagenes desechadas\imagenes0000_{}.tif'.format(exp_time)
    dst_path_csv = r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\dark_51G_-15c\imagenes desechadas\information_{}.txt'.format(exp_time)
    shutil.move(src_path, dst_path)
    shutil.move(src_path_csv, dst_path_csv)

#%%

out = ['light', 'dark']
for j in range(2):
    file_path =r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\output_{}.txt'.format(out[j])
    temp_df = pd.read_csv(file_path)
    temp_df_1 = temp_df.drop(temp_df[temp_df[temp_df.columns[0]].str.contains('txt')].index)
    temp_df_2 = temp_df_1.drop(temp_df[temp_df[temp_df.columns[0]].str.contains('0000')].index) 
    temp_df_2.to_csv(r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\filenames_{}.csv'.format(out[j]), header=False, index=False)

#%%

filenames_light = pd.read_csv(r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\filenames_light.csv',header=None)
filenames_dark = pd.read_csv(r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\filenames_dark.csv',header=None)

for k in range(len(filenames_light)):
    src_path_light = filenames_light.iloc[:,0][k]
    char1, char2 = 'c\\', 's\\'
    exp_time = float(src_path_light[src_path_light.find(char1)+2 : src_path_light.find(char2)])
    char3, char4 = 'y_', '.tif'
    name = float((src_path_light[src_path_light.find(char3)+2 : src_path_light.find(char4)]))
    dst_path_light = r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\2.31V_51G_-15c\allimg\img_{}'.format(exp_time) + '_{}.tif'.format(name)
    shutil.move(src_path_light, dst_path_light)
    src_path_dark = filenames_dark.iloc[:,0][k]
    char1, char2 = 'c\\', 's\\'
    exp_time = float(src_path_dark[src_path_dark.find(char1)+2 : src_path_dark.find(char2)])
    char3, char4 = 'y_', '.tif'
    name = float((src_path_dark[src_path_dark.find(char3)+2 : src_path_dark.find(char4)]))
    dst_path_dark = r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\dark_51G_-15c\allimg\img_{}'.format(exp_time) + '_{}.tif'.format(name)
    shutil.move(src_path_dark, dst_path_dark)
    
#%%
for k in range(len(filenames_light)):
    light = cv2.imread(filenames_light.iloc[:,0][k])
    dark = cv2.imread(filenames_dark.iloc[:,0][k])
    subtracted = cv2.subtract(light,dark)
    char1, char2 = 'c\\', 's\\'
    exp_time = float(filenames_light.iloc[:,0][k][filenames_light.iloc[:,0][k].find(char1)+2 : filenames_light.iloc[:,0][k].find(char2)])
    char3, char4 = 'y_', '.tif'
    name = float(filenames_light.iloc[:,0][k][filenames_light.iloc[:,0][k].find(char3)+2 : filenames_light.iloc[:,0][k].find(char4)])
    cv2.imwrite(r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\subtracted\{}'.format(int(exp_time))+'_{}.tif'.format(int(name)), subtracted)
    cropped = subtracted[200:1800, 350:2500]
    cv2.imwrite(r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\subtractedncropped\200to1800_350to2500_{}'.format(int(exp_time))+'_{}.tif'.format(int(name)), cropped)
 
#%%

for i in range(26):
    img = cv2.imread()
cropped = subtracted[200:1800, 350:2500]
cv2.imwrite(r'D:\Imagenes Labo6\light_2-31V_51G_20_07_22\subtractedncropped\200to1800_350to2500_{}'.format(int(exp_time))+'_{}.tif'.format(int(name)), cropped)


#%%

from PIL import Image, ImageChops 
    
# creating a image1 object 
im1 = Image.open(r"D:\Imagenes Labo6\light_2-31V_51G_20_07_22\dark_51G_-15c\60s\2022-07-20-1945_3-U-L-Deepsky_0001.tif") 
    
# creating a image2 object 
im2 = Image.open(r"D:\Imagenes Labo6\light_2-31V_51G_20_07_22\2.31V_51G_-15c\60s\2022-07-20-1831_5-U-L-Deepsky_0001.tif") 
    
# applying subtract method 
im3 = ImageChops.add(im1, im2, scale = 1.0, offset = 0) 
    
im3.show()

#%%

#sacar todas las variables (o sea histograms y despues means): primero de los dos sucesivos og (cropped y subtracted), despues de los means 

def plot_histogram(m, i):
    #df_list = []
    #df_hist_list = []
    n=[10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300]
    img_path ='D:\\Imagenes Labo6\\light_2-31V_51G_20_07_22\\allsubtracted\\cropped\\mean_img\\img_{}'.format(n[m])+'.0_{}.0.tif_.tif_.tif'.format(i)
    img = plt.imread(img_path)
    freq, bins, a = plt.hist(img.ravel(), bins=256, range=(0, 65528), label='{}s'.format(n[m])) #calculating histogram
    plt.xlabel('Gray Values')
    plt.ylabel('Frequency')
    plt.title('Histograma - 2.31V - {}s'.format(n[m]) + '_{}'.format(i))
    temp_df_hist = pd.DataFrame(np.array((freq, bins[0:len(bins)-1], np.full((256),n[m]), np.full((256),i) )).transpose(),columns=['Frequency', 'Gray Value', 'Exp Time', 'Orden'])
    temp_df_hist.to_csv(r'D:\\Imagenes Labo6\\light_2-31V_51G_20_07_22\\allsubtracted\\cropped\\mean_img\\hist_filenames_{}'.format(n[m])+ '_{}.csv'.format(i))


#%%
for m in range(len(tiempos)):
    plot_histogram(m,1)

#%%
plt.figure()
plot_histogram(3,1)
plot_histogram(5,1)
plot_histogram(9,1)
plt.legend(loc='upper center')
plt.show()
#%%

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

df_list_1 = []
df_list_2 = []

for m in range(len(tiempos)):
    n= [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300]
    file_path_1 = r'D:\\Imagenes Labo6\\light_2-31V_51G_20_07_22\\allsubtracted\\cropped\\hist_filenames_{}'.format(n[m])+ '_{}.csv'.format(1)
    temp_df_his_1 = pd.read_csv(file_path_1)
    df_list_1.append(temp_df_his_1)

for m in range(len(tiempos)):
    n= [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300]
    file_path_2 = r'D:\\Imagenes Labo6\\light_2-31V_51G_20_07_22\\allsubtracted\\cropped\\hist_filenames_{}'.format(n[m])+ '_{}.csv'.format(2)
    temp_df_his_2 = pd.read_csv(file_path_2)
    df_list_2.append(temp_df_his_2)
        


#%%
ym = []
yd = []
exp = []

for i in range(len(df_list_2)):
    temp_ym, temp_yd = stats(df_list_2,i)
    ym.append(temp_ym)
    yd.append(temp_yd)
    temp_exp = int(np.array((df_list_2[i]['Exp Time']))[0])
    exp.append(temp_exp)

df_to_save = pd.DataFrame(np.array((exp,ym,yd)).transpose())
df_to_save.to_csv(r'D:\\Imagenes Labo6\\light_2-31V_51G_20_07_22\\allsubtracted\\cropped\\hist_filenames_{}.csv'.format(2))

#%%

#para las imagenes del promedio

for m in range(len(tiempos)):
    plot_histogram(m,1)

#%%

df_list_mean = []

for m in range(len(tiempos)):
    n= [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300]
    file_path_mean = r'D:\\Imagenes Labo6\\light_2-31V_51G_20_07_22\\allsubtracted\\cropped\\mean_img\\hist_filenames_{}'.format(n[m])+ '_{}.csv'.format(1)
    temp_df_his_mean = pd.read_csv(file_path_mean)
    df_list_mean.append(temp_df_his_mean)
    
for i in range(len(df_list_mean)):
    temp_ym, temp_yd = stats(df_list_mean,i)
    ym.append(temp_ym)
    yd.append(temp_yd)
    temp_exp = int(np.array((df_list_mean[i]['Exp Time']))[0])
    exp.append(temp_exp)

df_to_save = pd.DataFrame(np.array((exp,ym,yd)).transpose())
df_to_save.to_csv(r'D:\\Imagenes Labo6\\light_2-31V_51G_20_07_22\\allsubtracted\\cropped\\hist_filenames_mean.csv')

#%%


data_mean = pd.read_csv(r'D:\\Imagenes Labo6\\light_2-31V_51G_20_07_22\\allsubtracted\\cropped\\hist_filenames_mean.csv')
data_1 = pd.read_csv(r'D:\\Imagenes Labo6\\light_2-31V_51G_20_07_22\\allsubtracted\\cropped\\hist_filenames_1.csv')
data_2 = pd.read_csv(r'D:\\Imagenes Labo6\\light_2-31V_51G_20_07_22\\allsubtracted\\cropped\\hist_filenames_2.csv')

uy = np.array((data_mean.iloc[:,2]))
uy_1 = np.array((data_1.iloc[:,2]))
uy_2 = np.array((data_2.iloc[:,2]))
sigmay = np.array((data_mean.iloc[:,3]))
dif_uy = abs(uy_2-uy_1)

#%%

plt.figure()
plot_histogram(3,1)
plot_histogram(5,1)
plot_histogram(9,1)
plt.axvline(x=uy[3], color='r',label='promedio 60s')
plt.axvline(x=uy[5], color='purple',label='promedio 100s')
plt.axvline(x=uy[9], color='blue',label='promedio 180s')
plt.axvline(x=uy[3]+sigmay[3], color='r',linestyle='dashed')
plt.axvline(x=uy[3]-sigmay[3], color='r',linestyle='dashed')
plt.legend(loc='upper right')
plt.show()

#%%

def linealidad(x,a,b):
    return a*x+b

#%%


plt.figure()
plt.title('Estabilidad - 2.31V')
plt.xlabel('Promedio Gray Value (2 imagenes)')
plt.ylabel('Diferencia de los promedios Gray Value')
plt.scatter(uy, dif_uy)
plt.show()

#%%

py = sigmay**2

popt0, pcov0 = curve_fit(linealidad, uy[:7],py[:7] , maxfev= 5000)
perr0 = np.sqrt(np.diag(pcov0)) # errores de 1 sigma
px0 = np.linspace(uy[0], uy[7],300)

plt.figure()
plt.title('Curva de transferencia de fotones - 2.31V')
plt.xlabel('Promedio Gray Value')
plt.ylabel('Desviación Gray Value$^2$')
plt.scatter(uy, sigmay**2)
plt.plot(px0,linealidad(px0,*popt0),color='orange',label='Ajuste lineal')
plt.scatter(uy[8],py[8],color='r',label='Saturación')
plt.legend(loc='upper left')
plt.show()

#%%

#SNR

plt.figure()
plt.title('Signa to Noise Ratio - 2.31V')
plt.xlabel('Tiempo de exposición (s)')
plt.ylabel('SNR')
plt.scatter(tiempos, uy/sigmay)
plt.show()


#%%
plt.figure()
plt.title('Curva característica - 2.31V')
plt.ylabel('Promedio Gray Value')
plt.xlabel('Tiempo de exposición (s)')
plt.scatter(tiempos, uy)
plt.show()

#%%

popt, pcov = curve_fit(linealidad, tiempos[:10], uy[:10], maxfev= 5000)
perr = np.sqrt(np.diag(pcov)) # errores de 1 sigma

px = np.linspace(tiempos[0], tiempos[10],300)

plt.figure()
plt.plot(px,linealidad(px,popt),color='orange',label='Ajuste lineal')
plt.scatter(tiempos, uy)
plt.title('Curva característica - 2.31V')
plt.ylabel('Promedio Gray Value')
plt.xlabel('Tiempo de exposición (s)')
plt.legend(loc='upper left')
plt.show()

