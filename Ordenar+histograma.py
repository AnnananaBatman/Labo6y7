# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 20:51:49 2022

@author: luo
"""
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'qt5')
plt.rcParams.update({'font.size': 14})
#%%

n=[-15, 5, 10, 15]

#%%

for i in range(len(n)):
    path = r'D:\Imagenes Labo6\nuevas_dark_51G_04_08_22\nuevo_{}c'.format(n[i])
    with open(r"D:\Imagenes Labo6\nuevas_dark_51G_04_08_22\output_{}.txt".format(n[i]), "w") as a:
        for path, subdirs, files in os.walk(path):
            for filename in files:
                f = os.path.join(path, filename)
                a.write(str(f) + os.linesep) 

#%%

for j in range(len(n)):
    file_path =r'D:\Imagenes Labo6\nuevas_dark_51G_04_08_22\output_{}.txt'.format(n[j])
    temp_df = pd.read_csv(file_path, skiprows=1)
    temp_df_1 = temp_df.drop(temp_df[temp_df[temp_df.columns[0]].str.contains('txt')].index)
    temp_df_2 = temp_df_1.drop(temp_df[temp_df[temp_df.columns[0]].str.contains('0000')].index) 
    temp_df_2.to_csv(r'D:\Imagenes Labo6\nuevas_dark_51G_04_08_22\filenames_{}.csv'.format(n[j]), header=False, index=False)

#%%

for j in range(len(n)):
    file_path =r'D:\Imagenes Labo6\dark_2-30V_51G_12_07_22\output_{}.txt'.format(n[j])
    temp_df = pd.read_csv(file_path, skiprows=1)
    #temp_df_1 = temp_df[(temp_df[temp_df.columns[0]].str.contains('txt')) & (temp_df[temp_df.columns[0]].str.contains('0000'))]
    temp_df_1 = temp_df[temp_df[temp_df.columns[0]].str.contains('0000')]
    temp_df_1.to_csv(r'D:\Imagenes Labo6\dark_2-30V_51G_12_07_22\filenames_0000_{}.csv'.format(n[j]), header=False, index=False)


#%%


#mover la imagen 0000 a otro lado

for m in range(len(n)):
    file_path =r'D:\Imagenes Labo6\dark_2-30V_51G_12_07_22\filenames_0000_{}.csv'.format(n[m])
    temp_df = pd.read_csv(file_path, header=None)
    for l in range(len(temp_df)):
        src_path = temp_df[0][l]
        dst_path = r'D:\Imagenes Labo6\dark_2-30V_51G_12_07_22\imagenes_0000_{}'.format(n[m])+'\{}.tif'.format(temp_df[0][l])
        shutil.move(src_path, dst_path)

#%%

#sacar histograma de las imagenes no nulas




def plot_histogram(m):
    df_list = []
    n=[-15, 5, 10, 15]
    file_path =r'D:\Imagenes Labo6\nuevas_dark_51G_04_08_22\filenames_{}.csv'.format(n[m])
    temp_df = pd.read_csv(file_path, header=None)
    df_hist_list = []
    for l in range(len(temp_df)):
        img_path = temp_df[0][l]
        char1, char2 = 'c\\', 's\\'
        exp_time = float(img_path[img_path.find(char1)+2 : img_path.find(char2)])
        img = plt.imread(img_path)
        plt.figure()
        freq, bins, a = plt.hist(img.ravel(), bins=256, range=(0, 65528)) #calculating histogram
        plt.xlabel('Gray Values')
        plt.ylabel('Frequency')
        plt.title('Histograma - corriente oscura - {}C'.format(n[m]) + ' - {}s'.format(exp_time))
        plt.show()
        temp_df_hist = pd.DataFrame(np.array((freq, bins[0:len(bins)-1], np.full((256),exp_time), np.full((256),n[m]) )).transpose(),columns=['Frequency', 'Gray Value', 'Exp Time', 'Temperature'])
        df_hist_list.append(temp_df_hist)
    df_list.append(df_hist_list)
    pd.concat(df_list[0]).to_csv(r'D:\Imagenes Labo6\nuevas_dark_51G_04_08_22\hist_filenames_{}.csv'.format(n[m]))
    df_hist_list = 0
    df_list = 0
    return df_list


#%%

for m in range(len(n)):
    plot_histogram(m)
    

#%%

#Promedio de 1s

for i in range(len(n)):
    file_path =r'D:\Imagenes Labo6\nuevas_dark_51G_04_08_22\hist_filenames_{}.csv'.format(n[i])
    df_1s = pd.read_csv(file_path)
    df_list_1s = [g for _,g in df_1s.groupby('Exp Time')]
    temp_df_list_1s = [df_list_1s[0].iloc[i:i + 256] for i in range(0, len(df_list_1s[0]), 256)]
    df_GV_1s = pd.DataFrame(np.array((temp_df_list_1s[0].iloc[:,1],temp_df_list_1s[1].iloc[:,1]
                                       ,temp_df_list_1s[2].iloc[:,1],temp_df_list_1s[3].iloc[:,1]
                                       ,temp_df_list_1s[4].iloc[:,1],temp_df_list_1s[5].iloc[:,1]
                                       ,temp_df_list_1s[6].iloc[:,1],temp_df_list_1s[7].iloc[:,1]
                                       ,temp_df_list_1s[8].iloc[:,1],temp_df_list_1s[8].iloc[:,1])).transpose())
    temp_df_list_1s[0]['Frequency'] = np.array((df_GV_1s.mean(axis=1)))
    dfh_1s = temp_df_list_1s[0].set_index('Unnamed: 0')
    dfh_1s.to_csv(r'D:\Imagenes Labo6\nuevas_dark_51G_04_08_22\mean_hist_{}_1s.csv'.format(n[i]))

#%%

#Promedio de 5s

for i in range(len(n)):
    file_path =r'D:\Imagenes Labo6\nuevas_dark_51G_04_08_22\hist_filenames_{}.csv'.format(n[i])
    df_5s = pd.read_csv(file_path)
    df_list_5s = [g for _,g in df_5s.groupby('Exp Time')]
    temp_df_list_5s = [df_list_5s[1].iloc[i:i + 256] for i in range(0, len(df_list_5s[1]), 256)]
    df_GV_5s = pd.DataFrame(np.array((temp_df_list_5s[0].iloc[:,1],temp_df_list_5s[1].iloc[:,1]
                                       ,temp_df_list_5s[2].iloc[:,1],temp_df_list_5s[3].iloc[:,1])).transpose())
    temp_df_list_5s[1]['Frequency'] = np.array((df_GV_5s.mean(axis=1)))
    dfh_5s = temp_df_list_5s[1].set_index('Unnamed: 0')
    dfh_5s.to_csv(r'D:\Imagenes Labo6\nuevas_dark_51G_04_08_22\mean_hist_{}_5s.csv'.format(n[i]))

#%%

#Resto de los promedios

for i in range(len(n)):
    file_path =r'D:\Imagenes Labo6\nuevas_dark_51G_04_08_22\hist_filenames_{}.csv'.format(n[i])
    df_exp = pd.read_csv(file_path)
    df_list_exp = [g for _,g in df_exp.groupby('Exp Time')]
    for j in range(2,len(df_list_exp)):
        temp_df_list = [df_list_exp[j].iloc[i:i + 256] for i in range(0, len(df_list_exp[j]), 256)]
        df_GV_exp = pd.DataFrame(np.array((temp_df_list[0].iloc[:,1],temp_df_list[1].iloc[:,1])).transpose())
        temp_df_list[0]['Frequency'] = np.array((df_GV_exp.mean(axis=1)))
        dfh_exp = temp_df_list[0].set_index('Unnamed: 0')
        dfh_exp.to_csv(r'D:\Imagenes Labo6\nuevas_dark_51G_04_08_22\mean_hist_{}'.format(n[i])+'_{}s.csv'.format(int(dfh_exp['Exp Time'][0])))

#%%


