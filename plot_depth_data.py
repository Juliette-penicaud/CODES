# 31/05/23 : Je créée les plots pour visualiser les différences éventuelles de profondeur du fichier recap des prof

import pandas as pd
import numpy as np
import sys, os
import cmocean
from datetime import datetime
import scipy.signal as signal
import re
from openpyxl import load_workbook
import gsw
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


# VARIABLE AND PARAMETER
year = '2022'
list_month = ['June', 'August', 'Octobre']
i = 0  # 0 1 2 a voir pour faire une boucle si besoin
month = list_month[i]

rep = '/home/penicaud/Documents/Data/Survey_'+month +'/'
file_depth = rep + 'diff_depth_surface_allinstrum_' + month + '.xlsx'

# Load dataframe
df_depth = pd.read_excel(file_depth)
df_depth = df_depth.rename(columns={'index':'Stations','height up' : 'LISST up','height down' : 'LISST down', 'depth ADCP':'ADCP', 'depth CTD':'CTD'})
df_depth = df_depth.set_index('Stations')
df_depth = df_depth.drop([df_depth.columns[0]], axis = 1)
df_depth2 = -df_depth.copy()
df_depth2 = df_depth2[['LISST up', 'LISST down', 'ADCP', 'CTD']]
#plot transect par transect, avec % marée, distance à embouchure et phi
fontsize = 10
dict_month = {'June' : {'nrows':6, 'nbsta': 10 ,'ylim': -15},
              'August' : {'nrows':6, 'nbsta' : 11 , 'ylim' : -20},
              'Octobre' : {'nrows':6, 'nbsta' : 14, 'ylim' : -20} }

# X rows of 10 stations
ylim = dict_month[month]['ylim']
x = dict_month[month]['nbsta']
f = 0
if f :
    for i in range(dict_month[month]['nrows']):
        print(i)
        fig, ax = plt.subplots()
        ax = df_depth2[i*x :i*x + x].plot(kind='bar', rot=0,
                                      figsize=(10, 6),
                                      colormap=cmocean.cm.tempo, alpha=0.85,
                                      edgecolor='grey', zorder=10)
        ax.set_ylim(ylim, 0)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        # ax.set_yticks(np.arange(ylim, 0, 1))
        # ax.minorticks_on()
        ax.grid(axis='y', which='both', zorder=0)
        plt.ylabel('Depth (m)')
        plt.suptitle(month)
        plt.savefig('Diff_depth_allinstrum_' + month + '_' + str(i))

fig, axs = plt.subplots(nrows = 3 , ncols = 2, sharex=False , sharey= True, figsize = (25,10))
fig.suptitle(month, fontsize=fontsize+2)
axs[0,0].set_ylabel('Depth (m)', fontsize=fontsize)
axs[1,0].set_ylabel('Depth (m)', fontsize=fontsize)
axs[2,0].set_ylabel('Depth (m)', fontsize=fontsize)
axs[2,0].set_xlabel('Stations', fontsize=fontsize)
axs[2,1].set_xlabel('Stations', fontsize=fontsize)

list_ax1 = [0,1,2,0,1,2]
list_ax2 = [0,0,0,1,1,1]
for i in range(dict_month[month]['nrows']):
    print(i,list_ax1[i],list_ax2[i] )
    ax = axs[list_ax1[i],list_ax2[i]]
    ax.set_ylim(ylim, 0)
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.grid(axis='y', which='both', zorder=0)
    plt.subplots_adjust(left=0.06, bottom=None, right=0.98, top=0.92, wspace=0.05, hspace=0.25)
    df_depth2[i*x :i*x + x].plot(ax = ax , kind='bar', rot=0,
                                  figsize=(10, 6),
                                  colormap=cmocean.cm.tempo, alpha=0.85,
                                  edgecolor='grey', zorder=10, legend=False)

    ax.xaxis.label.set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-3)

ax.legend(fontsize = fontsize-3, loc = 'best')
fig.savefig('Diff_depth_allinstrum_' + month)

sys.exit(1)

fig, axs = plt.subplots(ncols=2, nrows=2)
fig.title = month
axs[1,0].set_xlabel('Distance to the mouth (m)', fontsize=fontsize)
axs[1,1].set_xlabel('Distance to the mouth (m)', fontsize=fontsize)
axs[0,0].set_ylabel('$\Phi$ ($J.m^{-3}$)', fontsize=fontsize)
axs[1,0].set_ylabel('$\Phi$ ($J.m^{-3}$)', fontsize=fontsize)
axs.set_xlim(-1500,6200)
axs.set_ylim(0,600)
i=0
for t in ['T1', 'T2', 'T3', 'T4']:
    condition = (data['transect'] == t)
    if i == 0 :
        ax=axs[0,0]
    if i == 1 :
        ax=axs[0,1]
    if i == 2 :
        ax=axs[1,0]
    if i == 3 :
        ax=axs[1,1]
    ax.scatter(data['distance'].loc[condition], data['simpson'].loc[condition],marker='x', alpha=0.8,  color=dict_transect[t] )
    i=i+1
fig.savefig('test_simpson')