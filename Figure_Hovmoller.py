#18/07/2022 Graph évolution des T, S, turbidity focntion des stations / temps
import csv
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
#from sklearn.linear_model import LinearRegression, TheilSenRegressor
#from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
#import csv
import sys, os, glob
#import xarray as xr
#import matplotlib.colors as mcolors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
#import gsw
import random
import pickle
from itertools import zip_longest

year='2022'
month='06'

fontsize=10
cmap=plt.cm.jet

variable=['T', 'S', 'turb']

if month=='06':
    #transect='T1' #TRANSECT TO CHANGE #T1 T2 T3 T4 or fixed
    # T1 : S1 à S11
    # T2 : S12 à S17
    # T3 : S18 S28
    # T4 : S29 S39
    transect=['T1', 'T2', 'T3', 'T4']
    nb_bin=45
    y=np.linspace(0,140,5)
    prof=['-14', '-10.5', '-7', '-3.5' ,'0']
    #transect=['fixed']
elif month=='08':
    transect=['TA1', 'TA2', 'TA3', 'TA4', 'SF1', 'SF_24']
    nb_bin=200 #180 or 200
    if nb_bin==200:
        y=np.linspace(0,200,5)
        prof=['-20', '-15', '-10', '-5' ,'0']
    elif nb_bin==180:
        y=np.linspace(0,180,5)
        prof=['-18','-13.5', '-9', '-4.5' ,'0']
elif month=='10':
    transect=['TO1']
else :
    print('PB MONTH')
    sys.exit(1)

#Loop to open and draw the figures
#Figure of 3 subplots : hovmoller, T S and Turb, il faudra faire N2

for t in transect :
    if t == 'T1' :
        day='16'
        stations=['S1', 'S2', 'S3','S4', 'S5', 'S6', 'S7' , 'S8', 'S9', 'S10', 'S11']
        x=np.arange(0,11,1)
    elif t=='T2' :
        day='16'
        stations=['S12', 'S14', 'S15', 'S16', 'S17']
        x=np.arange(0,5,1)
    elif t=='T3' :
        day='17'
        stations=['S18', 'S19', 'S20','S21', 'S22', 'S23', 'S24' , 'S25', 'S26', 'S27', 'S28']
        x=np.arange(0,11,1)
    elif t=='T4' :
        day='17'
        stations=['S29', 'S30','S31', 'S32', 'S33', 'S34' , 'S35', 'S37', 'S38', 'S39']
        x=np.arange(0,10,1)
    elif t=='fixed':
        day='18'
        #stations=['1']
        x=np.arange(0,25,1)
        y=np.linspace(0,10,5)
        prof=['-10', '-5', '0']

    elif t=='TA1':
        day='10'
        sta=range(1,6)
        stations=[]
        for s in sta :
            stations.append('A' + str(s))
        x=np.arange(0,len(stations),1)
    elif t=='TA2':
        day='10'
        sta=range(6,13)
        stations=[]
        for s in sta :
            stations.append('A' + str(s))
        x=np.arange(0,len(stations),1)
    elif t=='TA3':
        day='11'
        sta=range(26,41)
        stations=[]
        for s in sta :
            stations.append('A' + str(s))
        x=np.arange(0,len(stations),1)
    elif t=='TA4':
        day='12'
        sta=range(41,48)
        stations=[]
        for s in sta :
            stations.append('A' + str(s))
        x=np.arange(0,len(stations),1)
    elif t=='SF1':
        day='10'
        sta=range(13,26)
        stations=[]
        for s in sta :
            if s == 17:
                print('break 17')
                continue
            else :
                stations.append('A' + str(s))
        #stations.append('A25-1')
        x=np.arange(0,len(stations),1)
    elif t=='SF_24':
        day='13'
        sta=range(1,39)
        stations=[]
        for s in sta :
            if s in [5, 8, 10, 14, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:  # 14 exists but not on <ctd log
                print('break AF')
                continue
            else:
                stations.append('AF' + str(s))
        x=np.arange(0,len(stations),1)
    else:
        print('PB int the stations')
    print('stations', stations)
    print('x', x)

    i=0

    print('max x', np.max(x))

    fig, axs = plt.subplots(nrows=3)
    fig.suptitle(day + '/' + month + '/' + year + ', ' + t)
    #axs.set_ylim(-10, 0)
    #axs.set_ylabel('Depth (m)', fontsize=fontsize)
    # set the spacing between subplots
    #majoryticks = np.linspace(-140, 0, 3)
    #axs.set_yticklabels(majoryticks, minor=False)
    #axs.set_xticklabels(stations , minor=True )
    plt.subplots_adjust(left=0.11,
                        bottom=0.1,
                        right=0.95,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    #axs.set_xlabel(stations, fontsize=fontsize)  # ('Conservative Temperature (°C)', fontsize=fontsize)

    if t=='SF_24':
        fontsize2=6.5
    else :
        fontsize2=fontsize-2

    i=0
    v='S'
    vmin=0
    vmax=31
    disp='Salinity (PSU)'

    file = 'Tab_' + v + '_' + day + month + year + '_' + t
    if nb_bin==200:
        file=file+'_20m'
    file=file+ '.csv'
    tab = pd.read_csv(file, header=None)
    tab = pd.DataFrame(data=tab)
    tab = tab.T  # need to transpose because the files are recorded by rows
    print(tab)
    tab = tab.reindex(index=tab.index[::-1])  # Put the values upsidedown tto have the Nan in the 0 rows, and to plot the values at same surface
    print('tab', np.shape(tab))

    # 0 Temperature, 1, Salinity, 2 Turbidity
    ax = axs[i]
    # ax.set_ylim(-10, 0)
    # ax2 = ax.twiny()
    ax.set_ylabel('Depth (m)', fontsize=fontsize)
    ax.set_ylim(0,100)
    ax.set_xlim(0, np.max(x)+1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels=prof)
    ax.set_xticks(ticks=x)#labels=stations)
    ax.set_xticklabels(stations, fontsize=fontsize2)#, minor=True)
    p1 = ax.pcolor(tab, cmap=cmap, vmin=vmin, vmax=vmax)  # , label=disp)
    cbar = plt.colorbar(p1, label=disp, ax=ax)  # , ticks=1)#ax=ax
    cbar.ax.tick_params(labelsize=8)
    # majoryticks=np.linspace(-140,0,3)
    # minoryticks=[-5,5,15,25]
    # ax.set_yticks( minoryticks, minor=True )
    # ax.set_yticks(np.arange(140))
    # ax.set_yticklabels( majoryticks, minor=False )
    # majorxticks=[90,140]
    # minorxticks=[100,110,120,130]
    # ax.set_xticklabels(stations)#, minor=True )
    # aa.set_xticks( majorxticks, minor=False )
    # aa.tick_params(axis='both', which='major', labelsize=8)

    i=i+1
    v='T'
    vmin=28 #27 for june
    vmax=32 #31 for june
    disp='Temperature (°C)'

    file = 'Tab_' + v + '_' + day + month + year + '_' + t
    if nb_bin==200:
        file=file+'_20m'
    file=file+ '.csv'
    tab = pd.read_csv(file, header=None)
    tab = pd.DataFrame(data=tab)
    tab = tab.T  # need to transpose because the files are recorded by rows
    print('tab i ', tab, i)
    tab = tab.reindex(index=tab.index[
                            ::-1])  # Put the values upsidedown tto have the Nan in the 0 rows, and to plot the values at same surface
    print('tab', tab)

    # 0 Temperature, 1, Salinity, 2 Turbidity
    ax = axs[i]
    # ax.set_ylim(-10, 0)
    # ax2 = ax.twiny()
    ax.set_ylabel('Depth (m)', fontsize=fontsize)
    ax.set_ylim(0,100)
    ax.set_xlim(0, np.max(x)+1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels=prof)
    ax.set_xticks(ticks=x)#, labels=stations)
    ax.set_xticklabels(stations, fontsize=fontsize2)#, minor=True)
    p1 = ax.pcolor(tab, cmap=cmap, vmin=vmin, vmax=vmax)  # , label=disp)
    cbar = plt.colorbar(p1, label=disp, ax=ax)  # , ticks=1)#ax=ax
    cbar.ax.tick_params(labelsize=8)
    # majoryticks=np.linspace(-140,0,3)
    # minoryticks=[-5,5,15,25]
    # ax.set_yticks( minoryticks, minor=True )
    # ax.set_yticks(np.arange(140))
    # ax.set_yticklabels( majoryticks, minor=False )
    # majorxticks=[90,140]
    # minorxticks=[100,110,120,130]
    # ax.set_xticklabels(stations)#, minor=True )
    # aa.set_xticks( majorxticks, minor=False )
    # aa.tick_params(axis='both', which='major', labelsize=8)

    i=i+1
    v='turb'
    vmin=0
    vmax=250 #200 for june
    disp='Turbidity (FTU)'
    print('Variable', v)
    #load the tab
    file = 'Tab_' + v + '_' + day + month + year + '_' + t
    if nb_bin==200:
        file=file+'_20m'
    file=file+ '.csv'
    tab=pd.read_csv(file, header=None)
    tab=pd.DataFrame(data=tab)
    tab=tab.T #need to transpose because the files are recorded by rows
    print('tab i ' , tab, i)
    tab=tab.reindex(index=tab.index[::-1]) #Put the values upsidedown tto have the Nan in the 0 rows, and to plot the values at same surface
    print('tab', tab)

    #0 Temperature, 1, Salinity, 2 Turbidity
    ax = axs[i]
    # ax.set_ylim(-10, 0)
    # ax2 = ax.twiny()
    ax.set_ylabel('Depth (m)', fontsize=fontsize)
    ax.set_ylim(0,100)
    ax.set_xlim(0, np.max(x)+1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels=prof)
    ax.set_xticks(ticks=x)#, labels=stations)
    ax.set_xticklabels(stations, fontsize=fontsize2)#, minor=True)
    p1 = ax.pcolor(tab, cmap=cmap, vmin=vmin, vmax=vmax)  # , label=disp)
    cbar = plt.colorbar(p1, label=disp, ax=ax)  # , ticks=1)#ax=ax
    cbar.ax.tick_params(labelsize=8)
    #majoryticks=np.linspace(-140,0,3)
    #minoryticks=[-5,5,15,25]
    #ax.set_yticks( minoryticks, minor=True )
    #ax.set_yticks(np.arange(140))
    #ax.set_yticklabels( majoryticks, minor=False )
    # majorxticks=[90,140]
    #minorxticks=[100,110,120,130]
    #ax.set_xticklabels(stations)#, minor=True )
    # aa.set_xticks( majorxticks, minor=False )
    # aa.tick_params(axis='both', which='major', labelsize=8)

    plt.show()
    #fig.savefig('Figure_stations_TSturb_'+month+year+'_'+t+'.png', format='png')

        # for v in variable:
        #     if v == 'S':
        #         vmin = 0
        #         vmax = 31
        #         disp = 'Salinity (PSU)'
        #     elif v == 'T':
        #         vmin = 27
        #         vmax = 31
        #         disp = 'Temperature (°C)'
        #     elif v == 'turb':
        #         vmin = 30
        #         vmax = 500
        #         disp = 'Turbidity (FTU)'
        #     else:
        #         print('PROBLEM OF VARIABLE')
        #     print('Variable', v)
        #     # load the tab
        #     file = 'Tab_' + v + '_' + day + month + year + '_' + t + '.csv'
        #     tab = pd.read_csv(file)
        #     tab = pd.DataFrame(data=tab)
        #     tab = tab.T  # need to transpose because the files are recorded by rows
        #     print('tab i ', tab, i)
        #     tab = tab.reindex(index=tab.index[
        #                             ::-1])  # Put the values upsidedown tto have the Nan in the 0 rows, and to plot the values at same surface
        #     print('tab', tab)
        #
        #     # 0 Temperature, 1, Salinity, 2 Turbidity
        #     ax = axs[i]
        #     # ax.set_ylim(-10, 0)
        #     # ax2 = ax.twiny()
        #     a = np.arange(140)
        #     ax.set_ylabel('Depth (cm)', fontsize=fontsize)
        #     ax.set_xticklabels(stations, minor=True)
        #     p1 = ax.pcolor(tab, cmap=cmap, vmin=vmin, vmax=vmax)  # , label=disp)
        #     cbar = plt.colorbar(p1, label=disp)  # , ticks=1)#ax=ax
        #     cbar.ax.tick_params(labelsize=8)
        #     # majoryticks=np.linspace(-140,0,3)
        #     # minoryticks=[-5,5,15,25]
        #     # ax.set_yticks( minoryticks, minor=True )
        #     # ax.set_yticks(np.arange(140))
        #     # ax.set_yticklabels( majoryticks, minor=False )
        #     # majorxticks=[90,140]
        #     # minorxticks=[100,110,120,130]
        #     # ax.set_xticklabels(stations)#, minor=True )
        #     # aa.set_xticks( majorxticks, minor=False )
        #     # aa.tick_params(axis='both', which='major', labelsize=8)
        #
        #     plt.show()
        #
        #     i = i + 1