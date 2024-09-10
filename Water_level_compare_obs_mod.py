# Compare between Obs and modelled
# 12/10 : Water level at Hon Dau and Trung Trang

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import sys, pprint, os, glob
import xarray as xr
import dask
import cmcrameri.cm as cmc
#import cartopy
#import cartopy.crs as ccrs
#import seaborn as sns
from functools import reduce
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import pandas as pd
import cmcrameri as cmc
import seaborn as sns
import xarray as xr
from datetime import datetime, timedelta
import matplotlib as mpl
from openpyxl import load_workbook

# 1. Modelled data to download
rep_modele = '/home/penicaud/Documents/Modèles/Mustang/GRAPHIQUES/'
file_rough4 = rep_modele + 'rough_1e-4/201701_all_ssh.nc'
file_rough3 = rep_modele + 'rough_1e-3/201701_all_ssh.nc'
zoom = 0
dask.config.set(**{'array.slicing.split_large_chunks': False}) # Needed to avoid slicing
ds_rough4 = xr.open_dataset(file_rough4)
ds_rough3 = xr.open_dataset(file_rough3)

# Dict with location i j
dict_loc = {'ST1': {'i': 61, 'j': 456 },
                 'ST2': {'i': 36 , 'j': 454 },
                 'ST3': {'i': 15 , 'j': 459 },
                 'HD' : {'i':103, 'j':463},
                 'TT': {'i':8, 'j':460},
                 'ST': {'i':2, 'j':460}}

# 2. Open the data of the hydro stations
# 2017 Data SPM Water level and Discharge #####################################################################
path = '/home/penicaud/Documents/Data/Décharge_waterlevel/'
file = path + 'Data_2017.xlsx'
columns_to_load = list(range(2,6))
# Water level at Trung Trang and discharge
df_TT_2017 = pd.read_excel(file, sheet_name = 'Q_trungtrang_vanuc_2017', usecols=columns_to_load, skiprows=3, nrows = 8772)
df_TT_2017['Datetime'] = pd.to_datetime(df_TT_2017['Date']) + pd.to_timedelta(df_TT_2017['Time (hours)'], unit='h')
df_TT_2017.sort_values("Datetime", inplace=True)
df_TT_2017.drop(['Date', 'Time (hours)'], axis=1, inplace=True)
# SPM DATA
df_SPM_2017 = pd.read_excel(file, sheet_name = 'SPM_TRungtrang_vanuc17', usecols=list(range(2,5)), skiprows=2, nrows = 365)
df_HD_2017 = pd.read_excel(file, sheet_name='Water_level_HonDau2017')
# 11/10 : Open and use the 2017 data at TT and HD
#file = path + 'Water_level_HonDau_2017_temporal_serie.xlsx'
#df_HD_2017 = pd.read_excel(file)

save = True
months = [1]
year = 2017
fontsize = 10
Q = 'Q (m3/s)'
Ebb = 'mean in ebb tide'
Flood = 'mean in flood tide'
Wat_lev = 'Water level (cm)'
date_form = DateFormatter("%d/%m")  # Define the date format

months = [1]
for month in months:
    a = '0' if month < 10 else ''
    month_constraint = True
    if month_constraint:
        selected_data = df_TT_2017[df_TT_2017['Datetime'].dt.month == month]
        selected_SPM = df_SPM_2017[(df_SPM_2017['Date'].dt.month == month)]
        selected_HD = df_HD_2017[(df_HD_2017['Datetime'].dt.month == month)]
    else:
        print('no constraint on year or month, I take the whole series')
        selected_data = df_TT_2017
        selected_SPM = df_SPM_2017
        selected_HD = df_HD_2017
    daily_mean = selected_data.resample('D', on='Datetime').mean()
    selected_SPM = selected_SPM.replace('-', np.nan)
    selected_SPM['Mean'] = (selected_SPM['mean in flood tide'] + selected_SPM['mean in ebb tide']) / 2

date_string1 = ds_rough4.attrs['simulation_start_date']
#date_string2 = ds_rough4.attrs['simulation_end_date__']
date_format = "%Y%m%d-%Hh%Mm%Ss"
time1 = datetime.strptime(date_string1, date_format)
time2 = datetime(time1.year, time1.month, time1.day + int(ds_rough4.attrs['Duration_in_days'])-1)
# Re build the temporal series
date_range = pd.date_range(start=time1, end=time2, freq='30T')
liste_time_sim = ['000000','002900','005800','012630','015530','022400','025259','032159','035029','041929','044759','051659','054600','061430','064330','071200','074100','080959','083830',
                  '090729','093559','100459','103359','110229','113130','120000','122900','125800','132630','135530','142400','145259','152159','155029','161929','164759','171659','174600',
                  '181430','184330','191200','194100','200959','203830','210729','213559','220459','223359','230229','233130']
time_serie=[]
for d in range(time1.day, time2.day+1):
    for t in liste_time_sim:
        #time_format = "%H%M%S"
        hour = t[0:2]
        min = t[2:4]
        sec = t[4:6]
        # print('day = ' , d)
        b = '0' if d < 10 else ''
        date_string_sortie = str(time1.year)+ a + str(time1.month) +b + str(d)+ '-' + hour +'h'+min+'m'+sec+'s'
        time = datetime.strptime(date_string_sortie, date_format)
        time = time + timedelta(hours=7) # MODEL hour is in UT : so we need to add 7hours
        time_serie.append(time)

# I plot 3 plots for the ssh at the hydrological stations
fig,axs = plt.subplots(nrows=2, ncols=1, figsize=(15,10), sharex=True)
fig2,ax2s = plt.subplots(nrows=2, ncols=1, figsize=(15,10), sharex=True)
#plt.gcf().subplots_adjust(left = 0.05 , bottom=0.1 , right=0.95, top=0.9, wspace=0.2, hspace=0.4)
fig.suptitle('z0 = 10-4m')
fig2.suptitle('z0 = 10-3m')
###### 1st subplot TT
if zoom :
    i_loc = dict_loc['TT']['i']
    j_loc = dict_loc['TT']['j']-420
else :
    i_loc = dict_loc['TT']['i']
    j_loc = dict_loc['TT']['j']
print(i_loc, j_loc)

ax = axs[0]
ax.grid(True,alpha = 0.5)
#ax.title.set_text('TT')
ax.plot(time_serie, ds_rough4.ssh_w.sel(ni_t=i_loc, nj_t=j_loc).values -
        np.average(ds_rough4.ssh_w.sel(ni_t=i_loc, nj_t=j_loc)), color='grey', label = 'Model', alpha=0.5, lw=3)
# plot de la donnée à cet endroit :
ax.set_ylabel('Water_level (m)', fontsize=fontsize)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize)
ax.title.set_text('Trung Trang')
ax.plot(selected_data['Datetime'], selected_data[Wat_lev]/100 - np.average(selected_data[Wat_lev]/100) , color='k', label='Obs', lw=3, alpha=0.5)
ax.legend()


ax = ax2s[0]
ax.grid(True,alpha = 0.5)
#ax.title.set_text('TT')
ax.plot(time_serie, ds_rough3.ssh_w.sel(ni_t=i_loc, nj_t=j_loc).values -
        np.average(ds_rough3.ssh_w.sel(ni_t=i_loc, nj_t=j_loc)), color='grey', label = 'Model', alpha=0.5, lw=3)
# plot de la donnée à cet endroit :
ax.set_ylabel('Water_level (m)', fontsize=fontsize)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize)
ax.title.set_text('Trung Trang')
ax.plot(selected_data['Datetime'], selected_data[Wat_lev]/100 - np.average(selected_data[Wat_lev]/100), color='k', label='Obs', lw=3, alpha=0.5)
ax.legend()

#### 2d subplot HD
if zoom :
    i_loc = dict_loc['HD']['i']
    j_loc = dict_loc['HD']['j']-420
else :
    i_loc = dict_loc['HD']['i']
    j_loc = dict_loc['HD']['j']

ax = axs[1]
ax.grid(True,alpha = 0.5)
ax.title.set_text('Hon Dau')
ax.plot(time_serie, ds_rough4.ssh_w.sel(ni_t=i_loc, nj_t=j_loc).values -
        np.average(ds_rough4.ssh_w.sel(ni_t=i_loc, nj_t=j_loc)), color='grey', label = 'Model', lw=3, alpha=0.5)
ax.set_ylabel('Water_level (m)', fontsize=fontsize)
ax.set_xlabel('Time', fontsize=fontsize)
ax.plot(selected_HD['Datetime'], selected_HD['Value']/100 - np.average(selected_HD['Value']/100),
        color='k', label='Obs', lw=3, alpha=0.5)
ax.legend()

ax = ax2s[1]
ax.grid(True, alpha=0.5)
ax.title.set_text('Hon Dau')
ax.plot(time_serie, ds_rough4.ssh_w.sel(ni_t=i_loc, nj_t=j_loc).values -
        np.average(ds_rough4.ssh_w.sel(ni_t=i_loc, nj_t=j_loc)), color='grey', label='Model', lw=3, alpha=0.5)
ax.set_ylabel('Water_level (m)', fontsize=fontsize)
ax.set_xlabel('Time', fontsize=fontsize)
ax.plot(selected_HD['Datetime'], selected_HD['Value'] / 100 -np.average(selected_HD['Value']/100),
        color='k', label='Obs', lw=3, alpha=0.5)
ax.legend()

outfile = rep_modele+'SSH_model_obs_HD_TT_january2017_z0_'
outfile1 = outfile + '10e-4.png'
outfile2 = outfile + '10e-3.png'
fig.savefig(outfile1, format='png')
fig2.savefig(outfile2, format='png')