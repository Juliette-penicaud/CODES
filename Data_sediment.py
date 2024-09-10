# 7/02/24 : traiter les données sédiments
import pandas as pd
import numpy as np
import sys, os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib as mpl
from openpyxl import load_workbook
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
from scipy import stats
from matplotlib.lines import Line2D
import xarray as xr
from scipy.ndimage import median_filter
import cmcrameri as cmc
import seaborn as sns


def median_filter_numeric_columns(column):
    if pd.api.types.is_numeric_dtype(column.dtype):
        return median_filter(column, size=5)
    return column


# fontsize = 28
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['xtick.labelsize'] = fontsize - 4
# plt.rcParams['ytick.labelsize'] = fontsize - 4
# plt.rcParams['legend.fontsize'] = fontsize - 4
# s = 25


# VARIABLE AND PARAMETER
year = '2022'
list_month = ['June', 'August', 'Octobre']
i = 0  # 0 1 2 a voir pour faire une boucle si besoin
month = list_month[i]
all_month = True

if all_month :
    dataframes = {}
    for i in range(3):
        month = list_month[i]
        rep = '/home/penicaud/Documents/Data/Survey_' + month
        file = rep + '/Recap_all_param_' + month + '.xlsx'
        dict_month = {'June': {'nrows': 87},
                      'August': {'nrows': 95},
                      'Octobre': {'nrows': 111}}

        file_station = rep + '/Stations_' + month + '.xlsx'
        df_global = pd.read_excel(file_station, sheet_name=0,
                                  nrows=dict_month[month]['nrows'])  # read the stations name
        df_global = df_global.dropna(subset=['Stations'])
        list_sheet = df_global['Stations'].values
        list_sheet = [col for col in list_sheet if not '-' in col]  # remove the 'bistations' to have a unique liste

        # Initialize an empty dictionary to store the data frames
        # Read each sheet and load specific columns
        with pd.ExcelFile(file) as xls:
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)  # , usecols=columns_to_load)
                dataframes[sheet_name] = df
else :
    rep = '/home/penicaud/Documents/Data/Survey_' + month
    file = rep + '/Recap_all_param_' + month + '.xlsx'
    dict_month = {'June': {'nrows': 87},
                  'August': {'nrows': 95},
                  'Octobre': {'nrows': 111}}

    file_station = rep + '/Stations_' + month + '.xlsx'
    df_global = pd.read_excel(file_station, sheet_name=0, nrows=dict_month[month]['nrows'])  # read the stations name
    df_global = df_global.dropna(subset=['Stations'])
    list_sheet = df_global['Stations'].values
    list_sheet = [col for col in list_sheet if not '-' in col]  # remove the 'bistations' to have a unique liste

    # 1. Plot de SPMVC vs Turbidité : Faire ca pour toutes les turbidités mélangées ?
    # Xarray pour tout ouvrir en meme temps ?
    excel_data = pd.read_excel(file, sheet_name=None)

    # Initialize an empty dictionary to store the data frames
    dataframes = {}
    columns_to_load = ['Turbidity filtered 5', 'SPMVC']
    # Read each sheet and load specific columns
    with pd.ExcelFile(file) as xls:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)  # , usecols=columns_to_load)
            dataframes[sheet_name] = df

# # I check the min and max values of N2 :
# minimum, maximum = [], []
# for sheet_name, df in dataframes.items():
#     df['N'] = df['N2'].apply(lambda x: 0 if x < 0 else np.sqrt(x))
#     df['N filtered'] = df['N'].copy()
#     df.loc[df['N filtered'] < 0.5, 'N filtered'] = df.loc[df['N filtered'] < 0.5, 'N filtered'] * 0
#     minimum.append(np.nanmin(df['N2']))
#     maximum.append(np.nanmax(df['N2']))
# N2_min = min(minimum)
# N2_max = max(maximum)

# 8/02/24 J'ajoute un traitement sur les paramètres de sédim, pour voir oter les données si N<seuil
seuil_N = 0.035
minimum, maximum = [], []
for sheet_names,df in dataframes.items():
    df.loc[df['Junge'] == 1, 'Junge'] = np.nan  # A cause de l'écriture des fichiers (Create_recap_data_file),
    # si SPMVC = 0, Junge = 1. PROBLEME RESOLU VIA CETTE MAGOUILLE
    df['N'] = df['N2'].apply(lambda x: 0 if x < 0 else np.sqrt(x))
    df['N filtered'] = df['N'].copy()
    df.loc[df['N filtered'] < seuil_N, 'N filtered'] = 0
    if not df['N filtered'].isna().all() :
        minimum.append(np.nanmin(df['N filtered'].dropna()))
        maximum.append(np.nanmax(df['N filtered'].dropna()))
    for col in ['Junge', 'D50', 'ws']:
        df[col + ' filtered'] = df[col].copy()
        df.loc[df['N filtered'] == 0, col + ' filtered'] = 0
N_min = min(minimum)
N_max = max(maximum)

# Plot de SPMVC vs Turbidité
fontsize=15
all_in_one = True  # True if all stations in one figure
save = True
filtered = True
if all_in_one:
    fontsize = 15
    fig, ax = plt.subplots()
    fig.suptitle('LISST-CTD confrontation on '+month+' survey')
    ax.set_xlabel('Turbidity (FTU)', fontsize=fontsize)
    ax.set_ylabel('SPMVC (µL/L)', fontsize=fontsize)
    ax.grid(True, which='major')
    ax.grid(True, which='minor')

x_column = 'Turbidity filtered 5'
y_column = 'SPMVC'
color_column = 'N2'
cmap = 'Spectral'
for sheet_name, df in dataframes.items():
    # Check if the column exists in the DataFrame
    df_plot = df.dropna(subset=[x_column, y_column])
    df_plot = df_plot[(df_plot[x_column] != 0.0) & (df_plot[y_column] != 0.0)]
    # df_plot = df_plot[(df_plot[y_column] < 1000.0)]
    df_plot[color_column] = df_plot[color_column].apply(lambda x: 0 if x < 0 else np.sqrt(x))
    if filtered:
        df_plot[y_column] = median_filter(df_plot[y_column], size=5)
    if x_column in df and y_column in df:
        x_data = df_plot[x_column]
        y_data = df_plot[y_column]
        color = np.random.rand(3, )
        color = df_plot[color_column]
        if not all_in_one:
            fig, ax = plt.subplots()
            ax.set_xlabel('Turbidity (FTU)', fontsize=fontsize)
            ax.set_ylabel('SPMVC (µL/L)', fontsize=fontsize)
            ax.grid(True, which='major')
            ax.grid(True, which='minor')
            fig.suptitle(sheet_name)
        p1 = ax.scatter(x_data, y_data, label=sheet_name, c=color, cmap=cmap, vmin=N_min, vmax=N_max)  # color=color)
        if not all_in_one & save:
            outfile = 'SPMVC_vs_Turbidity_' + month + '_' + sheet_name
            if filtered:
                outfile = outfile + '_SPMVC_filtered'
            outfile = outfile + '.png'
            fig.savefig(outfile)
cbar = plt.colorbar(p1)
cbar.set_label('N')
if save:
    outfile = 'SPMVC_vs_Turbidity_all_in_one_' + month
    if filtered:
        outfile = outfile + 'SPMVC_filtered'
    outfile = outfile + '.png'
    fig.savefig(outfile)

# Plot df en fonction de N
fontsize = 15
x_column = 'N'
y_column = 'df'
fig, ax = plt.subplots()
#fig.suptitle('LISST-CTD confrontation on '+month+' survey')
ax.set_xlabel('Buoyancy Frequency N (s$^{-1}$)', fontsize=fontsize)
ax.set_ylabel('Fractal dimension', fontsize=fontsize)
ax.set_xlim(0,1)
ax.set_ylim(1,3)
ax.grid(True, which='major')
ax.grid(True, which='minor')
for sheet_name, df in dataframes.items():
    # Check if the column exists in the DataFrame
    df_plot = df.dropna(subset=[x_column, y_column])
    df_plot = df_plot[(df_plot[x_column] != 0.0) & (df_plot[y_column] != 0.0)]
    # df_plot = df_plot[(df_plot[y_column] < 1000.0)]
    #df_plot[color_column] = df_plot[color_column].apply(lambda x: 0 if x < 0 else np.sqrt(x))
    if x_column in df and y_column in df:
        x_data = df_plot[x_column]
        y_data = df_plot[y_column]
        color = np.random.rand(3, )
        color = df_plot[color_column]
        p1 = ax.scatter(x_data, y_data, label=sheet_name, color='blue')  # color=color)
if save:
    outfile = 'df_vs_N_' + month
    outfile = outfile + '.png'
    fig.savefig(outfile)

# plot of D50 vs salinity
x_column = 'Salinity'
y_column = 'D50 filtered'
color_column = 'module vitesse u'
colors = True
fig, ax = plt.subplots()
#fig.suptitle('LISST-CTD confrontation on '+month+' survey')
ax.set_xlabel('Salinity (PSU)', fontsize=fontsize)
ax.set_ylabel('D50 (µm)', fontsize=fontsize)
ax.grid(True, which='major')
ax.grid(True, which='minor')
for sheet_name, df in dataframes.items():
    # Check if the column exists in the DataFrame
    df_plot = df.dropna(subset=[x_column, y_column])
    df_plot = df_plot[(df_plot[y_column] != 0.0)]
    # df_plot = df_plot[(df_plot[y_column] < 1000.0)]
    #df_plot[color_column] = df_plot[color_column].apply(lambda x: 0 if x < 0 else np.sqrt(x))
    if x_column in df and y_column in df:
        x_data = df_plot[x_column]
        y_data = df_plot[y_column]
        if colors :
            color = df_plot[color_column]/1000
            p1 = ax.scatter(x_data, y_data, label=sheet_name, c=color, cmap='Spectral',  s=10, zorder=2,  vmin = 0, vmax=2)
        else :
            color = np.random.rand(3, )
            p1 = ax.scatter(x_data, y_data, label=sheet_name, color='k', s=10, zorder=2)  # color=color)
if colors :
    cbar = plt.colorbar(p1)
    cbar.set_label('velocity (m/s)')
if save:
    outfile = 'D50_vs_salinity' + month
    outfile = outfile + '.png'
    fig.savefig(outfile)

# plot of D50 vs velocity
x_column = 'module vitesse u'
y_column = 'D50 filtered'
color_column = 'Salinity'
colors = True
fig, ax = plt.subplots()
#fig.suptitle('LISST-CTD confrontation on '+month+' survey')
ax.set_xlabel('Velocity (m/s)', fontsize=fontsize)
ax.set_ylabel('D50 (µm)', fontsize=fontsize)
ax.grid(True, which='major')
ax.grid(True, which='minor')
for sheet_name, df in dataframes.items():
    # Check if the column exists in the DataFrame
    df_plot = df.dropna(subset=[x_column, y_column])
    df_plot = df_plot[(df_plot[y_column] != 0.0)]
    # df_plot = df_plot[(df_plot[y_column] < 1000.0)]
    #df_plot[color_column] = df_plot[color_column].apply(lambda x: 0 if x < 0 else np.sqrt(x))
    if x_column in df and y_column in df:
        x_data = df_plot[x_column]
        y_data = df_plot[y_column]
        if colors :
            color = df_plot[color_column]
            p1 = ax.scatter(x_data, y_data, label=sheet_name, c=color, cmap='Spectral',  s=10, zorder=2, vmin=0, vmax=30)
        else :
            color = np.random.rand(3, )
            p1 = ax.scatter(x_data, y_data, label=sheet_name, color='k', s=10, zorder=2)  # color=color)
if colors :
    cbar = plt.colorbar(p1)
    cbar.set_label('Salinity (PSU)')
if save:
    outfile = 'D50_vs_module_velocity_' + month
    outfile = outfile + '.png'
    fig.savefig(outfile)

# D50 vs grad vertical velocity
x_column = 'grad vitesse horiz'
y_column = 'D50 filtered'
color_column = 'Salinity'
colors = True
fig, ax = plt.subplots()
#fig.suptitle('LISST-CTD confrontation on '+month+' survey')
ax.set_xlabel('velocity gradient (m/s/m)', fontsize=fontsize)
ax.set_ylabel('D50 (µm)', fontsize=fontsize)
ax.grid(True, which='major')
ax.grid(True, which='minor')
for sheet_name, df in dataframes.items():
    # Check if the column exists in the DataFrame
    df_plot = df.dropna(subset=[x_column, y_column])
    df_plot = df_plot[(df_plot[y_column] != 0.0)]
    # df_plot = df_plot[(df_plot[y_column] < 1000.0)]
    #df_plot[color_column] = df_plot[color_column].apply(lambda x: 0 if x < 0 else np.sqrt(x))
    if x_column in df and y_column in df:
        x_data = abs(df_plot[x_column])/1000
        y_data = df_plot[y_column]
        if colors :
            color = df_plot[color_column]
            p1 = ax.scatter(x_data, y_data, label=sheet_name, c=color, cmap='Spectral',  s=10, zorder=2, vmin=0, vmax=30)
        else :
            color = np.random.rand(3, )
            p1 = ax.scatter(x_data, y_data, label=sheet_name, color='k', s=10, zorder=2)  # color=color)
if colors :
    cbar = plt.colorbar(p1)
    cbar.set_label('Salinity (PSU)')
if save:
    outfile = 'D50_vs_velocity_gradient_' + month
    outfile = outfile + '.png'
    fig.savefig(outfile)



# plot of D50 vs MO
x_column = '% OM'
y_column = 'D50 filtered'
color_column = 'Salinity'
colors = True
fig, ax = plt.subplots()
#fig.suptitle('LISST-CTD confrontation on '+month+' survey')
ax.set_xlabel('% OM', fontsize=fontsize)
ax.set_ylabel('D50 (µm)', fontsize=fontsize)
ax.grid(True, which='major')
ax.grid(True, which='minor')
for sheet_name, df in dataframes.items():
    # Check if the column exists in the DataFrame
    df_plot = df.dropna(subset=[x_column, y_column])
    df_plot = df_plot[(df_plot[y_column] != 0.0)]
    # df_plot = df_plot[(df_plot[y_column] < 1000.0)]
    #df_plot[color_column] = df_plot[color_column].apply(lambda x: 0 if x < 0 else np.sqrt(x))
    if x_column in df and y_column in df:
        x_data = df_plot[x_column]
        y_data = df_plot[y_column]
        if colors :
            color = df_plot[color_column]
            p1 = ax.scatter(x_data, y_data, label=sheet_name, c=color, cmap='Spectral',  s=10, zorder=2, vmin=0, vmax=30)
        else :
            color = np.random.rand(3, )
            p1 = ax.scatter(x_data, y_data, label=sheet_name, color='k', s=10, zorder=2)  # color=color)
if colors :
    cbar = plt.colorbar(p1)
    cbar.set_label('Salinity (PSU)')
if save:
    outfile = 'D50_vs_%OM_' + month
    outfile = outfile + '.png'
    fig.savefig(outfile)



# Nombre de valeurs de N2 < 0
seuil = 0.02
N2_inf_0, N2_sup_0, N2_sup_seuil = [], [], []
for sheet_names,df in dataframes.items():
    N2_inf_0.append(df['N2'].where(df["N2"] < 0).count())
    N2_sup_0.append(df['N2'].where(df["N2"] > 0).count())
    N2_sup_seuil.append(df['N2'].where(df["N2"] >= seuil).count())

float_sum_inf = sum(item for item in N2_inf_0)
float_sum_sup = sum(item for item in N2_sup_0)
float_sum_sup_seuil = sum(item for item in N2_sup_seuil)



# 15/02/24 : I use pairplot to see the different correlation possibles
concat_df = pd.concat(dataframes)
# 8/02/24 J'ajoute un traitement sur les paramètres de sédim, pour voir oter les données si N<seuil
seuil_N = 0.035
minimum, maximum = [], []
concat_df.loc[concat_df['Junge'] == 1, 'Junge'] = np.nan  # A cause de l'écriture des fichiers (Create_recap_data_file),
# si SPMVC = 0, Junge = 1. PROBLEME RESOLU VIA CETTE MAGOUILLE
concat_df['N'] = concat_df['N2'].apply(lambda x: 0 if x < 0 else np.sqrt(x))
concat_df['N filtered'] = concat_df['N'].copy()
concat_df.loc[concat_df['N filtered'] < seuil_N, 'N filtered'] = 0
if not concat_df['N filtered'].isna().all() :
    minimum.append(np.nanmin(concat_df['N filtered'].dropna()))
    maximum.append(np.nanmax(concat_df['N filtered'].dropna()))
for col in ['Junge', 'D50', 'ws']:
    concat_df[col + ' filtered'] = concat_df[col].copy()
    concat_df.loc[concat_df['N filtered'] == 0, col + ' filtered'] = 0
N_min = min(minimum)
N_max = max(maximum)


concat_df.loc[concat_df['D50 filtered'] ==0,  'D50 filtered'] = np.nan

list_vars_x = ['N filtered', 'module vitesse u', '% OM', 'Salinity', 'SPMVC']
list_vars_y = ['D50 filtered']
sns.pairplot(concat_df, x_vars=list_vars_x, y_vars=list_vars_y)

concat_df['SPMVC'].where(concat_df['SPMVC'] > 1000).count()