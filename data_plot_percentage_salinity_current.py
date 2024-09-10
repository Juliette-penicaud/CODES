
import numpy as np
import csv
import pandas as pd
import numpy as np
import cmcrameri as cmc
import seaborn as sns
import xarray as xr
import sys, os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib as mpl
from openpyxl import load_workbook
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from scipy import stats
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
mpl.use('Agg')

fontsize = 28
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['xtick.labelsize'] = fontsize - 4
plt.rcParams['ytick.labelsize'] = fontsize - 4
plt.rcParams['legend.fontsize'] = fontsize - 4
s = 25

# 5/12 : plot percentage of salinity
df_percentage_salinity = pd.DataFrame({'Salinity_last0':[80.5,79.2, 70.8],'Salinity_firstnon0':[81.9,83.3, 75],
                                       'first_23PSU_salinity':[89.58, 100, 92], 'discharge':[1954,1577,691]})
df_percentage_velocity = pd.DataFrame({'percentage_velocity+to-_min':[65.3,62.5, 45.83],
                                       'percentage_velocity+to-_max':[66.7,66.7, 50],
                                       'percentage_velocity-to+_min': [np.nan, -100, -88.461538],
                                       'percentage_velocity-to+_max': [np.nan, -96.15,  -84.615385],
                                       'percentage_min':[94.4,91.7, 75] ,'value_min':[-0.621,-0.61927, -0.982] ,
                                       'percentage_max':[np.nan,-67.1 , -50] ,
                                       'value_max':[np.nan,1.673, 1.896],
                                       'discharge': [1954, 1577, 691]})


# FIgure of the percentage vs discharge for the salinity parameters
fig, ax = plt.subplots(figsize=(18, 10))
ax.grid(True, alpha=0.5)
ax.set_ylabel('Arrival salinity (% of tide)', fontsize=fontsize - 2)
ax.set_xlabel('Discharge (m$^{3}$/s)', fontsize=fontsize - 2)
ax.set_ylim(65,100)
ax.set_xlim(500, 2100)
s=55
ax.scatter(df_percentage_salinity["discharge"], df_percentage_salinity['Salinity_last0'], color='grey', s=s, label='Last 0 salinity')
ax.scatter(df_percentage_salinity["discharge"], df_percentage_salinity['Salinity_firstnon0'], color='grey', s=s, marker = 'd', label='First non 0 salinity')
ax.scatter(df_percentage_salinity["discharge"],  (df_percentage_salinity['Salinity_firstnon0']+df_percentage_salinity['Salinity_last0'])/2, color='black', s=s)
ax.fill_between(df_percentage_salinity["discharge"], df_percentage_salinity['Salinity_last0'], df_percentage_salinity['Salinity_firstnon0'],
                color='grey', alpha = 0.2, label = 'Arrival salinity')

ax.scatter(df_percentage_salinity["discharge"],  df_percentage_salinity['first_19PSU_salinity'] , color='grey', marker='+', s=s, label = 'First 19 salinity')
ax.fill_between(df_percentage_salinity["discharge"], df_percentage_salinity['Salinity_firstnon0'], df_percentage_salinity['first_19PSU_salinity'],
                color='orange', alpha = 0.2, label='Salinity instauration')
x = np.arange(500, 2000, 10)

slope, intercept, r_value, p_value, std_err = stats.linregress(df_percentage_salinity["discharge"], df_percentage_salinity['Salinity_last0'])
label = "{:.1e}".format(slope) + ' discharge + ' + str(np.round(intercept, 2)) + '% of tide, r=' + str(np.round(r_value, 3))
print(p_value)
ax.plot(x, slope * x + intercept, lw=1, color='grey', label=label)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_percentage_salinity["discharge"], df_percentage_salinity['Salinity_firstnon0'])
label = "{:.1e}".format(slope) + ' discharge + ' + str(np.round(intercept, 2)) + ' % of tide, r=' + str(np.round(r_value, 3))
ax.plot(x, slope * x + intercept, lw=1,ls='--',  color='grey', label=label)
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_percentage_salinity["discharge"],
                                                               (df_percentage_salinity['Salinity_firstnon0']+df_percentage_salinity['Salinity_last0'])/2)
label = "{:.1e}".format(slope) + ' discharge + ' + str(np.round(intercept, 2)) + ' % of tide, r=' + str(np.round(r_value, 3))
ax.plot(x, slope * x + intercept, lw=1, ls=':', color='black', label=label)
print(p_value)
legend = ax.legend(loc='upper left', ncol=2, framealpha=0.5)
# Set the font size for the legend labels
for label in legend.get_texts():
    label.set_fontsize(18)  # Set the desired font size
fig.savefig('Percentage_last0salinity_vs_discharge.png', format='png')


# Figure of the percentage vs discharge for the velocity parameters
fig, ax = plt.subplots(figsize=(18, 10))
ax.grid(True, alpha=0.5)
ax.set_ylabel('Current changing sign (% of tide)', fontsize=fontsize - 2)
ax.set_xlabel('Discharge (m$^{3}$/s)', fontsize=fontsize - 2)
ax.set_xlim(500, 2100)
s=55
ax.scatter(df_percentage_velocity["discharge"], df_percentage_velocity['percentage_velocity+to-_min'],
           color='grey', s=s, label='Last positive velocity')
ax.scatter(df_percentage_velocity["discharge"], df_percentage_velocity['percentage_velocity+to-_max'],
           color='grey', s=s, marker = 'd', label='First negative velocity')
ax.scatter(df_percentage_velocity["discharge"],  (df_percentage_velocity['percentage_velocity+to-_min']+
                                                  df_percentage_velocity['percentage_velocity+to-_max'])/2, color='black', s=s)
ax.fill_between(df_percentage_velocity["discharge"], df_percentage_velocity['percentage_velocity+to-_min'],
                df_percentage_velocity['percentage_velocity+to-_max'],
                color='grey', alpha = 0.2, label = 'Reverse current')

ax.scatter(df_percentage_velocity["discharge"],  df_percentage_velocity['percentage_min'] , color='grey', marker='+',
           s=s, label = 'min velocity values')
ax.fill_between(df_percentage_velocity["discharge"], df_percentage_velocity['percentage_velocity+to-_max'],
                df_percentage_velocity['percentage_min'],
                color='orange', alpha = 0.2, label='Negative velocity instauration')
x = np.arange(500, 2000, 10)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_percentage_velocity["discharge"],
                                                               df_percentage_velocity['percentage_velocity+to-_min'])
label = "{:.1e}".format(slope) + ' discharge + ' + str(np.round(intercept, 2)) + '% of tide, r=' + str(np.round(r_value, 3))
print(p_value)
ax.plot(x, slope * x + intercept, lw=1, color='grey', label=label)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_percentage_velocity["discharge"],
                                                               df_percentage_velocity['percentage_velocity+to-_max'])
label = "{:.1e}".format(slope) + ' discharge + ' + str(np.round(intercept, 2)) + ' % of tide, r=' + str(np.round(r_value, 3))
ax.plot(x, slope * x + intercept, lw=1, ls='--', color='grey', label=label)
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_percentage_velocity["discharge"],
                                                               (df_percentage_velocity['percentage_velocity+to-_min'] +
                                                                df_percentage_velocity['percentage_velocity+to-_max'])/2)
label = "{:.1e}".format(slope) + ' discharge + ' + str(np.round(intercept, 2)) + ' % of tide, r=' + str(np.round(r_value, 3))
ax.plot(x, slope * x + intercept, lw=1, ls=':', color='k', label=label)
print(p_value)
legend = ax.legend(loc='upper left', ncol=2,  framealpha=0.5)
# Set the font size for the legend labels
for label in legend.get_texts():
    label.set_fontsize(18)  # Set the desired font size
fig.savefig('Percentage_velocity+to-_vs_discharge.png', format='png')


# Figure percentage - to +
fig, ax = plt.subplots(figsize=(18, 10))
ax.grid(True, alpha=0.5)
ax.set_ylabel('Current changing sign (% of tide)', fontsize=fontsize - 2)
ax.set_xlabel('Discharge (m$^{3}$/s)', fontsize=fontsize - 2)
ax.set_xlim(500, 2100)
s=55
ax.scatter(df_percentage_velocity["discharge"], df_percentage_velocity['percentage_velocity-to+_min'],
           color='grey', s=s, label='Last negative velocity')
ax.scatter(df_percentage_velocity["discharge"], df_percentage_velocity['percentage_velocity-to+_max'],
           color='grey', s=s, marker = 'd', label='First positive velocity')
ax.scatter(df_percentage_velocity["discharge"],  (df_percentage_velocity['percentage_velocity-to+_min']+
                                                  df_percentage_velocity['percentage_velocity-to+_max'])/2, color='black', s=s)
ax.fill_between(df_percentage_velocity["discharge"], df_percentage_velocity['percentage_velocity-to+_min'],
                df_percentage_velocity['percentage_velocity-to+_max'],
                color='grey', alpha = 0.2, label = 'Reverse current')

ax.scatter(df_percentage_velocity["discharge"],  df_percentage_velocity['percentage_max'] , color='grey', marker='+',
           s=s, label = 'max velocity values')
ax.fill_between(df_percentage_velocity["discharge"], df_percentage_velocity['percentage_velocity-to+_max'],
                df_percentage_velocity['percentage_max'],
                color='orange', alpha = 0.2, label='Positive velocity instauration')
x = np.arange(500, 2000, 10)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_percentage_velocity.dropna()["discharge"],
                                                               df_percentage_velocity['percentage_velocity-to+_min'].dropna())
label = "{:.1e}".format(slope) + ' discharge + ' + str(np.round(intercept, 2)) + '% of tide, r=' + str(np.round(r_value, 3))
print(p_value)
ax.plot(x, slope * x + intercept, lw=1, color='grey', label=label)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_percentage_velocity.dropna()["discharge"],
                                                               df_percentage_velocity['percentage_velocity-to+_max'].dropna())
label = "{:.1e}".format(slope) + ' discharge + ' + str(np.round(intercept, 2)) + ' % of tide, r=' + str(np.round(r_value, 3))
ax.plot(x, slope * x + intercept, lw=1, ls='--', color='grey', label=label)
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_percentage_velocity.dropna()["discharge"],
                                                               (df_percentage_velocity.dropna()['percentage_velocity-to+_min'] +
                                                                df_percentage_velocity.dropna()['percentage_velocity-to+_max'])/2)
label = "{:.1e}".format(slope) + ' discharge + ' + str(np.round(intercept, 2)) + ' % of tide, r=' + str(np.round(r_value, 3))
ax.plot(x, slope * x + intercept, lw=1, ls=':', color='k', label=label)
print(p_value)
legend = ax.legend(loc='upper left', ncol=2,  framealpha=0.5)
# Set the font size for the legend labels
for label in legend.get_texts():
    label.set_fontsize(18)  # Set the desired font size
fig.savefig('Percentage_velocity-to+_vs_discharge.png', format='png')
