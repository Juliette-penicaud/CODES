# 30/08/2023 :  study discharge data.

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

# Calculation of the lag correlation
def calculate_corr(data1, data2, year_constraint, year, month_constraint, month):
    max_lag = 5  # Maximum lag value to test
    correlations = []
    if year_constraint and month_constraint :
        df1 = data1[(data1['Datetime'].dt.year == year) & (data1['Datetime'].dt.month == month)]
        df2 = data2[(data2['Datetime'].dt.year == year) & (data2['Datetime'].dt.month == month)]
    elif year_constraint :
        df1 = data1[(data1['Datetime'].dt.year == year)]
        df2 = data2[(data2['Datetime'].dt.year == year)]
    elif month_constraint :
        df1 = data1[data1['Datetime'].dt.month == month]
        df2 = data2[data2['Datetime'].dt.month == month]
    else :
        print('no constraint on year or month, I take the whole series')
        df1 = data1
        df2 = data2
    for lag in range(-max_lag, max_lag + 1):
        shifted_df2 = df2.shift(periods=lag)
        m2 = pd.concat([df1.reset_index().drop('index', axis=1), shifted_df2.reset_index().drop('index', axis=1)], axis=1)
        # correlation = m2['Q'].corr(m2['Q (m3/s)'])
        correlation = m2.corr().iloc[0, 1]
        correlations.append((lag, correlation))
    # Find the lag with the highest correlation
    best_lag, best_correlation = max(correlations, key=lambda x: abs(x[1]))
    # print(f"Best lag: {best_lag}")
    # print(f"Best correlation: {best_correlation}")
    return best_lag,best_correlation
    # ON WATER LEVEL :
    # For the whole temporal serie  Best lag: 2 Best correlation: 0.9403780905327238
    # For 2022 : Best lag: 2 Best correlation: 0.9231507853916974
    # Monthly : worst for May to Sept, but still > 0.87
    # For january, Best lag: 2 Best correlation: 0.974414815770983 febr,Lag: 2 Corr: 0.9841633877568988 ,  march, Lag: 2 Corr: 0.98359689458487
    # april, Lag: 2 Corr: 0.985 may, Lag: 2 Corr: 0.940 june, Lag: 2 Corr: 0.874 july lag: 2 correlation: 0.8816781512215971
    # Aug, Lag: 2 Corr: 0.881 Sept, Lag: 2 Corr: 0.957 Oct, Lag: 2 Corr: 0.953  Nov  Lag: 2 Corr: 0.979 Dec, Lag: 2 Corr: 0.980


######################### 2017 Data SPM Water level and Discharge ##################################################
path = '/home/penicaud/Documents/Data/Décharge_waterlevel/'
file = path + 'Data_2017.xlsx'
columns_to_load = list(range(2,6))
# Water level at Trung Trang and discharge
TT_2017 = pd.read_excel(file, sheet_name = 'Q_trungtrang_vanuc_2017', usecols=columns_to_load, skiprows=3, nrows = 8772)
TT_2017['Datetime'] = pd.to_datetime(TT_2017['Date']) + pd.to_timedelta(TT_2017['Time (hours)'], unit='h')
TT_2017.sort_values("Datetime", inplace=True)
TT_2017.drop(['Date', 'Time (hours)'], axis=1, inplace=True)
# SPM DATA
df_SPM_2017 = pd.read_excel(file, sheet_name = 'SPM_TRungtrang_vanuc17', usecols=list(range(2,5)), skiprows=2, nrows = 365)

# 11/10 : Open and use the 2017 data at TT and HD
df_HD_2017 = pd.read_excel(file, sheet_name='Water_level_HonDau2017')

Q = 'Q (m3/s)'
Ebb = 'mean in ebb tide'
Flood = 'mean in flood tide'
Wat_lev = 'Water level (cm)'

# Manip over the tables to find the datetime of the maximum and minimum (only negative) discharge
TT_2017['Date'] = TT_2017['Datetime'].dt.date
# Find the maximum discharge data.
max_value_indices = TT_2017.groupby('Date')[Q].idxmax()
max_values_per_day = TT_2017.loc[max_value_indices]
max_values_per_day['Date'] = pd.to_datetime(max_values_per_day['Date']) # Changing the columns type from object to datetime
# Find the minimum AND negative discharge data
min_value_indices = TT_2017.groupby('Date')[Q].idxmin()
min_values_per_day = TT_2017.loc[min_value_indices]
neg_values_per_day = min_values_per_day.loc[min_values_per_day[Q]<0]
neg_values_per_day['Date'] = pd.to_datetime(neg_values_per_day['Date']) # Changing the columns type from object to datetime
# Split Ebb and flood mean spm to recreate a corresponding table (max values velocity and Ebb mean SPM ; neg max values
# of flood tide and mean Flood spm)
Ebb_SPM = df_SPM_2017[['Date', Ebb]].copy()
Flood_SPM = df_SPM_2017[['Date', Flood]].copy()
Flood_df = neg_values_per_day.merge(Flood_SPM, on='Date', how='inner')
Ebb_df = max_values_per_day.merge(Ebb_SPM, on='Date', how='inner')

# FIGURE
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
save = True
months = [1]
year = 2017
fontsize = 12

month_constraint = False
if month_constraint :
    for month in months:
        a = '0' if month < 10 else ''
        selected_data = TT_2017[TT_2017['Datetime'].dt.month == month]
        selected_SPM = df_SPM_2017[(df_SPM_2017['Date'].dt.month == month)]
        selected_HD = df_HD_2017[(df_HD_2017['Datetime'].dt.month == month)]
else:
    print('no constraint on year or month, I take the whole series')
    selected_data = TT_2017
    selected_SPM = df_SPM_2017
    selected_HD = df_HD_2017
daily_mean = selected_data.resample('D', on='Datetime').mean()
selected_SPM = selected_SPM.replace('-', np.nan)
#selected_SPM['Mean'] = (selected_SPM['mean in flood tide'] + selected_SPM['mean in ebb tide']) / 2
selected_SPM['Mean'] = np.nanmean(selected_SPM[['mean in flood tide', 'mean in ebb tide']], axis=1)
selected_SPM.set_index('Date', inplace=True)
#selected_SPM.set_index('Date', inplace=True)
# 21/05/24
# Je créé un df avec les données de Q et SPM
df_2017 = pd.merge(daily_mean.drop(columns='Water level (cm)'), selected_SPM, left_index=True, right_index=True, how='inner')

# First, I remove the outliers to derive the trend. 1 : selection of the outliers thanks to the z-score method :
# Calculate Z-scores for the 'y' column
z_scores = np.abs((Ebb_df[Ebb] - Ebb_df[Ebb].mean()) / Ebb_df[Ebb].std())
# Set a threshold for outliers (e.g., Z-score > 2)
threshold = 2
# Filter out the outliers
filtered_Ebb_df = Ebb_df[z_scores <= threshold] # Now 'filtered_df' contains the data without outliers

# Calculation of the parameters of the predicted values :
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_Ebb_df.dropna(how ='any')[Q], filtered_Ebb_df.dropna(how ='any')[Ebb])
predicted_values = slope*filtered_Ebb_df[Q]+intercept

errors = [observed - predicted for observed, predicted in zip(filtered_Ebb_df[Ebb], predicted_values)]
squared_errors = [np.mean(np.array(errors_i) ** 2) for errors_i in errors]
mean_squared_error = np.nanmean(squared_errors)
rmse = np.sqrt(mean_squared_error)

# 21/05/24 : Plot d'un plot avec 1 débit liquide et autre échelle = débit solide
fontsize = 18
fig, ax = plt.subplots(figsize=(18, 10))  # , sharex=True)
fig.suptitle('Discharge and SPM at Trung Trang, 2017', fontsize=fontsize)  # + a + str(month) + '/' + str(year))
ax.grid(True, alpha=0.5)
ax.set_ylabel('Discharge ($m^{3}/s$)', fontsize=fontsize)
ax.set_xlim(datetime(2017,1,1), datetime(2017,12,31))
ax.grid(True, alpha=0.5)
date_form = DateFormatter("%m/%Y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize)
# twin axe
twin2 = ax.twinx()
twin2.set_ylabel('Concentration (mg/L)', fontsize=fontsize, color='coral')
# Set the tick label sizes
ax.tick_params(axis='both', which='major', labelsize=fontsize)
twin2.tick_params(axis='both', which='major', labelsize=fontsize)
# Set the color of tick labels and spines for twin2
twin2.tick_params(axis='y', colors='coral')  # Set tick label color
twin2.spines['right'].set_color('coral')    # Set spine color

l1, = ax.plot(df_2017.index, df_2017['Q (m3/s)'], color='k', label='Daily discharge', lw=2)
l2, = twin2.plot(df_2017.index, df_2017['Mean'], color='coral',
              ls='--', label='Daily SSC')
# Combine the handles and labels from both axes
lines = [l1, l2]
labels = [line.get_label() for line in lines]
# Create a single legend
ax.legend(lines, labels, loc='upper left', fontsize = fontsize)

fig.savefig('Discharge_and_SPM_at_TT_2017.png', format='png')




# 16/10 : Plot of concentration vs discharge both at TT
only_ebb = True
fig, ax = plt.subplots(figsize=(15, 10))#, nrows=1, sharex=True)
fig.suptitle('Discharge vs Concentration at Trung Trang ' + str(year))
ax.grid(True, alpha=0.5)
ax.set_xlabel('Discharge ($m^{3}/s$)', fontsize=fontsize)
#ax.set_ylim(-500, 3000)
ax.set_ylabel('Concentration (mg/L)', fontsize=fontsize)
ax.scatter(filtered_Ebb_df[Q], filtered_Ebb_df[Ebb], color=sns.color_palette("colorblind")[2],
           label='Ebb tide daily mean', s= 3 )
ax.plot(filtered_Ebb_df[Q], predicted_values, label = str(np.round(slope,2))+'x'+str(np.round(intercept,2)),
        color=sns.color_palette("colorblind")[3])
ax.scatter(filtered_Ebb_df[Q], slope*filtered_Ebb_df[Q]+intercept, s=3, color=sns.color_palette("colorblind")[3],
           marker='x')
if not only_ebb :
    ax.scatter(Flood_df[Q], Flood_df[Flood], color=sns.color_palette("colorblind")[3],
               label='Flood tide daily mean', s = 3)
    #correlation_flood = Flood_df.corr()[Q][Flood]
ax.legend(fontsize=fontsize - 2)
if save :
    outfile = 'Discharge_VS_SPM_at_TT_'
    if only_ebb :
        outfile = outfile + 'only_ebb_'
    else :
        outfile = outfile + 'both_ebb_and_flood_'
    outfile = outfile + str(year) + '.png'
    fig.savefig(outfile, format='png')

# 16/10 Figure of predicted vs observed concetration values
only_ebb = True
fig, ax = plt.subplots(figsize=(15, 10))#, nrows=1, sharex=True)
fig.suptitle('Observed vs predicted concentration at Trung Trang ' + str(year))
ax.grid(True, alpha=0.5)
ax.set_xlabel('Observed concentration (mg/L)', fontsize=fontsize)
#ax.set_ylim(-500, 3000)
ax.set_ylabel('Predicted concentration (mg/L)', fontsize=fontsize)
ax.scatter(filtered_Ebb_df[Ebb], predicted_values, color='grey',
           s= 3, zorder = 3 )
r = np.round(stats.linregress(predicted_values, filtered_Ebb_df.dropna(how ='any')[Ebb])[2],4)
biases = predicted_values - filtered_Ebb_df.dropna(how ='any')[Ebb]
mean_bias = np.round(np.nanmean(biases),4) # Calculate the mean bias
label = 'RMSE=' + str(np.round(rmse,4)) + '\n r=' + str(r) + '\n b=' + str(mean_bias)
ax.plot(filtered_Ebb_df[Ebb], filtered_Ebb_df[Ebb], color='k', alpha=0.5, zorder = 1, label = label)
ax.legend(fontsize=fontsize - 2)
if save :
    outfile = 'Predicted_vs_observed_SSC_'
    if only_ebb :
        outfile = outfile + 'only_ebb_'
    else :
        outfile = outfile + 'both_ebb_and_flood_'
    outfile = outfile + str(year) + '.png'
    fig.savefig(outfile, format='png')

# Water level at trung trang and Hon Dau
# Figure of Water level in 2017 at TT and HD
fig, axs = plt.subplots(figsize=(18, 10), nrows=3, sharex=True)
fig.suptitle('Water level' + a + str(month) + '/' + str(year))
ax = axs[0]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level (m)', fontsize=fontsize)
# ax.set_ylim(-500, 3000)

ax = axs[1]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water_level (m)', fontsize=fontsize)
date_form = DateFormatter("%d/%m/%y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
ax.xaxis.set_major_formatter(date_form)
# ax.set_xlabel('Water level (cm)', fontsize=fontsize)
ax.set_xlabel('Time', fontsize=fontsize)

ax = axs[0]
ax.title.set_text('Trung Trang')
ax.plot(selected_data['Datetime'], selected_data[Wat_lev], color=sns.color_palette("colorblind")[0],
        label='Trung Trang', lw=3)
#ax.legend(fontsize=fontsize - 2)

ax = axs[1]
ax.title.set_text('Hon Dau')
ax.plot(selected_HD['Datetime'], selected_HD['Value'], color=sns.color_palette("colorblind")[2],
           label='Hon Dau', lw=3)
#ax.legend(fontsize=fontsize - 2)
if save:
    fig.savefig('Water_level_HD_TT' + str(month) + str(year))

# Figure of discharge and SPM at TT
fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
fig.suptitle('Discharge at Trung Trang ' + a + str(month) + '/' + str(year))
ax = axs[0]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Discharge ($m^{3}/s$)', fontsize=fontsize)
ax.set_ylim(-500, 3000)

ax = axs[1]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Discharge (mg/L)', fontsize=fontsize)
date_form = DateFormatter("%d/%m/%y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_formatter(date_form)
# ax.set_xlabel('Water level (cm)', fontsize=fontsize)
ax.set_xlabel('Time', fontsize=fontsize)

ax = axs[0]
ax.plot(selected_data['Datetime'], selected_data[Q], color=sns.color_palette("colorblind")[0],
        label='Hourly discharge')
ax.plot(daily_mean.index, daily_mean[Q], color=sns.color_palette("colorblind")[1],
        label='Daily discharge\n Monthly mean = ' + str(np.round(daily_mean[Q].mean(axis=0), 1)) + '($m^{3}/s$)')
ax.legend(fontsize=fontsize-2)

ax = axs[1]
ax.scatter(selected_SPM['Date'], selected_SPM[Ebb], color=sns.color_palette("colorblind")[2],
           label='Ebb tide daily mean')
ax.scatter(selected_SPM['Date'], selected_SPM[Flood], color=sns.color_palette("colorblind")[3],
           label='Flood tide daily mean')
ax.plot(selected_SPM['Date'], selected_SPM['Mean'], color=sns.color_palette("colorblind")[4],
        label='Mean daily discharge\n Monthly mean = ' + str(
            np.round(selected_SPM['Mean'].mean(axis=0), 1)) + '(mg/L)')
ax.legend(fontsize=fontsize - 2)
if save :
    fig.savefig('Discharge_and_SPM_at_TT_' + str(month) + str(year))


# 11/10 : 3 subplots one for ST, one for TT, one for HD for Water levels
fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
fig.suptitle('Water levels' + a + str(month) + '/' + str(year))

ax = axs[0]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level (m)', fontsize=fontsize)
ax.plot()

print('oj')


"""
# I sort the values by x and y values
data = list(zip(Ebb_df[Q], Ebb_df[Ebb]))
sorted_data = sorted(data, key=lambda x: x[0])
sorted_x_values, sorted_observed_values = zip(*sorted_data)
# Check if there are identical values in sorted_x_values :
if len(sorted_x_values) != len(set(sorted_x_values)):
    print("There are identical sorted_x_values.")
else:
    print("There are no identical sorted_x_values.")
# Creation of a sub_set to have several y_values sorted by x_values
x_y_mapping = {}
for x, y_values in zip(sorted_x_values, sorted_observed_values):
    if x in x_y_mapping:
        if isinstance(y_values, list):
            x_y_mapping[x].extend(y_values)
        else:
            x_y_mapping[x].append(y_values)
    else:
        x_y_mapping[x] = y_values if isinstance(y_values, list) else [y_values]
# Now you can access sub-tables for each unique x-value
for x, sub_table in x_y_mapping.items():
    print(f"x = {x}, y_values = {sub_table}")
"""