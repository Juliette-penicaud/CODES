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
def calculate_corr(data1, data2, year_constraint, year, month_constraint, month, datetime=True):
    correlations = []
    if datetime :
        max_lag = 5  # Maximum lag value to test
        if year_constraint and month_constraint:
            df1 = data1[(data1['Datetime'].dt.year == year) & (data1['Datetime'].dt.month == month)]
            df2 = data2[(data2['Datetime'].dt.year == year) & (data2['Datetime'].dt.month == month)]
        elif year_constraint:
            df1 = data1[(data1['Datetime'].dt.year == year)]
            df2 = data2[(data2['Datetime'].dt.year == year)]
        elif month_constraint:
            df1 = data1[data1['Datetime'].dt.month == month]
            df2 = data2[data2['Datetime'].dt.month == month]
        else:
            print('no constraint on year or month, I take the whole series')
            df1 = data1
            df2 = data2
    else :
        max_lag = 50
        if year_constraint and month_constraint:
            df1 = data1[(data1.index.year == year) & (data1.index.month == month)]
            df2 = data2[(data2.index.year == year) & (data2.index.month == month)]
        elif year_constraint:
            df1 = data1[(data1.index.year == year)]
            df2 = data2[(data2.index.year == year)]
        elif month_constraint:
            df1 = data1[data1.index.month == month]
            df2 = data2[data2.index.month == month]
        else:
            print('no constraint on year or month, I take the whole series')
            df1 = data1
            df2 = data2
    for lag in range(-max_lag, max_lag + 1):
        shifted_df2 = df2.shift(periods=lag)
        if datetime :
            m2 = pd.concat([df1.reset_index().drop('index', axis=1), shifted_df2.reset_index().drop('index', axis=1)],
                           axis=1)
            # correlation = m2['Q'].corr(m2['Q (m3/s)'])
        else :
            m2 = pd.concat([df1, shifted_df2],axis=1)
        correlation = m2.corr().iloc[0, 1]
        correlations.append((lag, correlation))
    # Find the lag with the highest correlation
    best_lag, best_correlation = max(correlations, key=lambda x: abs(x[1]))
    # print(f"Best lag: {best_lag}")
    # print(f"Best correlation: {best_correlation}")
    return best_lag, best_correlation
    # ON WATER LEVEL :
    # For the whole temporal serie  Best lag: 2 Best correlation: 0.9403780905327238
    # For 2022 : Best lag: 2 Best correlation: 0.9231507853916974
    # Monthly : worst for May to Sept, but still > 0.87
    # For january, Best lag: 2 Best correlation: 0.974414815770983 febr,Lag: 2 Corr: 0.9841633877568988 ,  march, Lag: 2 Corr: 0.98359689458487
    # april, Lag: 2 Corr: 0.985 may, Lag: 2 Corr: 0.940 june, Lag: 2 Corr: 0.874 july lag: 2 correlation: 0.8816781512215971
    # Aug, Lag: 2 Corr: 0.881 Sept, Lag: 2 Corr: 0.957 Oct, Lag: 2 Corr: 0.953  Nov  Lag: 2 Corr: 0.979 Dec, Lag: 2 Corr: 0.980


dict_month = {6: range(16, 19), 8: range(10, 14), 10: range(2, 6)}

file = '/home/penicaud/Documents/Data/Décharge_waterlevel/Data_2021-2022.xlsx'
save = False

columns_to_load = list(range(25))
# Water level at Trung Trang
df = pd.read_excel(file, sheet_name='water_level_TrungTrang2021-2022', usecols=columns_to_load, skiprows=4, nrows=730)
df = df.rename(columns={'Unnamed: 0': 'Date'})
melted_df = pd.melt(df, id_vars=["Date"], var_name="Hour", value_name="Value")
melted_df['Datetime'] = pd.to_datetime(melted_df['Date']) + pd.to_timedelta(melted_df['Hour'], unit='h')
melted_df.sort_values("Datetime", inplace=True)
melted_df = melted_df.rename(columns={'Value': 'Water level Trung Trang'})
melted_df.drop(['Date', 'Hour'], axis=1, inplace=True)

# Water level at Hon Dau
df2 = pd.read_excel(file, sheet_name='sea_level-HonDau_2021-2022', usecols=columns_to_load, skiprows=4, nrows=730)
df2 = df2.rename(columns={'Unnamed: 0': 'Date'})
melted_df2 = pd.melt(df2, id_vars=["Date"], var_name="Hour", value_name="Value")
melted_df2['Datetime'] = pd.to_datetime(melted_df2['Date']) + pd.to_timedelta(melted_df2['Hour'], unit='h')
melted_df2.sort_values("Datetime", inplace=True)
melted_df2 = melted_df2.rename(columns={'Value': 'Water level Hon Dau'})
melted_df2.drop(['Date', 'Hour'], axis=1, inplace=True)

merged_df = pd.merge(melted_df, melted_df2, on='Datetime', how='inner')

df_SPM = pd.read_excel(file, sheet_name='SPM_Trungtrang_2021-2022', usecols=list(range(3)), skiprows=2)
df_Q = pd.read_excel(file, sheet_name='Q_TrungTrang_2021-2022', skiprows=2)
df_Q['Datetime'] = pd.to_datetime(df_Q['Date']) + pd.to_timedelta(df_Q['Hour'], unit='h')
df_Q.drop(['Date', 'Hour'], axis=1, inplace=True)

merged_df = pd.merge_asof(merged_df, df_Q, on='Datetime', direction='nearest')

df_ST = pd.read_excel(file, sheet_name='Q_SonTay_2021-2022', skiprows=2, nrows=732)  # usecols=columns_to_load,
df_ST = df_ST.drop(['Unnamed: 0', 'Month', 'Day'], axis=1)

df_ST_SPM = pd.read_excel(file, sheet_name='SPM_SonTay_2021-2022', skiprows=2, nrows=732)  # usecols=columns_to_load,
df_ST_SPM = df_ST_SPM.applymap(lambda x: np.nan if not isinstance(x, (float, int)) else x)
df_ST['Concentration (g/m3)'] = df_ST_SPM['Concentration (g/m3)'].copy()
# df_ST = df_ST.rename(columns={'Q (m3/s)': 'Q'})

# FIGURE
fontsize = 25
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
s = 25


months = [6]  # , 8, 10]
for month in months:
    a = '0' if month < 10 else ''
    year_constraint = False
    year = 2022
    month_constraint = False
    # month = 6
    if year_constraint and month_constraint:
        selected_data = merged_df[(merged_df['Datetime'].dt.year == year) & (merged_df['Datetime'].dt.month == month)]
        selected_SPM = df_SPM[(df_SPM['Date'].dt.year == year) & (df_SPM['Date'].dt.month == month)]
        selected_ST = df_ST[(df_ST['Datetime'].dt.year == year) & (df_ST['Datetime'].dt.month == month)]
    elif year_constraint:
        selected_data = merged_df[merged_df['Datetime'].dt.year == year]
        selected_SPM = df_SPM[(df_SPM['Date'].dt.year == year)]
        selected_ST = df_ST[(df_ST['Datetime'].dt.year == year)]
    elif month_constraint:
        selected_data = merged_df[merged_df['Datetime'].dt.month == month]
        selected_SPM = df_SPM[(df_SPM['Date'].dt.month == month)]
        selected_ST = df_ST[(df_ST['Datetime'].dt.month == month)]
    else:
        print('no constraint on year or month, I take the whole series')
        selected_data = merged_df
        selected_SPM = df_SPM
        selected_ST = df_ST
    daily_mean = selected_data.resample('D', on='Datetime').mean()
    monthly_mean = selected_data.resample('M', on='Datetime').mean()
    # selected_SPM = selected_SPM.replace('-', np.nan)
    # selected_SPM = selected_SPM.replace('x', np.nan)
    # selected_SPM = selected_SPM.applymap(lambda x: np.nan if not isinstance(x, (float, int)) else x)
    selected_SPM[["Flood tide", "Ebb tide"]] = selected_SPM[["Flood tide", "Ebb tide"]].applymap(
        lambda x: np.nan if not isinstance(x, (float, int)) else x)
    selected_SPM['Mean'] = (selected_SPM['Flood tide'].values + selected_SPM['Ebb tide'].values) / 2

daily_mean = merged_df.resample('D', on='Datetime').mean()
monthly_mean = merged_df.resample('M', on='Datetime').mean()
# Sum over the survey period :
Q_mean_june = np.sum(daily_mean['Q'].loc['2022-06-' + str(i)] for i in range(16, 19)) / 3
Q_mean_august = np.sum(daily_mean['Q'].loc['2022-08-' + str(i)] for i in range(10, 13)) / 3
Q_mean_octobre = np.sum(daily_mean['Q'].loc['2022-10-0' + str(i)] for i in range(2, 5)) / 3

daily_mean_Q_TT = daily_mean.copy()
daily_mean_Q_TT = daily_mean_Q_TT.reset_index()
daily_mean_Q_TT.drop(['Water level Trung Trang', 'Water level Hon Dau'], axis=1, inplace=True)
# Calculation of the lag correlation of
best_lag = calculate_corr(daily_mean_Q_TT, selected_ST, year_constraint, year, month_constraint, month)[0]
best_corr = calculate_corr(daily_mean_Q_TT, selected_ST, year_constraint, year, month_constraint, month)[1]

shifted_ST = selected_ST.copy()
shifted_ST['Datetime'] = selected_ST['Datetime'] + pd.DateOffset(
    days=best_lag)  # add the best_lag correlation to plot the data.
percentage_ST = 14.5
discharge_ST_survey = pd.DataFrame()
for months in [6, 8, 10]:
    for d in dict_month[months]:
        d1 = shifted_ST[(df_ST['Datetime'].dt.year == year) & (shifted_ST['Datetime'].dt.month == months) & (
                    shifted_ST['Datetime'].dt.day == d)]
        discharge_ST_survey = pd.concat([discharge_ST_survey, d1], ignore_index=True)
discharge_ST_survey[str(percentage_ST) + '% Q'] = percentage_ST / 100 * discharge_ST_survey['Q (m3/s)']

discharge_TT_survey = pd.DataFrame()
for months in [6, 8, 10]:
    for d in dict_month[months]:
        d2 = daily_mean_Q_TT[(daily_mean_Q_TT['Datetime'].dt.year == year) &
                             (daily_mean_Q_TT['Datetime'].dt.month == months) &
                             (daily_mean_Q_TT['Datetime'].dt.day == d)]
        discharge_TT_survey = pd.concat([discharge_TT_survey, d2], ignore_index=True)
# Print with the 14.5% (Vinh et al 2014)
# discharge_ST_survey['Q (m3/s)']*0.145

# Plot ST and TT discharge
with_survey = False
with_daily_mean = True
with_monthly_mean = True
lw = 2

# 2/10/23 : Plot the discharge at TT and 14.5% of the ST discharge over the year with best_lag shift.
fig, ax = plt.subplots(figsize=(18, 10))
#fig.suptitle('Daily discharge at TT and ST', fontsize=fontsize)
ax.grid(True, alpha=0.5)
ax.set_ylabel('Discharge (m3/s)', fontsize=fontsize - 2)
ax.set_ylim(0, 2300)
date_form = DateFormatter("%m/%y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize - 2)
label = str(percentage_ST) + ' % of daily mean at ST\n with lag = ' + str(best_lag) + 'h, r=' + str(
    np.round(best_corr, 4))
ax.plot(shifted_ST['Datetime'], percentage_ST / 100 * shifted_ST['Q (m3/s)'], color='grey', lw=lw, label=label)
if with_daily_mean:
    ax.plot(daily_mean.index, daily_mean['Q'], color='blue', label='Daily mean at TT', lw=lw)
    if with_survey:
        d1 = daily_mean[(daily_mean.index.month == 6) & (daily_mean.index.day.isin([16, 17, 18]))]
        d2 = daily_mean[(daily_mean.index.month == 8) & (daily_mean.index.day.isin([10, 11, 12]))]
        d3 = daily_mean[(daily_mean.index.month == 10) & (daily_mean.index.day.isin([2, 3, 4]))]
        count = 0
        for d in [d1, d2, d3]:
            if count == 0:
                ax.plot(d.index, d['Q'], color='orange', label='Surveys', lw=lw)
            else:
                ax.plot(d.index, d['Q'], color='orange', lw=lw)
            count = count + 1
if with_monthly_mean:
    ax.plot(monthly_mean.index, monthly_mean['Q'], color='black', label='Monthly mean at TT', alpha=0.8, lw=lw)
    monthly_mean_ST = selected_ST.resample('M', on='Datetime').mean()
    ax.plot(monthly_mean_ST.index, percentage_ST / 100 * monthly_mean_ST['Q (m3/s)'], color='green',
            label=str(percentage_ST) + '% of monthly mean at ST', alpha=0.8, lw=lw)
plt.legend(fontsize=fontsize - 4)
if save:
    out = 'Daily_mean_discharge_ST_TT'  # 'Discharge_monthly_TT_year_'
    if with_survey:
        out = out + 'withsurveys_'
    fig.savefig(out + str(year))

fig, ax = plt.subplots(figsize=(18, 10))
#fig.suptitle('Daily discharge at TT and ST', fontsize=fontsize)
ax.grid(True, alpha=0.5)
ax.set_ylabel('Discharge (m${³}$/s)', fontsize=fontsize - 2)
ax.set_ylim(0, 2300)
date_form = DateFormatter("%m/%y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize - 2)
# ax.plot(monthly_mean.index, monthly_mean['Q'], color='black', label='Monthly mean at TT')
ax.plot(selected_ST['Datetime'], percentage_ST / 100 * selected_ST['Q (m3/s)'], color='grey', lw=lw)
# label = str(percentage_ST) + '% of daily mean at ST')
label = 'Discharge at Trung Trang\n lag = ' + str(best_lag) + 'h, corr=' + str(np.round(best_corr, 4))
if with_daily_mean:
    ax.plot(daily_mean.index, daily_mean['Q'], color='blue', label='Daily mean at TT', lw=lw)
    if with_survey:
        d1 = daily_mean[(daily_mean.index.month == 6) & (daily_mean.index.day.isin([16, 17, 18]))]
        d2 = daily_mean[(daily_mean.index.month == 8) & (daily_mean.index.day.isin([10, 11, 12]))]
        d3 = daily_mean[(daily_mean.index.month == 10) & (daily_mean.index.day.isin([2, 3, 4]))]
        count = 0
        for d in [d1, d2, d3]:
            if count == 0:
                ax.plot(d.index, d['Q'], color='orange', label='Surveys', lw=lw)
            else:
                ax.plot(d.index, d['Q'], color='orange')
            count = count + 1
plt.legend(fontsize=fontsize - 4)
if save:
    out = 'Daily_mean_discharge_ST_TT'  # 'Discharge_monthly_TT_year_'
    if with_survey:
        out = out + 'withsurveys_'
    fig.savefig(out + str(year))

############## Figure of discharge at TT,  monthly mean and daily
with_ST = True
year_2022 = False
fig, ax = plt.subplots(figsize=(18, 10))
ax.grid(True, alpha=0.5)
ax.set_ylabel('Discharge (m${³}$/s)', fontsize=fontsize - 2)
ax.set_ylim(0, 2300)
date_form = DateFormatter("%m/%y")  # Define the date format
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize - 5)
# ax.plot(monthly_mean.index, monthly_mean['Q'], color='black', label='Monthly mean at TT')
if year_2022:
    mean_year_daily = daily_mean_Q_TT['Q'].loc[(daily_mean_Q_TT['Datetime'].dt.year == 2022) == True].mean()
else :
    mean_year_daily = daily_mean_Q_TT['Q'].loc[((daily_mean_Q_TT['Datetime'].dt.year == 2022) & (daily_mean_Q_TT['Datetime'].dt.year == 2022)) == True].mean()
label1 = 'Daily mean at TT\n (mean = ' + str(np.round(mean_year_daily,0)) + ' m${³}$/s)'
ax.plot(daily_mean_Q_TT['Datetime'], daily_mean_Q_TT['Q'], color='grey', lw=lw, label=label1)
ax.plot(monthly_mean.index, monthly_mean['Q'], color='k', lw=lw, label='Monthly mean at TT')
#legend = ax.legend(fontsize=fontsize - 5 )
if with_ST :
    #monthly_mean_ST = selected_ST.resample('M', on='Datetime').mean()
    #ax.plot(monthly_mean_ST.index, percentage_ST / 100 * monthly_mean_ST['Q (m3/s)'], color='plum',
    #        label=str(percentage_ST) + '% of monthly mean at ST', alpha=0.8, lw=lw)
    ax.plot(selected_ST['Datetime'], percentage_ST / 100 * selected_ST['Q (m3/s)'], color='plum', lw=lw,
            label=str(percentage_ST) + '% of daily mean at ST')

if with_survey :
    d1 = daily_mean[(daily_mean.index.month == 6) & (daily_mean.index.day.isin([16, 17, 18])) & (daily_mean.index.year == 2022)]
    d2 = daily_mean[(daily_mean.index.month == 8) & (daily_mean.index.day.isin([10, 11, 12])) & (daily_mean.index.year == 2022)]
    d3 = daily_mean[(daily_mean.index.month == 10) & (daily_mean.index.day.isin([2, 3, 4])) & (daily_mean.index.year == 2022)]
    count = 0
    for d in [d1, d2, d3]:
        if count == 0:
            ax.plot(d.index, d['Q'], color='orange', label='Surveys period', lw=lw, ls='--')
        else:
            ax.plot(d.index, d['Q'], color='orange', lw=lw, ls='--')
        ax.axvspan(d.index[0], d.index[-1], ymin=-10, ymax=3000, facecolor='orange', alpha=0.2)
        count = count + 1
legend = ax.legend(fontsize=fontsize - 5 )
if year_2022 :
    ax.set_xlim(datetime(2022,1,1), datetime(2023,1,1))
else :
    ax.set_xlim(datetime(2021, 1, 1), datetime(2023, 1, 1))
for label in legend.get_texts():
    label.set_fontsize(15)
out = 'discharge_monthly_and_daily_at_TT_'
if year_2022 :
    out = out + '2022'
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
else :
    out = out + '2021-2022'
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
if with_survey:
    out = out + '_with_survey'
if with_ST :
    out = out + '_with_ST'
out= out + '.png'
fig.savefig(out, format='png')

# COMBIEN DE JOURS AU DESSUS DE 2000 m³/s :
daily_mean_2022 = daily_mean[(daily_mean.index.year == 2022)]
daily_mean_2021 = daily_mean[(daily_mean.index.year == 2021)]

####################################### Water level at HD and TT
year_constraint = True
year = 2021
month_constraint = True
months = np.arange(1, 13, 1)
title = 'Water levels '
Wat_HD = merged_df.columns[2]
D = merged_df.columns[1]
Wat_TT = merged_df.columns[0]
if year_constraint and not month_constraint:
    selected_data = merged_df[(merged_df['Datetime'].dt.year == year)]
    title = title + str(year)
elif not year_constraint and not month_constraint:
    print('no constraint on year or month, I take the whole series')
    selected_data = merged_df
    title = title + '2021-2022'
else:
    lags, corrs, corrs2 = [], [], []
    for month in months:
        a = '0' if month < 10 else ''
        title = 'Water levels '
        if year_constraint and month_constraint:
            selected_data = merged_df[(merged_df['Datetime'].dt.year == year) &
                                      (merged_df['Datetime'].dt.month == month)]
            title = title + a + str(month) + '/' + str(year)
            lag, corr = calculate_corr(selected_data[[Wat_TT, D]], selected_data[[Wat_HD, D]], year_constraint, year,
                                       month_constraint, month)
            lags.append(lag)
            corrs.append(corr)
        elif month_constraint and not year_constraint:
            selected_data = merged_df[merged_df['Datetime'].dt.month == month]
            title = title + a + str(month)
            lag, corr = calculate_corr(selected_data[[Wat_TT, D]], selected_data[[Wat_HD, D]], year_constraint, year,
                                       month_constraint, month)
            lags.append(lag)
            corrs.append(corr)
    for c in corrs:
        corrs2.append(np.round(c, 4))

# Plot of the correlation vs Q :
corrs_20212022 = corrs_2021 + corrs_2022
data_corr_Q = monthly_mean[['Q']].copy()
data_corr_Q['corr'] = corrs_20212022
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_ylabel('correlation r', fontsize=fontsize)
ax.set_xlabel('Q (m3/s)', fontsize=fontsize)
ax.set_xlim(200,1300)
ax.set_ylim(0.82,1)
ax.scatter(data_corr_Q['Q'], data_corr_Q['corr'], color='black',s=5)
ax.grid(True, alpha = 0.5)
slope, intercept, r_value, p_value, std_err = stats.linregress(data_corr_Q['Q'], data_corr_Q['corr'])
label = "{:.1e}".format(slope) + ' x + '+str(np.round(intercept,2))+' r='+str(np.round(r_value,2))
# ax.plot(np.arange(0.83,1,0.01), slope*np.arange(0.83,1,0.01)+intercept, alpha=0.5, lw = 1, color='grey', label=label)
x = np.arange(100,1500,1)
ax.plot(x, slope*x+intercept, alpha = 0.5, lw = 1, color='grey' , label=label )
ax.legend()
fig.savefig('Debit_correlation_relation_2years.png', format='png')

# 26/10 : I resampled the data in order to have a 10 mn value, to have a finer value of the lag and see if
# it is the reason why its quality decreases with discharge increase
resampled_series = merged_df.copy()
resampled_series = resampled_series.set_index('Datetime')
resampled_series = resampled_series.resample('5T').asfreq() # Resample by adding values every 5T
interpolated_series = resampled_series.interpolate(method='linear')

year_constraint = True
all_year = True
year = 2022
if all_year :
    year = [2021, 2022]
month_constraint = True
months = np.arange(1, 13, 1)
title = 'Water levels '
lags, corrs, corrs2 = [], [], []
if year_constraint and not month_constraint:
    selected_interp = interpolated_series[(interpolated_series.index.year == year)]
    title = title + str(year)
elif not year_constraint and not month_constraint:
    print('no constraint on year or month, I take the whole series')
    selected_interp = interpolated_series
    title = title + '2021-2022'
else:
    lags, corrs, corrs2 = [], [], []
    if all_year :
        for y in year :
            months = np.arange(1, 13, 1)
            for month in months:
                a = '0' if month < 10 else ''
                selected_interp = interpolated_series[(interpolated_series.index.month == month) &
                                                      (interpolated_series.index.year == y)]
                title = title + a + str(month) + '/' + str(y)
                lag, corr = calculate_corr(selected_interp[[Wat_TT]], selected_interp[[Wat_HD]], year_constraint, y,
                                           month_constraint, month, datetime=False)
                lags.append(lag)
                corrs.append(corr)
    else :
        for month in months:
            a = '0' if month < 10 else ''
            title = 'Water levels '
            if year_constraint and month_constraint:
                selected_interp = interpolated_series[(interpolated_series.index.year == year) &
                                          (interpolated_series.index.month == month)]
                title = title + a + str(month) + '/' + str(year)
                lag, corr = calculate_corr(selected_interp[[Wat_TT]], selected_interp[[Wat_HD]], year_constraint, year,
                                           month_constraint, month, datetime=False)
                lags.append(lag)
                corrs.append(corr)
            elif month_constraint and not year_constraint:
                selected_interp = interpolated_series[interpolated_series.index.month == month]
                title = title + a + str(month)
                lag, corr = calculate_corr(selected_interp[[Wat_TT]], selected_interp[[Wat_HD]], year_constraint, year,
                                           month_constraint, month, datetime=False)
                lags.append(lag)
                corrs.append(corr)
    for c in corrs:
        corrs2.append(np.round(c, 4))

lags_hours = []
for l in lags :
    delta_minutes = timedelta(minutes=5)
    delta_seconds = delta_minutes * l
    lags_hours.append(delta_seconds)

data_phase_Q = monthly_mean[['Q']].copy()
data_phase_Q["lag"] = lags_hours
data_phase_Q["corr"] = corrs
# Plot de la corrélation fonction du débit.
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_ylabel('correlation r', fontsize=fontsize)
ax.set_xlabel('Q (m3/s)', fontsize=fontsize)
ax.set_xlim(200,1300)
ax.set_ylim(0.82,1)
ax.scatter(data_phase_Q['Q'], data_phase_Q['corr'], color='black',s=5)
ax.grid(True, alpha = 0.5)
slope, intercept, r_value, p_value, std_err = stats.linregress(data_phase_Q['Q'], data_phase_Q['corr'])
label = "{:.1e}".format(slope) + ' x + '+str(np.round(intercept,2))+' r='+str(np.round(r_value,2))
# ax.plot(np.arange(0.83,1,0.01), slope*np.arange(0.83,1,0.01)+intercept, alpha=0.5, lw = 1, color='grey', label=label)
x = np.arange(100,1500,1)
ax.plot(x, slope*x+intercept, alpha = 0.5, lw = 1, color='grey' , label=label )
ax.legend()
fig.savefig('Debit_correlation_avecvalinterp_relation_2years.png', format='png')
# Plot du lag fonction du débit
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_ylabel('phase lag', fontsize=fontsize)
ax.set_xlabel('Q (m3/s)', fontsize=fontsize)
ax.set_xlim(200,1300)
#ax.set_ylim(0.82,1)
y_second = [td.total_seconds() for td in data_phase_Q['lag']]
ax.scatter(data_phase_Q['Q'], y_second, color='black',s=5)
ax.grid(True, alpha = 0.5)
slope, intercept, r_value, p_value, std_err = stats.linregress(data_phase_Q['Q'], y_second)
label = "{:.1e}".format(slope) + ' x + '+str(np.round(intercept,2))+' r='+str(np.round(r_value,2))
# ax.plot(np.arange(0.83,1,0.01), slope*np.arange(0.83,1,0.01)+intercept, alpha=0.5, lw = 1, color='grey', label=label)
x = np.arange(100,1500,1)
ax.plot(x, slope*x+intercept, alpha = 0.5, lw = 1, color='grey' , label=label )
ax.legend()
fig.savefig('Debit_lag_avecvalinterp_relation_2years.png', format='png')
# Puis il faudra aussi la différence d'amplitude ou indicateur d'amplitude fonction du débit

######################################################## Figure classique water levels TT et HD
fig, ax = plt.subplots(figsize=(18, 10))
fig.suptitle(title)
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level (m)', fontsize=fontsize)
# ax.set_ylim(-50, 450)
if year_constraint and not month_constraint:
    date_form = DateFormatter("%m/%Y")  # Define the date format
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
elif not year_constraint and month_constraint:
    date_form = DateFormatter("%d/%m")  # Define the date format
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize)
ax.plot(selected_data[D], selected_data[Wat_HD] / 100 - np.average(selected_data[Wat_HD] / 100), color='black',
        label=Wat_HD)
lag, corr = calculate_corr(selected_data[[Wat_TT, D]], selected_data[[Wat_HD, D]], year_constraint, year,
                           month_constraint, month)
label = Wat_TT + '\n lag = ' + str(lag) + 'h, r=' + str(np.round(corr, 4))
ax.plot(selected_data[D], selected_data[Wat_TT] / 100 - np.average(selected_data[Wat_TT] / 100), color='grey',
        label=label)
plt.legend(fontsize=fontsize)
if save:
    outfile = 'Waterlevel_HD_TT_'
    if year_constraint and not month_constraint:
        ax.set_xlim(datetime(year, 1, 1), datetime(year + 1, 1, 2))
        outfile = outfile + 'allyear' + str(year)
    elif not year_constraint and not month_constraint:
        ax.set_xlim(datetime(2021, 1, 1), datetime(20223, 1, 2))
        outfile = outfile + 'bothyears2021-2022'
    elif not year_constraint and month_constraint:
        outfile = outfile + '2years_month_' + str(month)
    elif year_constraint and month_constraint:
        outfile = outfile + a + str(month) + str(year)
        if month == 12:
            ax.set_xlim(datetime(year, month, 1), datetime(year + 1, 1, 2))
        else:
            ax.set_xlim(datetime(year, month, 1), datetime(year, month + 1, 2))

    outfile = outfile + '.png'
    fig.savefig(outfile, format='png')

    ########################################## Discharge at trung trang
    fontsize = 12
    fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
    fig.suptitle('Discharge at Trung Trang ')  # + a + str(month) + '/' + str(year))
    ax = axs[0]
    ax.grid(True, alpha=0.5)
    ax.set_ylabel('Discharge ($m^{3}/s$)', fontsize=fontsize)
    # ax.set_ylim(-2000, 2500)

    ax = axs[1]
    ax.grid(True, alpha=0.5)
    ax.set_ylabel('Discharge (mg/L)', fontsize=fontsize)
    date_form = DateFormatter("%m/%Y")  # Define the date format
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_formatter(date_form, fontsize=fontsize - 2)
    # ax.set_xlabel('Water level (cm)', fontsize=fontsize)
    ax.set_xlabel('Time', fontsize=fontsize)

    ax = axs[0]
    ax.plot(selected_data['Datetime'], selected_data['Q'], color=sns.color_palette("colorblind")[0],
            label='Hourly discharge', lw=2)
    ax.plot(daily_mean.index, daily_mean['Q'], color=sns.color_palette("colorblind")[1],
            label='Daily discharge\n Monthly mean = ' + str(np.round(daily_mean['Q'].mean(axis=0), 1)) + '($m^{3}/s$)',
            lw=2)
    ax.legend(fontsize=fontsize - 2)

    ax = axs[1]
    ax.scatter(selected_SPM['Date'], selected_SPM['Ebb tide'], color=sns.color_palette("colorblind")[2],
               label='Ebb tide daily mean', s=2)
    ax.scatter(selected_SPM['Date'], selected_SPM['Flood tide'], color=sns.color_palette("colorblind")[3],
               label='Flood tide daily mean', s=2)
    ax.plot(selected_SPM['Date'], selected_SPM['Mean'], color=sns.color_palette("colorblind")[4],
            label='Mean daily discharge\n Monthly mean = ' + str(
                np.round(selected_SPM['Mean'].mean(axis=0), 1)) + '(mg/L)', lw=2)
    ax.legend(fontsize=fontsize - 2)
    fig.savefig('Discharge_and_SPM_at_TT_')  # + str(month) + str(year))

# Discharge at Son Tay correlation with MES
fontsize = 12
fig, ax = plt.subplots(figsize=(18, 10))  # , sharex=True)
fig.suptitle('Discharge and SPM at Son Tay')  # + a + str(month) + '/' + str(year))
ax.grid(True, alpha=0.5)
ax.set_ylabel('Discharge ($m^{3}/s$)', fontsize=fontsize)
twin2 = ax.twinx()
twin2.set_ylabel('Concentration (g/m3)')
# ax.set_ylim(-2000, 2500)
ax.grid(True, alpha=0.5)

date_form = DateFormatter("%m/%Y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize)
ax.plot(df_ST['Datetime'], df_ST['Q (m3/s)'], color=sns.color_palette("colorblind")[1],
        label='Daily discharge', lw=2)
ax.legend('left')
twin2.scatter(df_ST['Datetime'], df_ST['Concentration (g/m3)'], color=sns.color_palette("colorblind")[2],
              s=2, label='Concentration MES')
twin2.legend('right')
fig.savefig('Discharge_and_SPM_at_ST', format='png')  # + str(month) + str(year))

########### 14% ST for TT and only ebb tide MES #######
fig, ax = plt.subplots(figsize=(18, 10))  # , sharex=True)
fig.suptitle('Discharge and SPM Trung Trang')  # + a + str(month) + '/' + str(year))
ax.grid(True, alpha=0.5)
ax.set_ylabel('Discharge ($m^{3}/s$)', fontsize=fontsize)
twin2 = ax.twinx()
twin2.set_ylabel('Concentration (g/m3)')
# ax.set_ylim(-2000, 2500)
ax.grid(True, alpha=0.5)
date_form = DateFormatter("%m/%Y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize)
ax.plot(df_ST['Datetime'], 0.14 * df_ST['Q (m3/s)'], color=sns.color_palette("colorblind")[1],
        label='14% of Son Tay daily discharge', lw=2)
ax.legend(loc='upper left')
twin2.scatter(df_SPM['Date'], df_SPM['Ebb tide'], color=sns.color_palette("colorblind")[2],
              s=2, label='Concentration MES at ebb tide')
ax.legend(loc='upper right')
fig.savefig('Discharge_14%ST_and_ebb_SPM_at_TT', format='png')

########### max positive discharge at TT and only ebb tide MES #######
fig, ax = plt.subplots(figsize=(18, 10))  # , sharex=True)
fig.suptitle('Max discharge and Ebb SPM Trung Trang')  # + a + str(month) + '/' + str(year))
ax.grid(True, alpha=0.5)
ax.set_ylabel('Discharge ($m^{3}/s$)', fontsize=fontsize)
twin2 = ax.twinx()
twin2.set_ylabel('Concentration (g/m3)')
# ax.set_ylim(-2000, 2500)
ax.grid(True, alpha=0.5)
date_form = DateFormatter("%m/%Y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize)
# Selecting the maximum value per day
df_Q['Date'] = df_Q['Datetime'].dt.date
max_value_indices = df_Q.groupby('Date')['Q'].idxmax()
max_values_per_day = df_Q.loc[max_value_indices]
ax.plot(max_values_per_day['Date'], max_values_per_day['Q'], color=sns.color_palette("colorblind")[1],
        label='Daily maximum discharge', lw=2)
ax.legend(loc='upper left')
twin2.scatter(df_SPM['Date'], df_SPM['Ebb tide'], color=sns.color_palette("colorblind")[2],
              s=2, label='SPM at ebb tide')
ax.legend(loc='upper right')
fig.savefig('Discharge_maxTT_and_ebb_SPM_at_TT', format='png')

#######################################  ALL SPM DATA ###############################"
####################################   2021-2022    #################################"
df_Q['Date'] = df_Q['Datetime'].dt.date
Q = 'Q (m3/s)'
Ebb = 'mean in ebb tide'
Flood = 'mean in flood tide'
Wat_lev = 'Water level (cm)'
df_Q = df_Q.rename(columns={'Q': Q})
df_SPM = df_SPM.rename(columns={"Ebb tide": Ebb, "Flood tide": Flood})
df_SPM[[Flood, Ebb]] = df_SPM[[Flood, Ebb]].applymap(lambda x: np.nan if not isinstance(x, (float, int)) else x)

max_value_indices_2022 = df_Q.groupby('Date')[Q].idxmax()
max_values_per_day_2022 = df_Q.loc[max_value_indices_2022]
max_values_per_day_2022['Date'] = pd.to_datetime(
    max_values_per_day_2022['Date'])  # Changing the columns type from object to datetime
# Find the minimum AND negative discharge data
min_value_indices_2022 = df_Q.groupby('Date')[Q].idxmin()
min_values_per_day_2022 = df_Q.loc[min_value_indices_2022]
neg_values_per_day_2022 = min_values_per_day_2022.loc[min_values_per_day_2022[Q] < 0]
neg_values_per_day_2022['Date'] = pd.to_datetime(
    neg_values_per_day_2022['Date'])  # Changing the columns type from object to datetime
# Split Ebb and flood mean spm to recreate a corresponding table (max values velocity and Ebb mean SPM ; neg max values
# of flood tide and mean Flood spm)
Ebb_SPM_2022 = df_SPM[['Date', Ebb]].copy()
Flood_SPM_2022 = df_SPM[['Date', Flood]].copy()
Flood_df_2022 = neg_values_per_day_2022.merge(Flood_SPM_2022, on='Date', how='inner')
Ebb_df_2022 = max_values_per_day_2022.merge(Ebb_SPM_2022, on='Date', how='inner')

######################### 2017 Data SPM Water level and Discharge ##################################################
path = '/home/penicaud/Documents/Data/Décharge_waterlevel/'
file = path + 'Data_2017.xlsx'
columns_to_load = list(range(2, 6))
# Water level at Trung Trang and discharge
TT_2017 = pd.read_excel(file, sheet_name='Q_trungtrang_vanuc_2017', usecols=columns_to_load, skiprows=3, nrows=8772)
TT_2017['Datetime'] = pd.to_datetime(TT_2017['Date']) + pd.to_timedelta(TT_2017['Time (hours)'], unit='h')
TT_2017.sort_values("Datetime", inplace=True)
TT_2017.drop(['Date', 'Time (hours)'], axis=1, inplace=True)
# SPM DATA
df_SPM_2017 = pd.read_excel(file, sheet_name='SPM_TRungtrang_vanuc17', usecols=list(range(2, 5)), skiprows=2, nrows=365)
# 11/10 : Open and use the 2017 data at TT and HD
df_HD_2017 = pd.read_excel(file, sheet_name='Water_level_HonDau2017')

# Manip over the tables to find the datetime of the maximum and minimum (only negative) discharge
TT_2017['Date'] = TT_2017['Datetime'].dt.date
# Find the maximum discharge data.
max_value_indices = TT_2017.groupby('Date')[Q].idxmax()
max_values_per_day = TT_2017.loc[max_value_indices]
max_values_per_day['Date'] = pd.to_datetime(
    max_values_per_day['Date'])  # Changing the columns type from object to datetime
# Find the minimum AND negative discharge data
min_value_indices = TT_2017.groupby('Date')[Q].idxmin()
min_values_per_day = TT_2017.loc[min_value_indices]
neg_values_per_day = min_values_per_day.loc[min_values_per_day[Q] < 0]
neg_values_per_day['Date'] = pd.to_datetime(
    neg_values_per_day['Date'])  # Changing the columns type from object to datetime
# Split Ebb and flood mean spm to recreate a corresponding table (max values velocity and Ebb mean SPM ; neg max values
# of flood tide and mean Flood spm)
Ebb_SPM = df_SPM_2017[['Date', Ebb]].copy()
Flood_SPM = df_SPM_2017[['Date', Flood]].copy()
Flood_df_2017 = neg_values_per_day.merge(Flood_SPM, on='Date', how='inner')
Ebb_df_2017 = max_values_per_day.merge(Ebb_SPM, on='Date', how='inner')
##########################################################################################
# CONCAT ALL
Ebb_df = pd.concat([Ebb_df_2022, Ebb_df_2017])
##########################################################################################
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
save = True
months = [1]
fontsize = 12

# First, I remove the outliers to derive the trend. 1 : selection of the outliers thanks to the z-score method :
# Calculate Z-scores for the 'y' column
z_scores = np.abs((Ebb_df[Ebb] - Ebb_df[Ebb].mean()) / Ebb_df[Ebb].std())
# Set a threshold for outliers (e.g., Z-score > 2)
threshold = 2
# Filter out the outliers
filtered_Ebb_df = Ebb_df[z_scores <= threshold]  # Now 'filtered_df' contains the data without outliers

# Calculation of the parameters for the predicted values :
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_Ebb_df.dropna(how='any')[Q],
                                                               filtered_Ebb_df.dropna(how='any')[Ebb])
predicted_values = slope * filtered_Ebb_df[Q] + intercept

errors = [observed - predicted for observed, predicted in zip(filtered_Ebb_df[Ebb], predicted_values)]
squared_errors = [np.mean(np.array(errors_i) ** 2) for errors_i in errors]
mean_squared_error = np.nanmean(squared_errors)
rmse = np.sqrt(mean_squared_error)

# 17/10 : Plot of concentration vs discharge both at TT FOR DATA of 2017,2021 2022
only_ebb = True
fig, ax = plt.subplots(figsize=(15, 10))  # , nrows=1, sharex=True)
fig.suptitle('Discharge vs Concentration at Trung Trang')
ax.grid(True, alpha=0.5)
ax.set_xlabel('Discharge ($m^{3}/s$)', fontsize=fontsize)
# ax.set_ylim(-500, 3000)
ax.set_ylabel('Concentration (mg/L)', fontsize=fontsize)
ax.scatter(filtered_Ebb_df[Q], filtered_Ebb_df[Ebb], color=sns.color_palette("colorblind")[2],
           label='Ebb tide daily mean', s=3)
ax.plot(filtered_Ebb_df[Q], predicted_values, label=str(np.round(slope, 2)) + 'x +' + str(np.round(intercept, 2)),
        color=sns.color_palette("colorblind")[3])
# ax.scatter(filtered_Ebb_df[Q], slope*filtered_Ebb_df[Q]+intercept, s=3, color=sns.color_palette("colorblind")[3],
#           marker='x')
if not only_ebb:
    ax.scatter(Flood_df[Q], Flood_df[Flood], color=sns.color_palette("colorblind")[3],
               label='Flood tide daily mean', s=3)
    # correlation_flood = Flood_df.corr()[Q][Flood]
ax.legend(fontsize=fontsize - 2)
if save:
    outfile = 'Discharge_VS_SPM_at_TT_'
    if only_ebb:
        outfile = outfile + 'only_ebb_'
    else:
        outfile = outfile + 'both_ebb_and_flood_'
    outfile = outfile + 'over3years' + '.png'
    fig.savefig(outfile, format='png')

# 17/10 Figure of predicted vs observed concetration values
only_ebb = True
fig, ax = plt.subplots(figsize=(15, 10))  # , nrows=1, sharex=True)
fig.suptitle('Observed vs predicted concentration at Trung Trang')
ax.grid(True, alpha=0.5)
ax.set_xlabel('Observed concentration (mg/L)', fontsize=fontsize)
# ax.set_ylim(-500, 3000)
ax.set_ylabel('Predicted concentration (mg/L)', fontsize=fontsize)
ax.scatter(filtered_Ebb_df[Ebb], predicted_values, color='grey', s=3, zorder=3)
r = np.round(stats.linregress(predicted_values, filtered_Ebb_df[Ebb])[2], 4)
biases = predicted_values - filtered_Ebb_df.dropna(how='any')[Ebb]
mean_bias = np.round(np.nanmean(biases), 4)  # Calculate the mean bias
label = 'RMSE=' + str(np.round(rmse, 4)) + '\n r=' + str(r) + '\n b=' + str(mean_bias)
ax.plot(filtered_Ebb_df[Ebb], filtered_Ebb_df[Ebb], color='k', alpha=0.5, zorder=1, label=label)
ax.legend(fontsize=fontsize - 2)
if save:
    outfile = 'Predicted_vs_observed_SSC_'
    if only_ebb:
        outfile = outfile + 'only_ebb_'
    else:
        outfile = outfile + 'both_ebb_and_flood_'
    outfile = outfile + 'over3years' + '.png'
    fig.savefig(outfile, format='png')

#########################################################################################################
# Son tay data
SSC = 'Concentration (g/m3)'
fichier = path + 'Data_2017.xlsx'
col_list_sontay = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']
# Download data of Son Tay
dfQ_ST_2017 = pd.read_excel(fichier, sheet_name='Q-Sontay2017', skiprows=3, usecols=col_list_sontay)
dfQ_ST_2017 = pd.melt(dfQ_ST_2017, var_name="Date", value_name="Value")
dfSPM_ST_2017 = pd.read_excel(fichier, sheet_name='SPM_sontay2017', skiprows=3, usecols=col_list_sontay)
dfSPM_ST_2017 = pd.melt(dfSPM_ST_2017, var_name="Date", value_name="Value")
start_date = datetime(2017, 1, 1)
end_date = start_date + timedelta(days=364)  # Assuming a non-leap year
current_date = start_date
datetime_series = []
while current_date <= end_date:
    datetime_series.append(current_date)
    current_date += timedelta(days=1)  # Increment by one day
dfSPM_ST_2017 = dfSPM_ST_2017.dropna().drop(['Date'], axis=1).assign(Date=datetime_series)
dfQ_ST_2017 = dfQ_ST_2017.dropna().drop(['Date'], axis=1).assign(Date=datetime_series)

df_ST_2017 = dfSPM_ST_2017.merge(dfQ_ST_2017, on='Date', how='inner')
df_ST_2017 = df_ST_2017.rename(columns={'Value_x': 'Concentration (g/m3)', 'Value_y': 'Q (m3/s)', 'Date': 'Datetime'})
df_ST_2017[[SSC, Q]] = df_ST_2017[[SSC, Q]].applymap(lambda x: np.nan if not isinstance(x, (float, int)) else x)

df_ST_all = pd.concat([df_ST_2017, df_ST])

#############################################################################################""
# First, I remove the outliers to derive the trend. 1 : selection of the outliers thanks to the z-score method :
# Calculate Z-scores for the 'y' column
z_scores = np.abs((df_ST_all[SSC] - df_ST_all[SSC].mean()) / df_ST_all[SSC].std())
# Set a threshold for outliers (e.g., Z-score > 2)
threshold = 2
# Filter out the outliers
filtered_df_ST_all = df_ST_all[z_scores <= threshold]  # Now 'filtered_df' contains the data without outliers

# Discharge vs concentration data at Son Tay
fig, ax = plt.subplots(figsize=(15, 10))  # , nrows=1, sharex=True)
fig.suptitle('Discharge vs Concentration at Son Tay')
ax.grid(True, alpha=0.5)
ax.set_xlabel('Discharge ($m^{3}/s$)', fontsize=fontsize)
ax.set_ylabel('Concentration (mg/L)', fontsize=fontsize)
ax.scatter(filtered_df_ST_all[Q], filtered_df_ST_all[SSC], color=sns.color_palette("colorblind")[2],
           label='Son Tay', s=3)
# ax.plot(filtered_Ebb_df[Q], predicted_values, label = str(np.round(slope,2))+'x +'+str(np.round(intercept,2)),
#        color=sns.color_palette("colorblind")[3])
# ax.scatter(filtered_Ebb_df[Q], slope*filtered_Ebb_df[Q]+intercept, s=3, color=sns.color_palette("colorblind")[3],
#           marker='x')
# ax.legend(fontsize=fontsize - 2)
if save:
    outfile = 'Discharge_VS_SPM_at_ST_over3years.png'
    fig.savefig(outfile, format='png')
##########################
fig, ax = plt.subplots(figsize=(18, 10))
fig.suptitle('Water levels ' + a + str(month) + '/' + str(year))
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level (cm)', fontsize=fontsize)
ax.set_ylim(-50, 450)
date_form = DateFormatter("%d/%m/%y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gcf().autofmt_xdate()
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize)
ax.plot(selected_data['Datetime'], selected_data['Water level Hon Dau'], color='black',
        label='Water level at Hon Dau')
label = 'Water level at Trung Trang\n lag = ' + \
        str(calculate_corr(melted_df, melted_df2, year_constraint, year, month_constraint, month)[0]) + 'h, corr=' + \
        str(np.round(calculate_corr(melted_df, melted_df2, year_constraint, year, month_constraint, month)[1], 4))
ax.plot(selected_data['Datetime'], selected_data['Water level Trung Trang'], color='grey', label=label)
plt.legend(fontsize=fontsize)
if save:
    fig.savefig('Waterlevel_' + str(month) + '_' + str(year))

average_day_Violaine, mean_day_Violaine, average_day_Juliette = [], [], []
nb_val_neg_Juliette, nb_val_neg_Violaine = [], []
survey_Violaine = [(26, 8), (27, 8), (28, 8), (3, 9), (4, 9), (5, 9), (6, 12), (7, 12), (8, 12), (12, 12), (13, 12),
                   (14, 12)]
survey_Juliette = [(16, 6), (17, 6), (18, 6), (10, 8), (11, 8), (12, 8), (13, 8), (2, 10), (3, 10), (4, 10), (5, 10)]
for d, m in (survey_Violaine):
    cond_Violaine = ((TT_2017['Datetime'].dt.month == m) & (TT_2017['Datetime'].dt.day == d))
    print(TT_2017[Q].loc[cond_Violaine])
    average_day_Violaine.append(np.average(TT_2017[Q].loc[cond_Violaine]))
    mean_day_Violaine.append(np.nanmean(TT_2017[Q].loc[cond_Violaine]))
    nb_val_neg_Violaine.append(TT_2017[Q].loc[cond_Violaine].loc[TT_2017[Q] < 0].count())
    # Selecting only the positive values : TT_2017.loc[cond_Violaine].loc[TT_2017[Q] > 0]
for d, m in survey_Juliette:
    cond_Juliette = ((merged_df['Datetime'].dt.year == 2022) & (merged_df['Datetime'].dt.month == m) & (
                merged_df['Datetime'].dt.day == d))
    average_day_Juliette.append(np.average(merged_df['Q'].loc[cond_Juliette]))
    nb_val_neg_Juliette.append(merged_df['Q'].loc[cond_Juliette].loc[merged_df['Q'] < 0].count())

sys.exit(1)
#########################    SON TAY 2018    ###########################
# 05/10 : code to reorganize a table to a one col file for 2018 Son Tay discharge;
file = '/home/penicaud/Documents/Data/Décharge_waterlevel/SonTay_2018.xlsx'
columns_to_load = list(range(13))
df = pd.read_excel(file, usecols=columns_to_load, skiprows=3)
df = df.rename(columns={'Ngày': 'Date'})
melted_df = pd.melt(df, id_vars=["Date"], var_name="Hour", value_name="Value")
melted_df.sort_values("Datetime", inplace=True)
melted_df.drop(['Date', 'Hour'], axis=1, inplace=True)

# 11/10 : 3 subplots one for ST, one for TT, one for HD for Water levels
fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
fig.suptitle('Water levels' + a + str(month) + '/' + str(year))

ax = axs[0]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level (m)', fontsize=fontsize)
ax.plot()

print('oj')
