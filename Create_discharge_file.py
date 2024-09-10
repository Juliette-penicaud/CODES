# 25/09/23 : create a file of daily mean values of the vanuc discharge from the discharge Trung Trang 206 file
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


rep = '/home/penicaud/Documents/Data/Décharge_waterlevel/'
file = rep + 'HonDau_water_level_2017.xlsx'

columns_to_load = list(range(25))
year = 2017
# Trung Trang file : hourly file
for month in range(1,13):
    df = pd.read_excel(file, sheet_name='Q_' + str(year), usecols=columns_to_load, skiprows=4+(month-1)*38, nrows = 34)
    print('Month loaded is : ', month)
    print(df[0:4])
    #df = df.rename(columns={'hour': 'Date'})
    for i in range(1,25):
        df = df.rename(columns={'Unnamed: '+str(i): str(i-1)})
    df_clean = df.dropna(how = 'any') # suppress every line where there are Nan
    df_clean = df_clean[df_clean['Date'] != 'Hour'] # Supress all the header of the table withe hour values
    melted_df = pd.melt(df_clean, id_vars=["Date"], var_name="Hour", value_name="Value")

    y = np.ones(len(melted_df['Date']), dtype=int) * year
    m = np.ones(len(melted_df['Date']), dtype = int) * month
    date = {'Year': y,
            'Month': m,  # Example months
            'Day': melted_df['Date'].values,  # Example days
            'Hour': melted_df['Hour'].values}  # Example hours
    df_date = pd.DataFrame(date)
    melted_df['Datetime'] = pd.to_datetime(df_date[['Year', 'Month', 'Day', 'Hour']])
    melted_df.sort_values("Datetime", inplace=True) # the df is in 1 column
    melted_df.drop(melted_df[['Date', 'Hour']], axis= 1)

    #daily_mean = melted_df.resample('D', on='Datetime').mean()
    if month == 1 :
        all_water_level = melted_df.copy()
        #all_daily_mean = daily_mean.copy()
    else :
        #all_daily_mean = pd.concat([all_daily_mean, daily_mean], axis = 0)
        all_water_level = pd.concat([all_water_level, melted_df], axis = 0)


out = rep + 'Water_level_HonDau_2017_temporal_serie.xlsx'
all_water_level[['Datetime', 'Value']].to_excel(out, index=False)
#all_daily_mean.to_csv(out)


# HonDau file, daily mean
file = '/home/penicaud/Documents/Data/Décharge_waterlevel/Décharges_rivières/Sontay_2016.xlsx'

columns_to_load = list(range(14))
year = 2016
df = pd.read_excel(file, usecols=columns_to_load, skiprows=3, nrows = 34)
print(df[0:4])
melted_df = pd.melt(df, id_vars=["Date"], var_name="Month", value_name="Value")
melted_df = melted_df.dropna(how = 'any')
melted_df.to_csv('Son_Tay_2016.txt')

print("OK")
########################    SON TAY 2016     #####################################
file = rep + 'TrungTrang_Q_SPM_2008-2016_en.xlsx'
year = 2016
# Trung Trang file : hourly file
for month in range(1,13):
    df = pd.read_excel(file, sheet_name='Q_' + str(year), usecols=columns_to_load, skiprows=3+(month-1)*38, nrows = 34)
    print('Month loaded is : ', month)
    print(df[0:4])
    df = df.rename(columns={'hour': 'Date'})
    df_clean = df.dropna(how = 'any') # suppress every line where there are Nan
    df_clean = df_clean[df_clean['Date'] != 'hour'] # Supress all the header of the table withe hour values
    melted_df = pd.melt(df_clean, id_vars=["Date"], var_name="Hour", value_name="Value")

    y = np.ones(len(melted_df['Date']), dtype=int) * year
    m = np.ones(len(melted_df['Date']), dtype = int) * month
    date = {'Year': y,
            'Month': m,  # Example months
            'Day': melted_df['Date'].values,  # Example days
            'Hour': melted_df['Hour'].values}  # Example hours
    df_date = pd.DataFrame(date)
    melted_df['Datetime'] = pd.to_datetime(df_date[['Year', 'Month', 'Day', 'Hour']])
    melted_df.sort_values("Datetime", inplace=True) # the df is in 1 column
    melted_df.drop(melted_df[['Date', 'Hour']], axis= 1)

    daily_mean = melted_df.resample('D', on='Datetime').mean()
    if month == 1 :
        all_daily_mean = daily_mean.copy()
    else :
        all_daily_mean = pd.concat([all_daily_mean, daily_mean], axis = 0)

all_daily_mean.to_csv(out)