# 11/03/24 : Je veux utiliser toutes les données de débit de toutes les années existantes.
# Difficulté, j'ai plusieurs types de formats de fichiers.

import pandas as pd
import matplotlib as mpl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
mpl.use('Agg')

fontsize=15
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['xtick.labelsize'] = fontsize - 4
plt.rcParams['ytick.labelsize'] = fontsize - 4
plt.rcParams['legend.fontsize'] = fontsize - 4
s = 25

rep = '/home/penicaud/Documents/Data/Décharge_waterlevel/'
file = rep + 'SonTay_alldata.xlsx'

create = False
if create :
    # Je créé le df qui contiendra toutes les données de débit journalier :
    df_all = pd.DataFrame(columns=['Date', 'Discharge'])

    ###############################################  Lecture de fichiers   ############################################
    # 1. Je commence par 2012 et 2013:
    list_col = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']
    #list_year = [2012, 2013]
    list_year = np.arange(1975,2014)
    nrows = 31
    skip = 3 # Correspond à l'en tête
    skip_new = 3 # 1364 si on commence en 2012
    skip2 = 2 # correspond au nombre de ligne entre les tableaux
    for year in list_year :
        df = pd.read_excel(file , sheet_name='Q 75_13', skiprows=skip_new, nrows=nrows, usecols=['Date']+list_col)
        df = df.rename(columns={'Date':'Day' }) # Je renomme les colonnes
        df = df.rename(columns={name: 'Month_'+str(i+1) for i,name in enumerate(list_col)}) # Y compris celles qui sont en mois en chiffre
        df['Year'] = year
        print(year, '\n', df[0:2], df[-3:])
        melted_df = pd.melt(df, id_vars=['Year', 'Day'], value_vars=[f'Month_{i+1}' for i in range(12)],
                            var_name='Month', value_name='Discharge')
        # Drop rows with NaN values (these correspond to days that do not exist in shorter months)
        melted_df = melted_df.dropna(subset=['Discharge'])
        # Sort the values by Year, Month, and Day to maintain chronological order
        melted_df['Month'] = melted_df['Month'].str.extract('(\d+)').astype(int)  # Extract month number
        melted_df = melted_df.sort_values(by=['Year', 'Month', 'Day']).reset_index(drop=True)
        melted_df['Date'] = pd.to_datetime(melted_df[['Year', 'Month', 'Day']])

        df_all = pd.concat([df_all, melted_df[['Date', 'Discharge']]], ignore_index=True)
        skip_new = skip_new + skip + nrows + skip2 + 1

    ########################
    # 2. Lecture de 2014
    # Je veux calculer les moyennes journalières à Son Tay de 2014 et 2015 qui sont à 3 mesures par jour
    # 1. Je lis le fichier 2014, qui est le plus chiant car pas en une seule colonne.
    write = False
    if write :
        file_bis = rep + 'Q_2014_Son_Tay.xlsx'
        df_2014 = pd.read_excel(file_bis, sheet_name='Q_2014_init', skiprows=2)
        df_2014 = df_2014.dropna(subset=['Day']).fillna(method='ffill')
        df_2014['Year'] = 2014
        df_2014['Date'] =  pd.to_datetime(df_2014[['Year', 'Month', 'Day']])
        writer = pd.ExcelWriter(file_bis, engine='openpyxl', mode='a', if_sheet_exists='overlay')
        df_2014.to_excel(writer, sheet_name='Q 2014', startrow=2, startcol=0, index=False)  # , header=header)
        writer.save()
    else :
        df_2014 = pd.read_excel(file, sheet_name='Q 2014', skiprows=2)
        df_all = pd.concat([df_all, df_2014[['Date', 'Discharge']]], ignore_index=True)

    ################################################
    # 3. 2015
    # Je lis maintenant le fichier 2015, qui lui est en une seule colonne
    df_2015 = pd.read_excel(file , sheet_name='Q 2015', skiprows=3, usecols=['Month', 'Day', 'mean Q(m3/s)'])
    df_2015 = df_2015.rename(columns={'mean Q(m3/s)':'Discharge'})
    df_2015 = df_2015.dropna(subset=['Day']).fillna(method='ffill')
    df_2015['Year'] = 2015
    df_2015['Date'] = pd.to_datetime(df_2015[['Year', 'Month', 'Day']])
    df_all = pd.concat([df_all, df_2015[['Date', 'Discharge']]], ignore_index=True)

    ########################################################################
    # 4. 2016-2020
    # #Je lis Q2016-2020
    skip = 3
    skip2 = 11
    skip_new = skip
    nrows = 31
    list_col = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']
    list_year = [2016, 2017, 2018, 2019, 2020]
    for year in list_year :
        df = pd.read_excel(file , sheet_name='Q 16_20', skiprows=skip_new, nrows=nrows)
        df = df.rename(columns={'Ngày':'Day' }) # Je renomme les colonnes
        df = df.rename(columns={name: 'Month_'+str(i+1) for i,name in enumerate(list_col)}) # Y compris celles qui sont en mois en chiffre
        df['Year'] = year
        melted_df = pd.melt(df, id_vars=['Year', 'Day'], value_vars=[f'Month_{i+1}' for i in range(12)],
                            var_name='Month', value_name='Discharge')
        # Drop rows with NaN values (these correspond to days that do not exist in shorter months)
        melted_df = melted_df.dropna(subset=['Discharge'])
        # Sort the values by Year, Month, and Day to maintain chronological order
        melted_df['Month'] = melted_df['Month'].str.extract('(\d+)').astype(int)  # Extract month number
        melted_df = melted_df.sort_values(by=['Year', 'Month', 'Day']).reset_index(drop=True)
        melted_df['Date'] = pd.to_datetime(melted_df[['Year', 'Month', 'Day']])

        df_all = pd.concat([df_all, melted_df[['Date', 'Discharge']]], ignore_index=True)
        skip_new = skip_new + nrows + skip2 + skip + 1

    #########################################################################################
    # 5. 2021-2022
    # Et enfin Q 2021-2022
    skip = 2
    list_col = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']

    for year in [2021,2022]:
        df = pd.read_excel(file , sheet_name='Q 21_22', skiprows=skip, nrows=nrows)
        df = df.rename(columns={'Unnamed: 0':'Year','Unnamed: 1': 'Day' }) # Je renomme les colonnes
        df = df.rename(columns={name: 'Month_'+str(i+1) for i,name in enumerate(list_col)}) # Y compris celles qui sont en mois en chiffre
        df['Year'] = year
        melted_df = pd.melt(df, id_vars=['Year', 'Day'], value_vars=[f'Month_{i+1}' for i in range(12)],
                            var_name='Month', value_name='Discharge')
        # Drop rows with NaN values (these correspond to days that do not exist in shorter months)
        melted_df = melted_df.dropna(subset=['Discharge'])
        # Sort the values by Year, Month, and Day to maintain chronological order
        melted_df['Month'] = melted_df['Month'].str.extract('(\d+)').astype(int)  # Extract month number
        melted_df = melted_df.sort_values(by=['Year', 'Month', 'Day']).reset_index(drop=True)
        melted_df['Date'] = pd.to_datetime(melted_df[['Year', 'Month', 'Day']])

        df_all = pd.concat([df_all, melted_df[['Date', 'Discharge']]], ignore_index=True)
        skip = 36

    print('ok')
    write_all = True
    if write_all :
        writer = pd.ExcelWriter(file, engine='openpyxl', mode='a', if_sheet_exists='replace')
        df_all.to_excel(writer, sheet_name='Q_1975-2022', startrow=2, startcol=0, index=False)  # , header=header)
        writer.save()

################################################
#################### SPM #######################
create_SPM = False
if create_SPM :
    # Je créé le df qui contiendra toutes les données de débit journalier :
    df_all = pd.DataFrame(columns=['Date', 'SPM'])

    ###############################################  Lecture de fichiers   ###########################################
    # ATTENTION : 1982 et 2020 sont full of nan (dans le fichier initial, 1982 est la copie de 1984, mais problème avec
    # années bissextiles...
    list_col = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']
    #list_year = [2012, 2013]
    list_year = np.arange(1975,2023)
    nrows = 31
    skip = 3 # Correspond à l'en tête
    skip_new = 3 # 1364 si on commence en 2012
    skip2 = 2 # correspond au nombre de ligne entre les tableaux
    for year in list_year :
        df = pd.read_excel(file , sheet_name='SPM 75_19', skiprows=skip_new, nrows=nrows, usecols=['Date']+list_col)
        df = df.rename(columns={'Date':'Day' }) # Je renomme les colonnes
        df = df.rename(columns={name: 'Month_'+str(i+1) for i,name in enumerate(list_col)}) # Y compris celles qui sont en mois en chiffre
        df['Year'] = year
        #print(year, '\n', df[0:2], df[-3:])
        melted_df = pd.melt(df, id_vars=['Year', 'Day'], value_vars=[f'Month_{i+1}' for i in range(12)],
                            var_name='Month', value_name='SPM')
        # Drop rows with NaN values (these correspond to days that do not exist in shorter months)
        melted_df = melted_df.dropna(subset=['SPM'])
        # Sort the values by Year, Month, and Day to maintain chronological order
        melted_df['Month'] = melted_df['Month'].str.extract('(\d+)').astype(int)  # Extract month number
        melted_df = melted_df.sort_values(by=['Year', 'Month', 'Day']).reset_index(drop=True)
        melted_df['Date'] = pd.to_datetime(melted_df[['Year', 'Month', 'Day']])
        print(melted_df['Date'][-3:])

        df_all = pd.concat([df_all, melted_df[['Date', 'SPM']]], ignore_index=True)
        skip_new = skip_new + skip + nrows + skip2 + 1

    print('ok')
    write_all = False
    if write_all :
        writer = pd.ExcelWriter(file, engine='openpyxl', mode='a', if_sheet_exists='replace')
        df_all.to_excel(writer, sheet_name='SPM_1975-2022', startrow=2, startcol=0, index=False)  # , header=header)
        writer.save()

df_all = pd.read_excel(file , sheet_name='Q_1975-2022', skiprows=2)
# Check de la taille de chaque année :
for year in np.arange(2012,2023):
    print(year , df_all[df_all['Date'].dt.year==year].shape[0])

# J'affiche le maximum de débit
print(df_all.loc[df_all['Discharge'].idxmax()])

# sub df avec toutes les valeurs de plus de 10 000 m3/s
print(df_all[df_all['Discharge'] > 10000])
df_2015_2022 = df_all[df_all['Date'].dt.year >=2015]
df_2015_2022['Year'] = df_2015_2022['Date'].dt.year
df_2015_2022['Month'] = df_2015_2022['Date'].dt.month
mean_discharge_per_year = df_2015_2022.groupby('Year')['Discharge'].mean().reset_index()
median_discharge_per_year = df_2015_2022.groupby('Year')['Discharge'].median().reset_index()

df_2022 = df_all[df_all['Date'].dt.year ==2022]
df_2022['Year'] = df_2022['Date'].dt.year
df_2022['Month'] = df_2022['Date'].dt.month


df_2002_2022= df_all[df_all['Date'].dt.year >=2002]
df_2002_2022_summer = df_2002_2022[(df_2002_2022['Date'].dt.month <= 10) & (df_2002_2022['Date'].dt.month >= 5)]
df_2002_2022_winter = df_2002_2022[(df_2002_2022['Date'].dt.month > 10) | (df_2002_2022['Date'].dt.month < 5)]

df_2002_2022['Year'] = df_2002_2022['Date'].dt.year
df_2002_2022['Month'] = df_2002_2022['Date'].dt.month

# Group by the 'Year' column and calculate the mean of the 'Discharge' column
mean_discharge_per_year = df_2002_2022.groupby('Year')['Discharge'].mean().reset_index()

# monthly mean discharge
mean_discharge_per_month = df_2002_2022.groupby('Month')['Discharge'].mean().reset_index()
month_names = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}
mean_discharge_per_month['Month'] = mean_discharge_per_month['Month'].map(month_names)
mean_discharge_per_month.set_index('Month', inplace=True)

van_uc_mean_discharge_per_month = 0.145 * mean_discharge_per_month

# 21/06/24 : Bar plot of the monthly discharge
# Plot the data as a bar plot
ax = mean_discharge_per_month.plot(kind='bar', legend=False, color='gray', figsize=(9, 6))
ax.bar(mean_discharge_per_month.index, van_uc_mean_discharge_per_month, color='blue', alpha=0.6, label='14.5% of monthly discharge')
plt.ylabel('Discharge (m$^3$/s)', fontsize=fontsize-2)
plt.title('Red River Climatological monthly mean discharge (2002-2022)', fontsize=fontsize-2)
plt.xticks(rotation=45)
plt.savefig('monthly_mean_discharge_with_vanuc.png')

#############
#############
#############
### 2000-2020:
var = 'SPM' # SPM ou Discharge

df_all = pd.read_excel(file , sheet_name=var+'_1975-2022', skiprows=2)
df_2000_2020= df_all[(df_all['Date'].dt.year >= 2000) & (df_all['Date'].dt.year <= 2020)]
df_2000_2020['Year'] = df_2000_2020['Date'].dt.year
df_2000_2020['Month'] = df_2000_2020['Date'].dt.month

df_2000_2020_summer = df_2000_2020[(df_2000_2020['Date'].dt.month <= 10) & (df_2000_2020['Date'].dt.month >= 5)]
df_2000_2020_winter = df_2000_2020[(df_2000_2020['Date'].dt.month > 10) | (df_2000_2020['Date'].dt.month < 5)]

# Group by the 'Year' column and calculate the mean of the 'Discharge' column
mean_per_year = df_2000_2020.groupby('Year')[var].mean().reset_index()

# monthly mean discharge
mean_per_month = df_2000_2020.groupby('Month')[var].mean().reset_index()
month_names = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}
mean_per_month['Month'] = mean_per_month['Month'].map(month_names)
mean_per_month.set_index('Month', inplace=True)

van_uc_mean_per_month = 0.144 * mean_per_month

# 21/06/24 : Bar plot of the monthly discharge
# Plot the data as a bar plot
fontsize = 15
if var == 'SPM':
    y_label = var + ' (g/m$^3$)'
    title = 'Red River Climatological monthly mean sediment discharge (2000-2020)'
    outfile = 'monthly_mean_SPM.png'
    color = 'brown'
elif var == 'Discharge':
    y_label = var + ' (m$^3$/s)'
    title = 'Red River Climatological monthly mean discharge (2000-2020)'
    outfile = 'monthly_mean_Discharge.png'
    color = 'darkslategrey'

ax = mean_per_month.plot(kind='bar', legend=False, color=color, figsize=(10, 6), zorder=5)
# ax.bar(mean_per_month.index, van_uc_mean_per_month, color='brown', alpha=0.6, label='14.5% of monthly discharge')
plt.ylabel(y_label, fontsize=fontsize - 2)
plt.title(title, fontsize=fontsize)
plt.xticks(rotation=45, fontsize=fontsize-2)
plt.xlabel('Month' , fontsize=fontsize-2)
ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', zorder=1)
plt.subplots_adjust(bottom=0.2)  # Adjust the bottom value as needed
plt.savefig(outfile)
