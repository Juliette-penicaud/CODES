# 13/08 : Analyse harmonique des données in situ HD

import pandas as pd
import matplotlib as mpl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import cmcrameri as cmc
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import cmocean
from scipy import stats
from scipy.stats import pearsonr
from matplotlib.ticker import MaxNLocator
import utide

mpl.use('Agg')

def find_local_minima(df, col, window_size, interp=False):  # possible que ca foire sur le 1e calcul
    # Si on a 2 min ou max consécutif : prendre la 2e valeur (pour repartir et trouver un autre min) et
    # ajouter 30 mn dans le time
    local_minima = []
    local_maxima = []
    time_local_min, time_local_max = [], []
    time_local_SW, local_SW = [], []
    i = 0
    len_max = 10
    if interp == True:
        len_max = 0.7*window_size
    while i < len(df[col]):
        window = df[col].loc[i:i + window_size].astype(float)
        if len(window) < len_max:  # To manage the last cycles
            print('len is too small')
            break
        local_max_idx = window.idxmax()
        # print('local max idx', local_max_idx)
        local_maxima.append(window[local_max_idx])
        # if df[col].loc[local_max_idx] == df[col].loc[local_max_idx + 1]:
        #     # print('2 consecutives MAX, we assume that the peak is +30mn')
        #     time_local_max.append(df['Datetime'].loc[local_max_idx] + timedelta(minutes=30))
        # elif df[col].loc[local_max_idx] == df[col].loc[local_max_idx + 1] == df[col].loc[local_max_idx + 2]:
        #     # print('3 consecutives MAX §§')
        #     time_local_min.append(df['Datetime'].loc[local_max_idx + 1])
        # elif df[col].loc[local_max_idx] == df[col].loc[local_max_idx + 1] == df[col].loc[local_max_idx + 2] \
        #         == df[col].loc[local_max_idx + 3]:
        #     # print('4 consecutives MAX §§')
        #     time_local_min.append(df['Datetime'].loc[local_max_idx + 1] + timedelta(minutes=30))
        # else:
        #     time_local_max.append(df['Datetime'].loc[local_max_idx])
        time_local_max.append(df['Datetime'].loc[local_max_idx])
        # 17/01 : I add the SW on both set of windows because we have 2 times more SW than HT or LT
        local_SW_idx = abs(window).idxmin()
        local_SW_idx_inf = local_SW_idx if local_SW_idx == 0 else local_SW_idx - 1
        local_SW_idx_sup = local_SW_idx if local_SW_idx == len(df[col])-1 else local_SW_idx + 1
        if df[col].loc[local_SW_idx_inf] * df[col].loc[local_SW_idx_sup] < 0:
            #18/05 : je rajoute une condition: il doit y avoir eu un changement de signe pour valider le SW
            local_SW.append(window[local_SW_idx])
            time_local_SW.append(df['Datetime'].loc[local_SW_idx])

        window2 = df[col].loc[local_max_idx:local_max_idx + window_size].astype(float)  # shift the window studied in order to find
        # another local min after the max
        local_min_idx = window2.idxmin()
        local_minima.append(window2[local_min_idx])
        # We do not take in account the values identical in this loop.
        # if df[col].loc[local_min_idx] == df[col].loc[local_min_idx + 1]:
        #     # print('2 consecutives MIN, we assume that the peak is +30mn')
        #     time_local_min.append(df['Datetime'].loc[local_min_idx] + timedelta(minutes=30))
        # elif df[col].loc[local_min_idx] == df[col].loc[local_min_idx + 1] == df[col].loc[local_min_idx + 2]:
        #     # print('3 consecutives MIN **')
        #     time_local_min.append(df['Datetime'].loc[local_min_idx + 1])
        # else:
        #     time_local_min.append(df['Datetime'].loc[local_min_idx])
        time_local_min.append(df['Datetime'].loc[local_min_idx])
        # 17/01 : I add the SW
        local_SW_idx = abs(window2).idxmin() # Attention, cela prend en compte le minimum de la valeur absolue,
        # et pas vraiment quand on est à 0, donc il est possible que des fois, la valeur n'est pas 0 car il n'y a
        # pas de passage à 0
        local_SW_idx_inf = local_SW_idx if local_SW_idx == 0 else local_SW_idx - 1
        local_SW_idx_sup = local_SW_idx if local_SW_idx == len(df[col])-1 else local_SW_idx + 1
        if df[col].loc[local_SW_idx_inf] * df[col].loc[local_SW_idx_sup] < 0:
            #18/05 : je rajoute une condition: il doit y avoir eu un changement de signe pour valider le SW
            local_SW.append(window2[local_SW_idx])
            time_local_SW.append(df['Datetime'].loc[local_SW_idx])

        i = local_min_idx + 1

    # 2d loop to check the min and max are well detected
    local_minima2 = []
    local_maxima2 = []
    time_local_min2, time_local_max2 = [], []
    for i in range(len(local_minima) - 1):
        min1 = df[col].loc[df['Datetime'] == time_local_min[i]].values[0]
        time_min1 = time_local_min[i]
        min1_2d = -9999
        time_min1_2d = -9999
        # print('min1 ', min1, time_min1)
        max1 = df[col].loc[df['Datetime'] == time_local_max[i]].values[0]
        time_max1 = time_local_max[i]
        max1_2d = -9999
        time_max1_2d = -9999
        for window_size in range(-5, 5):
            # Calculate time_val_min and time_val_max with the added window_size
            time_val_min = time_local_min[i] + timedelta(hours=window_size)
            time_val_max = time_local_max[i] + timedelta(hours=window_size)

            # Ensure that time_val_min and time_val_max are not before the earliest datetime in df
            start_time = df['Datetime'][0]
            time_val_min = max(time_val_min, start_time)
            time_val_max = max(time_val_max, start_time)

            test_min = df[col].loc[df['Datetime'] == time_val_min].values[0]
            test_max = df[col].loc[df['Datetime'] == time_val_max].values[0]
            if test_min < min1:
                min1 = test_min
                time_min1 = time_local_min[i] + timedelta(hours=window_size)
            elif test_min == min1 and window_size != 0:
                # print(" on a 2 extrema consécutif : que fait on ? ")
                min1_2d = test_min
                time_min1_2d = time_local_min[i] + timedelta(hours=window_size)
            if test_max > max1:
                max1 = test_max
                time_max1 = time_local_max[i] + timedelta(hours=window_size)
            elif test_max == max1 and window_size != 0:
                max1_2d = test_max
                time_max1_2d = time_local_max[i] + timedelta(hours=window_size)
        if min1 == min1_2d:
            time_min1 = time_min1 + (time_min1_2d - time_min1) / 2
        if max1 == max1_2d:
            time_max1 = time_max1 + (time_max1_2d - time_max1) / 2
        local_minima2.append(min1)
        time_local_min2.append(time_min1)
        local_maxima2.append(max1)
        time_local_max2.append(time_max1)

    return np.array(local_minima2), np.array(local_maxima2), np.array(time_local_min2), np.array(time_local_max2),\
           np.array(local_SW), np.array(time_local_SW)


fontsize=25
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['xtick.labelsize'] = fontsize - 4
plt.rcParams['ytick.labelsize'] = fontsize - 4
plt.rcParams['legend.fontsize'] = fontsize - 4
s = 25


rep = '/home/penicaud/Documents/Data/Décharge_waterlevel/'
file = rep + 'Data_HD_TT_2015-2022.xlsx'

columns_to_load = list(range(27))
df2 = pd.read_excel(file, sheet_name='HonDau_08-20', usecols=columns_to_load, skiprows=2)
df2 = df2.rename(columns={'Unnamed: '+str(i): str(i-3) for i in range(3,27)})
df2 = df2.rename(columns={'Nam':'Year', 'thang':'Month', 'Ngày':'Day'})
df2['Date'] = pd.to_datetime(df2[['Year', 'Month', 'Day']])
df2.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)

melted_df2 = pd.melt(df2, id_vars=["Date"], var_name="Hour", value_name="Value")
melted_df2['Datetime'] = pd.to_datetime(melted_df2['Date']) + pd.to_timedelta(melted_df2['Hour'].astype(int), unit='h')
melted_df2.sort_values("Datetime", inplace=True)
melted_df2 = melted_df2.rename(columns={'Value': 'Water level HD'})
melted_df2.drop(['Hour'], axis=1, inplace=True)
melted_df2['Water level HD'] = melted_df2['Water level HD']/100
df_all_HD = melted_df2.copy()
df_all_HD = df_all_HD.reset_index().drop(['index'], axis = 1)

columns_to_load = list(range(25))
list_month = np.arange(1, 13)
nrows = 31
skip = 4  # Correspond à l'en tête
list_skip2 = [2,1]  # correspond au nombre de ligne entre les tableaux

# Traitement spécifique de HD 2021 et 2022 qui sont dans des feuilles à part
i = 0
for y in [21, 22]:
    print('i', i)
    skip2 = list_skip2[i]  # correspond au nombre de ligne entre les tableaux
    print('skip2', skip2)
    skip_new = 4  # 1364 si on commence en 2012
    for month in list_month :
        print('month', month)
        df = pd.read_excel(file, sheet_name='HonDau'+str(y), skiprows=skip_new, nrows=nrows, usecols=columns_to_load)
        #print(df[0:2], df[-3:])
        df = df.rename(columns={'Ngày': 'Day'})  # Je renomme les colonnes
        df = df.rename(columns={'Unnamed: '+str(i): str(i-1) for i in range(0,25)})  # Y compris celles qui sont en mois en chiffre
        df['Year'] = 2000 + y
        df['Month'] = month
        #print(df[0:2], df[-3:])

        melted_df = pd.melt(df, id_vars=['Year', 'Month', 'Day'], value_vars=[f'{i}' for i in range(24)],
                            var_name='Hour', value_name='Water level HD')

        # Drop rows with NaN values (these correspond to days that do not exist in shorter months)
        melted_df = melted_df.dropna(subset=['Water level HD'])
        # Sort the values by Year, Month, and Day to maintain chronological order
        melted_df = melted_df.sort_values(by=['Year', 'Month', 'Day']).reset_index(drop=True)
        melted_df['Datetime'] = pd.to_datetime(melted_df[['Year', 'Month', 'Day', 'Hour']])
        melted_df['Date'] = pd.to_datetime(melted_df[['Year', 'Month', 'Day']])
        melted_df = melted_df[['Datetime', 'Date', 'Water level HD']]
        melted_df['Water level HD'] = melted_df['Water level HD']/100

        df_all_HD = pd.concat([df_all_HD, melted_df], ignore_index=True)
        skip_new = skip_new + skip + nrows + skip2 + 1
    i = i+1

print('File HD loaded')

# Sur le notebook exemple de Utide, ajout de l'anomalie
df_all_HD["anomaly"] = df_all_HD["Water level HD"] - df_all_HD["Water level HD"].mean()
df_all_HD["anomaly"] = df_all_HD["anomaly"].interpolate() # normalement je n'ai quasi pas de np.nan

coef = utide.solve(
    df_all_HD['Datetime'],
    df_all_HD["anomaly"],
    lat=20.66,
    method="ols", # ols ordinary least square ou wls weighted ls
    conf_int="MC", # Monte Carlo to calculate the confidence interval
    MC_n=0, # necessary to explicit the number of realisation
    verbose=True,
)

coef_mc = utide.solve(
    df_all_HD['Datetime'],
    df_all_HD["anomaly"],
    lat=20.66,
    method="ols", # ols ordinary least square ou wls weighted ls
    conf_int="MC", # Monte Carlo to calculate the confidence interval
    MC_n=1000, # necessary to explicit the number of realisation
    verbose=True,
)
#print(coef.keys())
# description of the name of all components
# aux : auxilaire informations (pas tout compris ce qu'il y a dedans)
# nR : number of realization of the monte carlo exp if conf_int="MC"
# nNR : number of non-redundant realizations or iterations
# nI : The number of iterations performed during the fitting process
# 'weights': Contains the weights used in the fitting process.
# A : Represents the amplitudes of the tidal constituents or other variables being fitted.
# 'g': Refers to the phases or arguments of the tidal constituents.
# 'mean': The mean value of the tidal heights or the mean value of the fitted tidal signal. It represents the average level of the tide.
# 'slope': Represents any linear trend or slope in the data or fitted model. This can be used to assess changes in tidal height over time.
# 'g_ci':The confidence intervals for the phase values ('g'). It provides a range within which the true phase values are likely to fall.
# 'A_ci' : same for amplitude
# diagn': Diagnostic information related to the fit. This might include goodness-of-fit statistics,
#       residuals, or other metrics used to assess the quality of the fit. Can contain R2
# 'PE': Potential energy or power estimates related to the tidal constituents.
# 'SNR':  Signal-to-noise ratio, which assesses the strength of the signal relative to noise. A higher SNR indicates a clearer signal.


# Reconstruit le signal de marée

tide = utide.reconstruct(df_all_HD['Datetime'], coef_mc, verbose=False)
print(tide.keys())

t = df_all_HD['Datetime']

###############################################"
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0,len(t)), df_all_HD['anomaly'] - tide.h)

# Figure : zoom sur une période plus précise :
fig, ax0 = plt.subplots(figsize=(15, 5))#, sharey=True, sharex=True)

ax0.plot(t, df_all_HD['anomaly'], label="Observations", color=CB_color_cycle[0]) # "C0"
ax0.tick_params(axis='both', which='major', labelsize=fontsize-5)
ax0.set_xlim(datetime(2010, 9, 5), datetime(2010, 9, 10))
ax0.set_xlabel('Time', fontsize=fontsize)
ax0.set_ylabel('Water level (m)', fontsize=fontsize)
fig.tight_layout()
fig.savefig('obs_water_level_2010.png', format = 'png')


# Figure de la reconstruction de tout le signal.
fig, (ax0, ax1, ax2) = plt.subplots(figsize=(17, 12), nrows=3, sharey=True, sharex=True)

ax0.plot(t, df_all_HD['anomaly'], label="Observations", color=CB_color_cycle[6]) # "C0"
ax0.tick_params(axis='both', which='major', labelsize=fontsize-5)

ax1.plot(t, tide.h, label="Prediction", color=CB_color_cycle[0])
ax1.tick_params(axis='both', which='major', labelsize=fontsize-5)
ax1.set_ylabel('Water level (m)', fontsize=fontsize)

ax2.plot(t, df_all_HD['anomaly'] - tide.h, label="Residual", color=CB_color_cycle[4])
ax2.tick_params(axis='both', which='major', labelsize=fontsize-5)

ax2.set_xlim(datetime(2007, 12, 1), datetime(2023, 1, 31))
ax2.set_xlabel('Time', fontsize=fontsize)

fig.legend(ncol=3, fontsize=fontsize-5) #loc="upper center"
fig.tight_layout()
fig.savefig('test_harmonic_analysis.png', format = 'png')


######################
# Je cherche à reconstruire seulement le signal lunar diurnal cycle
list_constituent = ['M2', 'S2', 'N2', "K2", "K1", "O1", "P1", "Mf"]
list_constituent = coef_mc['name'][:8] # ici ils sont rangés dans l'ordre de contribution des composantes à l'amplitude
tide_lunar = utide.reconstruct(df_all_HD['Datetime'], constit=list_constituent, coef=coef_mc)
tide_O1 = utide.reconstruct(df_all_HD['Datetime'], constit='O1', coef=coef_mc)
tide_K1 = utide.reconstruct(df_all_HD['Datetime'], constit='K1', coef=coef_mc)

print(tide_lunar.keys())

# Plot component O1 K1 and the height principal components
fig, (ax0, ax1, ax2) = plt.subplots(figsize=(17, 15), nrows=3, sharex=True) # sharey=True,

ax0.set_title(" ".join(list_constituent), fontsize=fontsize)
#ax0.yaxis.set_major_locator(MultipleLocator(0.5))
#ax0.yaxis.set_minor_locator(MultipleLocator(0.25))
ax0.grid(True, which='both')
ax0.tick_params(axis='both', which='major', labelsize=fontsize-5, pad = 15)
ax0.plot(tide_lunar['t_in'], tide_lunar['h'], label='principal component amplitude', color='royalblue', lw=0.5, alpha=0.7)
#ax0.set_ylim(-2,2)

ax1.set_title('O1', fontsize=fontsize)
ax1.yaxis.set_major_locator(MultipleLocator(0.5))
ax1.yaxis.set_minor_locator(MultipleLocator(0.25))
ax1.grid(True, which='both')
ax1.tick_params(axis='both', which='major', labelsize=fontsize-5, pad = 15)
ax1.plot(tide_O1['t_in'], tide_O1['h'], label='O1', color='royalblue', lw=0.5, alpha=0.5)
ax1.set_ylabel('Tidal amplitude (m)', fontsize=fontsize)
ax1.set_ylim(-1,1)

ax2.set_title('K1', fontsize=fontsize)
ax2.yaxis.set_major_locator(MultipleLocator(0.5))
ax2.yaxis.set_minor_locator(MultipleLocator(0.25))
ax2.grid(True, which='both')
ax2.tick_params(axis='both', which='major', labelsize=fontsize-5, pad = 15)
ax2.plot(tide_K1['t_in'], tide_K1['h'], label='K1', color='royalblue', lw=0.5, alpha=0.5)
ax2.set_xlabel('Time', fontsize=fontsize)
ax2.set_xlim(datetime(2007, 12, 1), datetime(2023, 1, 31))
ax2.set_ylim(-1,1)

fig.tight_layout()
fig.savefig('phase_and_amplitude_lunar_nodal_cycle_3subplots.png', format='png')

##################
##################
list_constituent = ['M2', 'S2', 'N2', "K2", "K1", "O1", "P1", "Mf"]

for onde in list_constituent :
    tide_lunar = utide.reconstruct(df_all_HD['Datetime'], constit=onde, coef=coef_mc)
    #print(tide_lunar.keys())

    # Plot the N2 component
    plt.figure(figsize=(12, 6))

    # Plot Amplitude of N2
    plt.subplot(2, 1, 1)
    plt.plot(tide_lunar['t_in'], tide_lunar['h'], label='Amplitude '+onde, color='blue', lw=0.5)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Tidal Component Amplitude')
    plt.legend()

    plt.subplot(2, 1, 1)
    plt.plot(tide_lunar['t_in'], tide_lunar['h'], label='Amplitude '+onde, color='blue', lw=0.5)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Tidal Component Amplitude')
    plt.legend()

    plt.tight_layout()
    outfile = 'phase_and_amplitude_lunar_nodal_cycle_'
    outfile = outfile + onde + '.png'
    plt.savefig(outfile, format='png')


#############################
#############################

# Figure all year spring tides at HD
figure_tidal_range_HD = True
window_size = 17
if figure_tidal_range_HD:
    df_harmonic_tide = pd.DataFrame()
    df_harmonic_tide['Datetime'] = tide['t_in']
    df_harmonic_tide['Water level HD'] = tide['h']

    s = 50
    # 1. Je fais un tableau avec seulement Ebb and flood HD :
    local_minima_HD, local_maxima_HD, time_local_min_HD, time_local_max_HD, a, b = \
        find_local_minima(df_harmonic_tide, 'Water level HD', window_size)

    fontsize = 35
    save = True
    zoom = False
    figure1 = True
    list_year = np.arange(2008,2010)
    if figure1 :
        for year in list_year :
            fig, ax = plt.subplots(figsize=(20, 12))
            fig.suptitle(year, fontsize=fontsize)

            ax.set_title('Hon Dau', fontsize=fontsize - 5)
            ax.grid('both', alpha=0.5)
            ax.plot(df_harmonic_tide['Datetime'], df_harmonic_tide['Water level HD'], label='hourly water level',
                    color='grey', zorder=0.1)
            ax.scatter(time_local_min_HD, local_minima_HD, label='min height values', marker='o', color='black', zorder=1)
            ax.scatter(time_local_max_HD, local_maxima_HD, label='max height values', marker='o', color='red', zorder=1)
            ax.set_ylim(-3, 3)
            ax.tick_params(axis='both', which='major', labelsize=fontsize - 5, pad=10)
            ax.set_ylabel('Water level (m)', fontsize=fontsize - 5)
            ax.set_xlabel('Time', fontsize=fontsize - 5)

            date_form = DateFormatter("%d/%m")  # Define the date format
            if zoom:
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
                plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=5))
                ax.set_xlim(datetime(year, 5, 1), datetime(year, 8, 1))
            else:
                plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
                ax.set_xlim(datetime(year, 1, 1), datetime(year + 1, 1, 1))

            ax.yaxis.set_major_locator(MultipleLocator(2))  # Set minor tick every 0.2 units on the x-axis
            ax.yaxis.set_minor_locator(MultipleLocator(1))

            ax.xaxis.set_major_formatter(date_form)
            if save:
                outfile = 'min_max_data_tidal_harmonic_'
                if zoom:
                    outfile = outfile + 'zoom_'
                outfile = outfile + str(year) + '.png'
                fig.savefig(outfile, format='png')

    Ebb_HD = pd.DataFrame(
        time_local_max_HD)  # the starting datetime is the beginning of the Ebb_HD i.e : max water levels at TT
    Flood_HD = pd.DataFrame(time_local_min_HD)
    Ebb_HD = Ebb_HD.rename(columns={0: 'Datetime ebb HD'})
    Ebb_HD['Datetime'] = Ebb_HD['Datetime ebb HD'].copy()
    Flood_HD = Flood_HD.rename(columns={0: 'Datetime flood HD'})
    Flood_HD['Datetime'] = Flood_HD['Datetime flood HD'].copy()
    if time_local_max_HD[0] > time_local_min_HD[0]:  # To know which one we need to substract
        print('The first extremum is the minimum data, so it is the flood_HD')
        Flood_HD['Duration HD'] = time_local_max_HD - time_local_min_HD
        Flood_HD['Amplitude HD'] = local_maxima_HD - local_minima_HD
        Ebb_HD['Duration HD'] = np.roll(time_local_min_HD, shift=-1) - time_local_max_HD
        Ebb_HD['Amplitude HD'] = np.roll(local_minima_HD, shift=-1) - local_maxima_HD
        Ebb_HD.loc[len(Ebb_HD) - 1, 'Duration HD'] = np.nan
        Ebb_HD.loc[len(Ebb_HD) - 1, 'Amplitude HD'] = np.nan
    else:
        print('The first extremum is the MAX data, so it is the Ebb_HD')
        Flood_HD['Duration HD'] = np.roll(time_local_max_HD, shift=-1) - time_local_min_HD
        Flood_HD['Amplitude HD'] = np.roll(local_maxima_HD, shift=-1) - local_minima_HD
        Flood_HD.loc[len(Flood_HD) - 1, 'Duration HD'] = np.nan
        Flood_HD.loc[len(Flood_HD) - 1, 'Amplitude HD'] = np.nan
        Ebb_HD['Duration HD'] = time_local_min_HD - time_local_max_HD
        Ebb_HD['Amplitude HD'] = local_minima_HD - local_maxima_HD

    Ebb_HD_abs = Ebb_HD.copy()
    Ebb_HD_abs['Amplitude HD'] = Ebb_HD_abs['Amplitude HD'].abs()
    Ebb_and_flood_HD = pd.concat([Ebb_HD_abs, Flood_HD], axis=0)

    # Calcul des spring tides per year : quantile 0.95, je groupe par year, et je vois l'amplitude à HD.
    val_quantile = 0.90
    spring_tides_per_year = Ebb_and_flood_HD.groupby(Ebb_and_flood_HD['Datetime'].dt.year)['Amplitude HD'].quantile(
        [val_quantile])
    amplitude1 = spring_tides_per_year.max() - spring_tides_per_year.min()

    val_quantile2 = 0.95
    spring_tides_per_year2 = Ebb_and_flood_HD.groupby(Ebb_and_flood_HD['Datetime'].dt.year)['Amplitude HD'].quantile(
        [val_quantile2])
    amplitude2 = spring_tides_per_year2.max() - spring_tides_per_year2.min()

    val_quantile3 = 0.99
    spring_tides_per_year3 = Ebb_and_flood_HD.groupby(Ebb_and_flood_HD['Datetime'].dt.year)['Amplitude HD'].quantile(
        [val_quantile3])
    amplitude3 = spring_tides_per_year3.max() - spring_tides_per_year3.min()

    x = spring_tides_per_year.index.get_level_values(0)
    y = spring_tides_per_year.values
    x2 = spring_tides_per_year2.index.get_level_values(0)
    y2 = spring_tides_per_year2.values
    x3 = spring_tides_per_year3.index.get_level_values(0)
    y3 = spring_tides_per_year3.values

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.grid(which='both', alpha=0.5)
    ax.set_xlabel('Year', fontsize=fontsize, labelpad=20)
    ax.set_ylabel('Tidal amplitude at HD (m)', fontsize=fontsize, labelpad=20)  #
    ax.plot(x, y, marker='o', c='royalblue',
            label='amplitude = ' + str(np.round(amplitude1, 2)) + ' m') # 10% highest spring tides,
    ax.plot(x2, y2, marker='D', c='orangered',
            label='amplitude = ' + str(np.round(amplitude2, 2)) + ' m') # 5% highest spring tides,
    ax.plot(x3, y3, marker='v', c='olivedrab',
            label='amplitude = ' + str(np.round(amplitude3, 2)) + ' m') # 1% highest spring tides,
    ax.set_ylim(2.5, 4)

    legend = ax.legend()
    for label in legend.get_texts():
        label.set_fontsize(fontsize - 10)  # Set the desired font size

    ax.tick_params(axis='both', which='major', labelsize=fontsize - 5)
    plt.tight_layout()
    ax.set_ylabel('', fontsize=fontsize, labelpad=20)  # Trick pour garde quand meme la meme taille qu'avec les données
    outfile = 'Spring_tide_per_year_HD_from_harmonic_' + str(val_quantile) + '_and_' + str(val_quantile2) + '_' + str(
        val_quantile3) + '.png'
    fig.savefig(outfile, format='png')
