# 22/05/23 : Je créée les plots pour visualiser les données du file recap all data
# mai : plot du paramètre de Simpson fonction distance à embouchure et tentative fonction du percentage de marée
# 2/06 : plot de la variation en temps de : niveau eau, vitesse (surface? ou colonne ?) et arrivée sal
# 11/08/23 : Je change de version et cherche à ouvrir tout ca avec xarray
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
mpl.use('Agg')

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def set_newticks(twin_axes):
    new_tick = []
    for i in range(len(twin_axes.get_xticks())):
        tick = twin_axes.get_xticks()[i]
        if tick <= 100:
            print('do not change')
            new_tick.append(tick)
        elif 100 < tick <= 200:
            print('Ebb tide :')
            new_tick.append(-(200 - tick))
        elif tick > 200:
            print('New flood tide')
            new_tick.append(tick - 200)
        else:
            print('PROBLEM IN FUNCTION NEW TICKS')
    return new_tick


def velocity_and_direction(type_fixe, month, data_fixe, name_sta, unit):
    # Several cases : # 16-08-23 introduction of case1 or case2 to discriminate the subcases i.e C and D
    # where we are sure to be in + or - case
    # CASE1:
    # Case A : nor and east are + : direction NE, sens value depends on the value of the angle
    # Case B : nor and east are - : direction SO, sens value depends on the value of the angle
    # Case C : nor + and east -, i.e angle value -, direction NO, sens is -
    # Case D : nor - and east +, i.e angle value -, direction SE, sens is +
    # CASE2 :
    # Case A : nor + east - : direction NE, sens value depends on the value of the angle
    # Case B : nor - east + : direction SO, sens value depends on the value of the angle
    # Case C : nor and east -, i.e angle value -, direction NO, sens is -
    # Case D : nor and east +, i.e angle value -, direction SE, sens is +
    # dict_angle = {'section1': {'angle': 50, 'case': 'case1'},
    #               'section2': {'angle': 0, 'case': 'case1'},
    #               'section3': {'angle': 15, 'case': 'case2'},
    #               'section4': {'angle': 25, 'case': 'case1'},
    #               'section5': {'angle': 77, 'case': 'case1'},
    #               'section6': {'angle': 50, 'case': 'case1'},
    #               'section7': {'angle': 10, 'case': 'case1'},
    #               'section8': {'angle': 65, 'case': 'case2'},
    #               'section9': {'angle': 28, 'case': 'case1'},
    #               'section10': {'angle': 80, 'case': 'case1'},
    #               'section11': {'angle': 33, 'case': 'case1'},
    #               'section12': {'angle': 44, 'case': 'case1'}}
    dict_angle = {'section2': {'angle': 55, 'case': 'case1'},
                  'section3': {'angle': 27, 'case': 'case2'}, # 18
                  'section4': {'angle': 80, 'case': 'case1'},
                  'section5': {'angle': 35, 'case': 'case1'},
                  'section6': {'angle': 20, 'case': 'case1'},
                  'section7': {'angle': 52, 'case': 'case1'},
                  'section8': {'angle': 25, 'case': 'case2'},
                  'section9': {'angle': 82, 'case': 'case1'},
                  'section10': {'angle': 25, 'case': 'case1'},
                  'section11': {'angle': 45, 'case': 'case1'},
                  'section12': {'angle': 75, 'case': 'case1'},
                  'section13': {'angle': 0, 'case': 'case1'},
                  "section14": {'angle': 35, 'case': 'case1'} }

    dict_section = {'section2': ['A7', 'O24', 'S17', 'O10', 'O31', 'O35', 'A14'],
                    'section3': ['O34', 'A15', 'S1', 'S32', 'S36', 'S34', 'S35', 'S33', 'O32', 'O11', 'O23', 'A16',
                                 'S37', 'O12', 'A8', 'A17', 'S38', 'S31'],
                    'section4': ['A9', 'O33'],
                    'section5': [''],
                    'section6': ['O22', 'O13', 'O22-2', 'A19', 'O14', "A11", 'A12', 'S39', 'A10', 'A18'],
                    'section7': ['SF1', 'A20'],
                    'section8': ['A21', 'O15', 'O16', 'A22', 'O17', 'A23', 'OF1'],
                    'section9': ['O18', 'A24'],
                    'section10': ["O19", 'A25', 'O20'],
                    'section11': ['A26', 'O21', 'A27'],
                    "section12": ["O29", 'S9', 'S23', "S10", "S22"], # S24, O6, O7
                    "section13": ["A33", "A34", "S18", "S19", "S20", "S21"],
                    'section14': ['']  # if not in other section, then it is section12}
                    }
    if type_fixe == 'fixe':
        case = 'case1'
        angle_seuil = 50  # angles for the station fixe values
    elif type_fixe == 'small fixe' and month == 'Octobre':
        case = 'case2'
        angle_seuil = 65
    elif type_fixe == 'small fixe' and month == 'August':
        angle_seuil = 10
        case = 'case1'
    else:
        s = 2
        while s < 14:
            if name_sta in dict_section['section' + str(s)]:
                break
            else:
                s = s + 1
        section = 'section' + str(s)
        angle_seuil = dict_angle[section]['angle']
        case = dict_angle[section]['case']

    vnor = 'vitesse north'
    veast = 'vitesse east'
    if case == 'case1':
        condA = (data_fixe[vnor] > 0) & (data_fixe[veast] > 0)
        condB = (data_fixe[vnor] < 0) & (data_fixe[veast] < 0)
        condC = (data_fixe[vnor] > 0) & (data_fixe[veast] < 0)
        condD = (data_fixe[vnor] < 0) & (data_fixe[veast] > 0)
        cond1 = (data_fixe['angle'].abs() > angle_seuil)
        cond2 = (data_fixe['angle'].abs() <= angle_seuil)
    elif case == 'case2':
        condD = (data_fixe[vnor] > 0) & (data_fixe[veast] > 0)
        condC = (data_fixe[vnor] < 0) & (data_fixe[veast] < 0)
        condA = (data_fixe[vnor] > 0) & (data_fixe[veast] < 0)
        condB = (data_fixe[vnor] < 0) & (data_fixe[veast] > 0)
        cond2 = (data_fixe['angle'].abs() > angle_seuil)
        cond1 = (data_fixe['angle'].abs() <= angle_seuil)
    new_v = 'Velocity sens'
    data_fixe[new_v] = np.nan  # data_fixe['Vel_'+v].values.copy()

    # Vel = -
    data_fixe.loc[condA & cond1, new_v] = -data_fixe.loc[condA & cond1, 'module vitesse u']
    data_fixe.loc[condB & cond2, new_v] = -data_fixe.loc[condB & cond2, 'module vitesse u']
    data_fixe.loc[condC, new_v] = -data_fixe.loc[condC, 'module vitesse u']
    # Vel +
    data_fixe.loc[condA & cond2, new_v] = data_fixe.loc[condA & cond2, 'module vitesse u']
    data_fixe.loc[condB & cond1, new_v] = data_fixe.loc[condB & cond1, 'module vitesse u']
    data_fixe.loc[condD, new_v] = data_fixe.loc[condD, 'module vitesse u']
    if unit=='m/s':
        data_fixe[new_v] = data_fixe[new_v]/1000

# VARIABLE AND PARAMETER
year = '2022'
list_month = ['June', 'August', 'Octobre']
i = 1  # 0 1 2 a voir pour faire une boucle si besoin

# Cmap business
fontsize = 10
cmap_D50 = cmc.cm.batlowW
cmap_turb = plt.cm.magma_r
cmap_sal = sns.color_palette("mako_r", as_cmap=True)
cmap_vel = cmc.cm.vik
cmap_Junge = cmc.cm.grayC
cmap_ws = cmc.cm.nuuk
# sns.color_palette("crest", as_cmap=True)
# sns.light_palette("seagreen", as_cmap=True)
# sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
# sns.color_palette("vlag", as_cmap=True)
# cmap_vel = sns.color_palette("icefire", as_cmap=True)
# sns.dark_palette("#69d", reverse=True, as_cmap=True)
# plt.cm.Spectral_r
# I tried : pink_r with truncature, gis_earth, terrain,OrangeBlue, RdBu, hsv, Spectral,
# new_cmap = truncate_colormap(cmap, 0, 1)
# bottom = plt.cm.get_cmap('Oranges', 128)
# top = plt.cm.get_cmap('Blues', 128)
# newcolors = np.vstack((top(np.linspace(0, 1, 128)),
#                        bottom(np.linspace(0, 1, 128))))
# new_cmap = mpl.colors.ListedColormap(newcolors, name='OrangeBlue')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 14/08/23 : I do a list of the first and last values of the transects. Are all sheets ranked ?
dict_limtransect = {'June': {'T1': {'first': 'S1', 'last': 'S10', 'ylim': -15},
                             'T2': {'first': 'S11', 'last': 'S17', 'ylim': -15},
                             'T3': {'first': 'S18', 'last': 'S28', 'ylim': -10},
                             'T4': {'first': 'S29', 'last': 'S39', 'ylim': -15},
                             'fixe': {'first': 'SF1', 'last': 'SF37', 'ylim': -10}},
                    'August': {'T1': {'first': 'A1', 'last': 'A5', 'ylim': -12},
                               'T2': {'first': 'A6', 'last': 'SA12', 'ylim': -20},
                               'T3': {'first': 'A13', 'last': 'A27', 'ylim': -20},
                               'T4': {'first': 'A28', 'last': 'A34', 'ylim': -10},
                               'small fixe': {'first': 'AF1.1', 'last': 'AF1.22', 'ylim': -15},
                               'fixe': {'first': 'AF1', 'last': 'AF38', 'ylim': -10}},
                    'Octobre': {'T1': {'first': 'O1', 'last': 'O7', 'ylim': -10},
                                'T2': {'first': 'O8', 'last': 'O21', 'ylim': -15},
                                'T3': {'first': 'O22', 'last': 'O29', 'ylim': -20},
                                'T4': {'first': 'O30', 'last': 'O35', 'ylim': -15},
                                'fixe': {'first': 'OF1', 'last': 'OF52', 'ylim': -12},
                                'small fixe': {'first': 'OF1.1', 'last': 'OF1.12', 'ylim': -15}}
                    }

variables = ['Sal_vel', 'Turbidity_D50', 'Junge_ws']
dict_var = {'Turbidity_D50': {'var1': 'Turbidity filtered 5', 'disp1': 'Turbidity (FTU)', 'vmin1': 0,
                              'vmax1': 250, 'cmap1': cmap_turb,
                              'var2': 'D50', 'disp2': 'D50 (µm)', 'vmin2': 0, 'vmax2': 200, 'cmap2': cmap_D50},
            'Junge_ws': {'var1': 'Junge', 'disp1': 'Junge parameter', 'vmin1': 2,
                         'vmax1': 4, 'cmap1': cmap_Junge,
                         'var2': 'ws', 'disp2': 'Ws (cm/s)', 'vmin2': 0, 'vmax2': 0.5, 'cmap2': cmap_ws},
            'Sal_vel': {'var1': 'Salinity', 'disp1': 'Salinity (PSU)', 'vmin1': 0,
                        'vmax1': 30, 'cmap1': cmap_sal,
                        'var2': 'Velocity sens', 'disp2': 'Velocity (m/s)', 'vmin2': -1.200, 'vmax2': 1.200,
                        'cmap2': cmap_vel},  # 'module vitesse u'
            'all1': {'var1': 'Salinity', 'disp1': 'Salinity (PSU)', 'vmin1': 0,
                     'vmax1': 30, 'cmap1': cmap_sal,
                     'var2': 'Velocity sens', 'disp2': 'Velocity (m/s)', 'vmin2': -1.200, 'vmax2': 1.200,
                     'cmap2': cmap_vel,
                     'var3': 'Turbidity filtered 5', 'disp3': 'Turbidity (FTU)', 'vmin3': 0,
                     'vmax3': 250, 'cmap3': cmap_turb,
                     'var4': 'D50 filtered', 'disp4': 'D50 (µm)', 'vmin4': 0, 'vmax4': 200, 'cmap4': cmap_D50,
                     'var5': 'Junge filtered', 'disp5': 'Junge parameter', 'vmin5': 2,
                     'vmax5': 4, 'cmap5': cmap_Junge,
                     'var6': 'ws filtered', 'disp6': 'Ws (cm/s)', 'vmin6': 0, 'vmax6': 0.5, 'cmap6': cmap_ws},
            'all': {'var1': 'Salinity', 'disp1': 'Salinity (PSU)', 'vmin1': 0,
                    'vmax1': 30, 'cmap1': cmap_sal,
                    'var2': 'Velocity sens', 'disp2': 'Velocity (m/s)', 'vmin2': -1.200, 'vmax2': 1.200,
                    'cmap2': cmap_vel,
                    'var3': 'N filtered', 'disp3': 'N (s$^{-1}$)', 'vmin3': 0,
                    'vmax3': 1, 'cmap3': 'Spectral',
                    'var4': 'Turbidity filtered 5', 'disp4': 'Turbidity (FTU)', 'vmin4': 0,
                    'vmax4': 250, 'cmap4': cmap_turb,
                    'var5': 'D50 filtered', 'disp5': 'D50 (µm)', 'vmin5': 0, 'vmax5': 200, 'cmap5': cmap_D50,
                    'var6': 'Junge filtered', 'disp6': 'Junge parameter', 'vmin6': 2,
                    'vmax6': 4, 'cmap6': cmap_Junge,
                    'var7': 'ws filtered', 'disp7': 'Ws (mm/s)', 'vmin7': 0, 'vmax7': 0.5, 'cmap7': cmap_ws}
            }


all_month_together = 0 # Allow to do a loop on the 3 surveys.
if all_month_together :
    length_month = 3
    dict_marker={'June' : 'o', 'August': '<', 'Octobre':'x'}
    out = "Ws_vs_D50_allmonthtogether.png"
    fig_all, ax_all = plt.subplots(figsize=(10, 10), constrained_layout=True)
    fig_all.suptitle('All months', fontsize=fontsize)
    ax_all.grid(True, alpha=0.5)
    ax_all.set_ylabel('Ws (mm/s)', fontsize=fontsize)
    ax_all.set_xlabel('D50 (µm)', fontsize=fontsize)
    ax_all.set_xlim(0, 160)
    ax_all.set_ylim(0, 0.5)
    ax_all.tick_params(axis='both', which='both', labelsize=fontsize, width=0.5)
    for spine in ax_all.spines.values():
        spine.set_linewidth(0.5)  # Adjust the linewidth as needed
else :
    length_month = 3 # VARIABLE !!
for i in range(length_month): #range(3):
    month = list_month[i]

    rep = '/home/penicaud/Documents/Data/Survey_' + month
    file = rep + '/Recap_all_param_' + month + '.xlsx'
    dict_month = {'June': {'nrows': 87},
                  'August': {'nrows': 95},
                  'Octobre': {'nrows': 111}}

    file_station = rep + '/Stations_' + month + '.xlsx'
    f = pd.ExcelFile(file)
    df_global = pd.read_excel(file_station, sheet_name=0, nrows=dict_month[month]['nrows'])  # read the stations name
    df_global = df_global.dropna(subset=['Stations'])
    list_sheet = df_global['Stations'].values
    list_sheet = [col for col in list_sheet if not '-' in col]  # remove the 'bistations' to have a unique liste
    # corresponding to the devices stations
    # EXCLUDE the SXX-25
    # df_global = pd.ExcelFile(file)  # df du LISST
    # list_sheet = df_global.sheet_names
    # WARNING : Commented method is NOT A GOOD ONE BECAUSE BASED ON LISST STATIONS, so missing stations
    dict_global = pd.read_excel(file, sheet_name=list_sheet)

    transect = ['fixe', 'T1', 'T2', 'T3', 'T4'] # WARNING : ['all'] if all, all together
    #if month == 'Octobre' or month == 'August':
    #    transect.append('small fixe')

    for t in transect:  # TODO : improve with dict_limtransect wich can set the limits
        if (all_month_together == False) | (all_month_together and month=='June'):
            list_dfs, list_name = [], []
        if t == 'all':
            for sheet in f.sheet_names:
                df = f.parse(sheet)
                print('sheet', sheet, df['Transect'][0])
                list_dfs.append(df)
                list_name.append(sheet)
        else :
            for sheet in f.sheet_names:
                df = f.parse(sheet)
                #print('sheet', sheet, df['Transect'][0])
                if df['Transect'][0] == t:
                    list_dfs.append(df)
                    list_name.append(sheet)

        condition = (t == 'fixe' or t == 'small fixe')
        list_tide = []
        if condition:
            x = 'Time'
        else:
            x = 'Distance'

        # 8/02/24 J'ajoute un traitement sur les paramètres de sédim, pour voir oter les données si N<seuil
        seuil_N = 0.035
        for df in list_dfs:
            df.loc[df['Junge'] == 1, 'Junge'] = np.nan # A cause de l'écriture des fichiers (Create_recap_data_file),
            # si SPMVC = 0, Junge = 1. PROBLEME RESOLU VIA CETTE MAGOUILLE
            df['N'] = df['N2'].apply(lambda x: 0 if x < 0 else np.sqrt(x))
            df['N filtered'] = df['N'].copy()
            df.loc[df['N filtered'] < seuil_N, 'N filtered'] = 0
            for col in ['Junge', 'D50', 'ws'] :
                df[col+' filtered'] = df[col].copy()
                df.loc[df['N filtered'] == 0, col+' filtered'] = 0

        v = 'all'
        if v == 'all':
            nrows = 7
        elif v == 'all1':
            nrows = 6
        else :
            nrows = 2
        unit='m/s'

        s = 15
        ylim = dict_limtransect[month][t]['ylim']
        fig, axs = plt.subplots(nrows=nrows, sharex=True, sharey=True, figsize = (7,13), constrained_layout=True)# (3,15))
        fig.suptitle(t + ' ' + str(list_dfs[0]['Time'][0].day) + ' ' + month, fontsize=fontsize)
        list_p = []
        for nrow in range(1,1+nrows) :
            list_p.append('p'+str(nrow))
            ax = axs[nrow-1]
            ax.grid(True, alpha=0.5)
            ax.set_ylabel('Depth (m)', fontsize=fontsize)
            ax.set_ylim(ylim, 0)
            ax.tick_params(axis='both', which='both', labelsize=fontsize, width=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)  # Adjust the linewidth as needed

        for i in range(len(list_dfs)):
            if condition:
                list_tide.append(list_dfs[i]['Percentage of tide'][0])
            for nrow in range(1,1+nrows):
                ax = axs[nrow-1]
                velocity_and_direction(t, month, list_dfs[i], list_name[i], unit)
                # TODO : si tout fonctionne,l'ajouter au fichier recap pour ne pas le recalculer à chq fois
                list_p[nrow-1] = ax.scatter(list_dfs[i][x], -list_dfs[i]['Depth'], c=list_dfs[i][dict_var[v]['var'+str(nrow)]],
                                alpha=0.8, s= s, cmap=dict_var[v]['cmap'+str(nrow)], vmin=dict_var[v]['vmin' + str(nrow)],
                                vmax=dict_var[v]['vmax'+str(nrow)])

        for nrow in range(1,1+nrows) :
            ax = axs[nrow-1]
            cbar = plt.colorbar(list_p[nrow-1], ax=ax, aspect = 10)  # , ticks=1)#ax=ax
            cbar.ax.tick_params(labelsize=fontsize-1, width = 0.5)
            cbar.set_label(label=dict_var[v]['disp'+str(nrow)], fontsize=fontsize-0.5)
            cbar.outline.set_linewidth(0.5)

        if condition:
            # Bidouille pour ajouter les valeurs de %tide
            data = pd.DataFrame(list_tide)
            data = data.rename(columns={0: 'Tide'})
            data['Tide'].where(data['Tide'] > 0, other=100 + (data['Tide'] + 100), inplace=True)
            shift = 1.5
            # 3/09 bidouille to add 100 to all values > 0 but < 100 after having some >100 values, to keep the logical addition
            if (data['Tide'] > 100).any():
                if data['Tide'].last_valid_index() != 100:
                    print('cas')
                    last_index = data[data['Tide'] > 100].index.max()
                    selected_values = data.loc[last_index + 1:][data['Tide'] < 100]['Tide']
                    data[data['Tide'].isin(selected_values.values)] = data[data['Tide'].isin(selected_values.values)] +200

            ax = axs[0]  # In order to have it over the 2 subplots
            for spine in ax.spines.values():
                spine.set_linewidth(0.2)  # Adjust the linewidth as needed
            twin2 = ax.twiny()
            twin2.set_xlim(np.nanmin(data['Tide']) - shift,
                           np.nanmax(data['Tide']) + shift)
            twin2.set_xticklabels(set_newticks(twin2), fontsize=fontsize)
            #twin2.tick_params(axis='both', which='both', labelsize=fontsize, width=0.5)
            date_form = DateFormatter("%H:%M")  # Define the date format
            ax.xaxis.set_major_formatter(date_form)

        save = True
        if save:
            outfile = 'test_All_data_' + month + '_' + t + '.png'
            fig.savefig(outfile)
            print('fig saved')

        ws_D50_figure = False
        if ws_D50_figure:
            if not all_month_together :
                out1 = "Ws_vs_D50_"+month + '.png'
                # To know how the min and max densities encoutered in the stations.
                # list_dens_max, list_dens_min= [], []
                # for a in range(len(list_dfs)):
                #     list_dens_max.append(np.nanmax(list_dfs[a]['Density']))
                #     list_dens_min.append(np.nanmin(list_dfs[a]['Density']))
                # dens_min = np.round(np.nanmin(list_dens_min), 1)-0.1
                # dens_max = np.round(np.nanmax(list_dens_max), 1)
                # print('Dens : ', dens_min, dens_max)
                fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
                fig.suptitle(month, fontsize=fontsize)
                ax.grid(True, alpha=0.5)
                ax.set_ylabel('Ws (mm/s)', fontsize=fontsize)
                ax.set_xlabel('D50 (µm)', fontsize=fontsize)
                ax.set_xlim(0,160)
                ax.set_ylim(0,0.5)
                ax.tick_params(axis='both', which='both', labelsize=fontsize, width=0.5)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)  # Adjust the linewidth as needed
                for i in range(len(list_dfs)):
                    p1 = ax.scatter(list_dfs[i]['D50'], list_dfs[i]['ws'], alpha=0.8, s=12, c=list_dfs[i]['Density'],
                                    cmap=cmap_D50, vmin=995, vmax=1020)
                cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
                cbar.ax.tick_params(labelsize=fontsize - 1, width=0.5)
                cbar.set_label(label='Density', fontsize=fontsize - 0.5)
                cbar.outline.set_linewidth(0.05)
                fig.savefig(out1)
            if all_month_together :
                for i in range(len(list_dfs)):
                    p1 = ax_all.scatter(list_dfs[i]['D50'], list_dfs[i]['ws'], alpha=0.8, s=25, c=list_dfs[i]['Density'],
                                        cmap=cmap_D50, vmin=995, vmax=1020, marker = dict_marker[month], label=month)

        # several plots with 2 subplots
        plot_variable = False
        if plot_variable :
            for v in variables:
                outfile = v + '_' + month + '_' + t + '.png'
                ylim = dict_limtransect[month][t]['ylim']
                fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True)
                fig.suptitle(t + ' ' + str(list_dfs[0]['Time'][0].day) + ' ' + month)
                ax = axs[0]
                ax.grid(True, alpha=0.5)
                ax.set_xlim(xlim1, xlim2)
                ax.set_ylabel('Depth (m)', fontsize=fontsize)
                ax.set_ylim(ylim, 0)

                ax = axs[1]
                ax.grid(True, alpha=0.5)
                ax.set_xlim(xlim1, xlim2)
                ax.set_ylabel('Depth (m)', fontsize=fontsize)
                ax.set_ylim(ylim, 0)

                for i in range(len(list_dfs)):
                    if condition:
                        list_tide.append(list_dfs[i]['Percentage of tide'][0])
                    ax = axs[0]
                    velocity_and_direction(t, month, list_dfs[i], list_name[i])
                    # TODO : si tout fonctionne,l'ajouter au fichier recap pour ne pas le recalculer à chq fois
                    p1 = ax.scatter(list_dfs[i][x], -list_dfs[i]['Depth'], c=list_dfs[i][dict_var[v]['var1']],
                                    alpha=0.8, cmap=dict_var[v]['cmap1'], vmin=dict_var[v]['vmin1'],
                                    vmax=dict_var[v]['vmax1'])
                    ax = axs[1]
                    p2 = ax.scatter(list_dfs[i][x], -list_dfs[i]['Depth'], c=list_dfs[i][dict_var[v]['var2']],
                                    alpha=0.8, cmap=dict_var[v]['cmap2'], vmin=dict_var[v]['vmin2'],
                                    vmax=dict_var[v]['vmax2'])

                ax = axs[0]
                cbar = plt.colorbar(p1, label=dict_var[v]['disp1'], ax=ax)  # , ticks=1)#ax=ax
                cbar.ax.tick_params(labelsize=8)
                ax = axs[1]
                cbar = plt.colorbar(p2, label=dict_var[v]['disp2'], ax=ax)  # , ticks=1)#ax=ax
                cbar.ax.tick_params(labelsize=8)

                if condition:
                    # Bidouille pour ajouter les valeurs de %tide
                    data = pd.DataFrame(list_tide)
                    data = data.rename(columns={0: 'Tide'})
                    data['Tide'].where(data['Tide'] > 0, other=100 + (data['Tide'] + 100), inplace=True)
                    shift = 1.5
                    ax = axs[0]  # In order to have it over the 2 subplots
                    twin2 = ax.twiny()
                    twin2.set_xlim(np.nanmin(data['Tide']) - shift,
                                   np.nanmax(data['Tide']) + shift)
                    twin2.set_xticklabels(set_newticks(twin2))
                    date_form = DateFormatter("%H:%M")  # Define the date format
                    ax.xaxis.set_major_formatter(date_form)

                fig.savefig(outfile)


if all_month_together :
    cbar = plt.colorbar(p1, ax=ax_all)  # , ticks=1)#ax=ax
    cbar.ax.tick_params(labelsize=fontsize - 1, width=0.5)
    cbar.set_label(label='Density', fontsize=fontsize - 0.5)
    cbar.outline.set_linewidth(0.05)
    fig_all.savefig(out)

print('okgoogle')
sys.exit(1)