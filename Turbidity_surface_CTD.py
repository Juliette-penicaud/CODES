# 24/02/2023 : programme qui lit les valeurs de la turbidity de surface des campagnes de la CTD IMER
# OBJECTIF : comparer les données de turb de la CTD et du turbidimètre

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, TheilSenRegressor
import scipy.signal as signal


# 22/09 : je rajoute le Theil Sen estimator et/ou OLS pour améliorer le critère
estimators = [
    ("OLS", LinearRegression()),
    ("Theil-Sen", TheilSenRegressor(random_state=42))]
#    ("RANSAC", RANSACRegressor(random_state=42)),
colors = {"OLS": "black", "Theil-Sen": "orange"}#, "RANSAC": "lightgreen"}
lw = 2

rep = '/home/penicaud/Documents/Data/CTD/'
create = 0  # create the tab 1 or read already saved one 0
save = 1
list_month = ['06.1', '06.2', '08', '10']

if create:  # Creation of a txt file with all turbidity of each station, from the CTD IMER turbidity
    df_tot = pd.DataFrame()
    for month in list_month:
        if month == '06.1':
            survey = 'Survey_June'
            file_imer = 'CTD_VU-I.xlsx'  # survey of 2- JUNE
            deb = 1
        elif month == '06.2':
            survey = 'Survey_June'
            file_imer = 'CTD_VU_16-20_06.xlsx'  # Survey to 16-20 june
            deb = 0
        elif month == '08':
            survey = 'Survey_August'
            file_imer = 'CTD_imer_aug2022.xlsx'
            deb = 2
        elif month == '10':
            survey = 'Survey_Octobre'
            file_imer = 'CTD_imer_octobre.xlsx'
            deb = 2  # to remove the vu-oct with no coherent data
        else:
            print('error in Month')
            sys.exit(1)

        f_imer = rep + survey + '/' + file_imer
        print('f_imer', f_imer)
        tabs = pd.ExcelFile(f_imer).sheet_names  # All the sheets name i.e the stations
        print(type(tabs), tabs[deb:len(tabs)])
        tab = []
        for t in tabs[deb:len(tabs)]:  # loop over all the stations of each station
            # print('sheet name', t)
            df = pd.read_excel(f_imer, sheet_name=str(t), skiprows=23, usecols=['Turbidity'])
            # print(df)
            tab.append(df.to_numpy().flatten())

        df_tab = pd.DataFrame(tab, dtype=object, index=tabs[deb:len(
            tabs)])  # Ok pour avoir les stations en rows, les valeurs en lignes, transpose
        df_tab = df_tab.T
        print('df_tab\n', df_tab)
        # df_tot.join(df_tab)
        if df_tot.empty:
            df_tot = df_tab
            print('df_tot empty, I copy df_tab')
        else:
            print('df_tot not empty, I try to join both df')
            # df_tot.join(df_tab)
            df_tot = pd.concat([df_tot, df_tab], axis=1)
            print('df_tot\n', df_tot)
        # try:
        #     df_tot = pd.merge(df_tot, df)#, how='outer', left_index=True, right_on='z')
        # except IndexError:
        #     df_tot = df_tot.reindex_axis(df_tot.columns.union(df.columns), axis=1)

    if save == 1:  # save the df to csv
        outfile = rep + 'All_turbidity.txt'
        df_tot.to_csv(outfile, index=False, header=True)

else:
    rep2 = '/home/penicaud/Documents/Data/Turbidity/'
    f_CTD = rep + 'All_turbidity_CTD.txt'
    df_tot2 = pd.read_csv(f_CTD, sep=',') # df with all the turbidity values at every depth
    # print('df_tot2\n',df_tot2)
    column_headers = list(df_tot2.columns.values)
    diff_cond = 0.5  # difference to check, *100=percentage of variation
    test_std = 10
    n_prof = 10 # select only from 0 to n_prof first cells
    passing_by_zero = True # For the linear regression : add a 0 to both Series to have a regression passing close to zero)
    # OBJ : trouver ou il y probleme entre les premières lignes de chaque station :
    dfT = df_tot2.T
    print(dfT)
    dfT = dfT.iloc[:, :n_prof]

    # OBJ : 1er process avec les données : mediane
    # dfT_filtered = signal.medfilt(df_tot2) # filter 3
    # dfT_filtered = df_tot2.rolling(5).median() #PB : put NANs until all value are not NAN
    # print(dfT_filtered[:][0:15])
    # print(df_tot2[0:20][0:10])
    # print(np.shape(dfT_filtered), dfT_filtered[0])
    # dfT = dfT.iloc[:, :n_prof]
    # sys.exit(1)

    # too free tests, not robuste
    # ############# TEST 1 ###################"
    # print("################### TEST1")
    # pb_df=dfT.loc[(abs(dfT[1]-dfT[0])>diff_cond)] #condition : select where difference of 2 first col > diff_cond
    # pb_df=pb_df.iloc[:,:n_prof]
    # print(pb_df)
    #
    # ############# TEST 2 ###################"
    # print("################### TEST2")
    # pb_df=dfT.loc[(abs(dfT[4]-dfT[0])>diff_cond)] #condition : select where difference of 2 first col > diff_cond
    # pb_df=pb_df.iloc[:,:n_prof]
    # print(pb_df)
    #
    # ############# TEST 3 ###################"
    print("###################  TEST3")
    pb_df = dfT.loc[(abs(dfT[1] - dfT[0]) > 20)]  # condition : select where difference of 2 first col > diff_cond
    # pb_df=pb_df.iloc[:,:n_prof]
    # print(pb_df)
    #
    # ############# TEST 4 ###################"
    # print("###################  TEST4")
    # pb_df=dfT.loc[(abs(dfT[1]-dfT[0])/dfT[1] >0.5)] #condition : select where difference of 2 first col > diff_cond
    # #pb_df=pb_df.iloc[:,:n_prof]
    # print(pb_df)
    #
    # ############# TEST 5 ###################"
    # print("###################  TEST5")
    # pb_df=dfT.loc[(abs(dfT[3]-dfT[0])/dfT[3] >0.5)] #condition : select where difference of 2 first col > diff_cond
    # #pb_df=pb_df.iloc[:,:n_prof]
    # print(pb_df)

    # condition = (abs(dfT.mean(axis=1)-dfT[0])/dfT.mean(axis=1) > diff_cond) or (dfT.std(axis=1)> 20) or (dfT[0]>300)
    ############# TEST 6 ###################" BIEN mais ne détecte pas O26
    print("###################  TEST6")
    pb_df1 = dfT.loc[(abs(dfT.mean(axis=1) - dfT[0]) / dfT.mean(
        axis=1) > diff_cond)]  # condition : select where difference of 2 first col > diff_cond
    # print(pb_df1)
    # print(dfT.mean(axis=1))

    # ############# TEST 6 ###################"
    # print("###################  TEST6")
    # pb_df=dfT.loc[(abs(dfT.mean(axis=1)-dfT[2])/dfT.mean(axis=1) > diff_cond)] #condition : select where difference of 2 first col > diff_cond
    # #pb_df=pb_df.iloc[:,:n_prof]
    # print(pb_df)
    # #print(dfT.mean(axis=1))

    ############# TEST 7 ###################" #Plus permissif, peut fiare ressortir des variations assez importantes mais pas délirantes, non captées par la méthode de la moyenne
    print("###################  TEST7")
    pb_df2 = dfT.loc[dfT.std(axis=1) > 20]  # condition : select where difference of 2 first col > diff_cond
    # print(pb_df2)
    # print(pb_df.std(axis=1))

    print("###################  TEST8")
    pb_df3 = dfT.loc[dfT.mean(axis=1) > 250]
    print(pb_df3)

    print("###################  DF all ")
    pb_all = pd.concat([pb_df1, pb_df2, pb_df, pb_df3], axis=0, sort=True).drop_duplicates()
    print(pb_all.sort_index())
    print(np.shape(df_tot2))
    print('shape pb all ', np.shape(pb_all))

    # Nouveau tableau ave la moyenne OU MEDIANE des X premieres mesures
    nb_stat = 7  # 3 or 5 : filtermedian to apply
    df = dfT.iloc[:, :nb_stat]
    var = 'median'
    if var == 'moy':
        df = df.mean(axis=1)
    elif var == 'median':
        df = df.median(axis=1)
    df.index.name = 'Station'
    # df.columns=["Turbidity (CTD)"]
    print('df\n', df)

    # Je charge les données du turbidimetre
    f_turb = rep2 + 'All_surveys_turbiditimeter.xlsx'
    d_turbidimeter = pd.read_excel(f_turb)
    data_turbidimeter = pd.DataFrame(d_turbidimeter, columns=['Station', 'Turbidity (turbidimeter)'])
    data_turbidimeter = data_turbidimeter.set_index(['Station'])
    print(data_turbidimeter)
    print(data_turbidimeter.count())

    all_data = df.copy()
    all_data = all_data.to_frame()
    all_data = all_data.join(data_turbidimeter, how='inner')
    all_data = all_data.rename(columns={0: "CTD", 'Turbidity (turbidimeter)': "Turbidi"})
    # with pd.option_context('display.max_rows', None, ):
    #    print(all_data)
    print(all_data)


    # I want to select only the rows where no pb CTD were detected
    if var == 'mean':
        all_data_nopb = all_data[~all_data.index.isin(pb_all.index)]
        all_data_pb = all_data[all_data.index.isin(pb_all.index)]
        print('all_data_nopb\n', all_data_nopb)
    elif var == 'median':
        all_data_pb = all_data.loc[df > 250]
        all_data_nopb = all_data.loc[df <= 250]

    print('DRAW A FIGURE')
    ###################### FIGURE   ##################################################################################"
    fontsize = 10
    to_compare = 'data_no_pb'
    if to_compare == 'data_no_pb':
        xlim = 200
        ylim = 200
        a = all_data_nopb['Turbidi']
        b = all_data_nopb['CTD']
        loc = 'upper right'
    elif to_compare == 'all_data':
        xlim = 250
        ylim = 600
        a = all_data['Turbidi']
        b = all_data['CTD']
        c = all_data_pb['Turbidi']
        d = all_data_pb['CTD']
        loc = 'upper left'

    outfile = rep2 + 'Turbidity_turbidmietervsCTD_' + to_compare + '_' + var + str(nb_stat) + '.png'
    fig, axs = plt.subplots(
        ncols=1)  # , ncols=2)#fonctionne avec plot:share=1,left=3 , right=5,bottom=5,top=7,wspace=10, hspace=5)
    fig.suptitle('Comparison of turbidity CTD and Turbidimeter')
    ax = axs
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.set_xlabel('Turbidity (turbidimeter) FTU',
                  fontsize=fontsize)  # ('Conservative Temperature (°C)', fontsize=fontsize)
    ax.set_ylabel('Turbidity (CTD) FTU', fontsize=fontsize)
    ax.plot(np.arange(0, 250), np.arange(0, 250), color='red', alpha=0.5)
    # ax.xaxis.set_major_locator(MultipleLocator(2))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # ax.xaxis.set_minor_locator(MultipleLocator(1))

    idx = np.isfinite(a) & np.isfinite(b)
    # print('idx, ', np.where(idx==False), a[np.where(idx==False)[0]], b[np.where(idx==False)[0]])
    if passing_by_zero :
        a = a.append(pd.Series([len(a)]))
        a[len(a)]==0
        b = b.append(pd.Series([len(b)]))
        b[len(b)] == 0
    z = np.polyfit(a[idx], b[idx], 1)
    p = np.poly1d(z)

    t = ax.scatter(a, b, color='grey', marker='o', alpha=0.8, s=6, lw=1)  # , trendline="ols")
    if to_compare == 'all_data':
        t2 = ax.scatter(c, d, color='gold', marker='o', alpha=0.8, s=6, lw=1)  # , trendline="ols")

    text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(b[idx], a[idx]):0.3f}$"
    plt.plot(a, p(a), c='cornflowerblue', label=text)  # ="y=%.6fx+%.6f" % (z[0], z[1]))
    ax.legend(loc=loc)
    # text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(b[idx], a[idx]):0.3f}$"
    # plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
    #               fontsize=14, verticalalignment='top')

    plt.savefig(outfile, format='png', dpi=600)


    # 22/09 : try with no condition, just by adding theil sen estimator
    fig, axs = plt.subplots(
        ncols=1)  # , ncols=2)#fonctionne avec plot:share=1,left=3 , right=5,bottom=5,top=7,wspace=10, hspace=5)
    fig.suptitle('Comparison of turbidity CTD and Turbidimeter')
    ax = axs
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.set_xlabel('Turbidity (turbidimeter) FTU',
                  fontsize=fontsize)  # ('Conservative Temperature (°C)', fontsize=fontsize)
    ax.set_ylabel('Turbidity (CTD) FTU', fontsize=fontsize)
    ax.plot(np.arange(0, 250), np.arange(0, 250), color='red', alpha=0.5)
    # ax.xaxis.set_major_locator(MultipleLocator(2))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # ax.xaxis.set_minor_locator(MultipleLocator(1))
    t = ax.scatter(a, b, color='grey', marker='o', alpha=0.8, s=6, lw=1)  # , trendline="ols")

    line_x = np.array([np.min(all_data["Turbidi"]), np.max(all_data["Turbidi"])])  # commun à tous, axe de turbidité max
    a = all_data['CTD']
    a = a[:, np.newaxis]
    b = all_data['Turbidi']
    idx = np.isfinite(a) & np.isfinite(b)
    for name, estimator in estimators[0]:
        estimator.fit(a[idx], b[idx])
        # elapsed_time = time.time() - t0
        y_pred = estimator.predict(line_x.reshape(2, 1))
        ax.plot(
            line_x,
            y_pred,
            color=colors[name],
            linewidth=lw,
            label="%s, R² = %2f" % (name, estimator.score(a[idx], b[idx]))  # (fit time: %.2fs)" % (name, elapsed_time),
        )
