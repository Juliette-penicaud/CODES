# 22/02/2023 : programme qui lit le fichier de décharge de Son Tay et TT
# OBJECTIF : comparer les 2 données pour voir si 14.5% de Son Tay est une approx ok

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

rep='/home/penicaud/Documents/Data/Décharge_waterlevel/'
fichier=rep+'Data_2017.xlsx'
save=False
TT=False #if TT false : on compare avec les 7 rivières de la config de Violaine (on en mettra 9 in fine)

lag_calcul=1 #0 no calculation of lag, 1, 1 calc 2, 2 calculations

col_list_sontay=['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']
#Download data of Son Tay
df_sontay = pd.read_excel(fichier, sheet_name='Q-Sontay2017', skiprows=3 , usecols=col_list_sontay)
#df_sontay=pd.DataFrame(data_sontay)
print(df_sontay)

#Download data of TT
col_list=['Date', 'Time (hours)', 'Water level (cm)', 'Q (flood) (m3/s)']
data_TT=pd.read_excel(fichier, sheet_name='Q_trungtrang_vanuc_2017', skiprows=3, usecols=col_list)
#print(data_TT)

df_TT=pd.DataFrame(data_TT, columns=col_list)
Date=df_TT['Date']
Time=df_TT['Time (hours)']
WL=df_TT['Water level (cm)']
Q_TT=df_TT['Q (flood) (m3/s)']

#réorganiser les données pour avoir une liste de jour et une liste de décharges associées
liste_sontay=[]
for ind in col_list_sontay:
    liste_sontay.append(df_sontay[ind].values) # produit une liste d'array (12*31 avec des nan pour les mois de moins de 31j)
#print(liste_sontay, np.shape(liste_sontay))

liste_sontay=np.concatenate(liste_sontay).ravel() #transforme en liste de 12*31 aplanie
liste_sontay=[x for x in liste_sontay if str(x) != 'nan'] #suppress all the nan
#print(l2, np.shape(l2))

#if save:
df_liste_sontay=pd.DataFrame(liste_sontay)
with pd.option_context('display.max_rows', None, ):
    print(df_liste_sontay)

#df_liste_sontay.to_csv(rep+'Décharge_ST_2017.txt', index=False, header=None)#, columns=['Trung Trang'] )
#sys.exit(1)

import datetime
#Objectif 1 : faire la moyenne journalière. I.E : toutes les 24 lignes
Date_unique = Date.drop_duplicates(keep='first')
# #print(Date_unique[20:60])
# Date_unique_bis=Date_unique[47:51]
# for d in Date_unique_bis :
#     print(d)
#     print(Q_TT[np.where(df_TT['Date']==d)[0]])

Q_moy_jour_TT=[]
for d in Date_unique :
    moy_jour=np.nanmean(Q_TT[np.where(df_TT['Date']==d)[0]])
    #print('d =',d)
    #print('moy jour = ', moy_jour)
    Q_moy_jour_TT.append(moy_jour)
corr=np.corrcoef(Q_moy_jour_TT, liste_sontay)
corr=np.round(corr[0,1],4)
print('corr', corr)
df_Q_moy_jour=pd.DataFrame(list(zip(liste_sontay,Q_moy_jour_TT)), columns=['Son Tay', 'Trung Trang'])
print(df_Q_moy_jour)

if save:
    df_Q_moy_jour.to_csv(rep+'Décharge_TT_2017.txt', index=False, header=None, columns=['Trung Trang'] )
    sys.exit(1)

#Idee : trouver le lag pour lequel la corrélation est la meilleure
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    result = pd.Series(dftest[0:4], index=['Test Statistic','P - value','Lags Used','No of Observations'])
    for key, value in dftest[4].items():
        result['Critical Value (%s)' % key] = value
    return result
#print(adf_test(df_Q_moy_jour['Trung Trang']))
#print(adf_test(df_Q_moy_jour['Son Tay']))

#Detecte la correlation pour des lags à tester , si mode='full' teste de -len à +len, ici -364 à +364
from scipy import signal
correlation = signal.correlate(df_Q_moy_jour['Son Tay'], df_Q_moy_jour['Trung Trang'], mode="full")
lags = signal.correlation_lags(df_Q_moy_jour['Son Tay'].size, df_Q_moy_jour['Trung Trang'].size, mode="full")
lag = lags[np.argmax(correlation)]
print('len corr, correlation[lag], lag', len(correlation), correlation[lag], lag)
#Meilleure correlation pour lag = -1

def df_shifted(df, target=None, lag=0):
    if not lag and not target:
        return df
    new = {}
    for c in df.columns:
        if c == target:
            new[c] = df[target]
        else:
            new[c] = df[c].shift(periods=lag)
    return  pd.DataFrame(data=new)

if lag_calcul>=1:
    df_new=df_shifted(df_Q_moy_jour, target='Trung Trang', lag=+1)
    print('corr non shifted, ', df_Q_moy_jour.corr().values[0,1], 'df shifted +1 ', df_new.corr().values[0,1])
    if lag_calcul==2:
        df_new2=df_shifted(df_Q_moy_jour, target='Trung Trang', lag=-1)
        print('-1 corr', df_new2.corr().values[0,1])
        df_new3=df_shifted(df_Q_moy_jour, target='Trung Trang', lag=-2)
        print('-2 d corr', df_new3.corr().values[0,1])


if TT : # 3 images pour comparer et valider les données TT ST de 2017

    fontsize=8
    fig, axs = plt.subplots(ncols=1, nrows=3, sharex=True)  # , ncols=2)#fonctionne avec plot:share=1,left=3 , right=5,bottom=5,top=7,wspace=10, hspace=5)
    fig.suptitle('Daily Discharge', fontsize=fontsize+2)

    ax=axs[0]
    t1 = ax.plot(Date_unique, liste_sontay , color='blue', label='Son Tay', linewidth=1)
    t2 = ax.plot(Date_unique, Q_moy_jour_TT, color='k', label='Trung Trang, corr='+str(corr)+'\n corr 1d lag ='+str(np.round(df_new.corr().values[0,1], 4)), linewidth=1)
    ax.set_ylabel('Discharge (m3/s)', fontsize=fontsize)
    ax.yaxis.set_major_locator(MultipleLocator(5000))
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(MultipleLocator(2500))
    #ax.yaxis.grid(b=True, which='minor', color='lightgrey', linestyle='-')
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.yaxis.grid(b=True, which='minor', color='lightgrey', linestyle='-')
    ax.tick_params(axis='y', labelsize=fontsize)

    ax = axs[1]
    ax.set_ylim(-900, 300)
    ax.set_ylabel('Discharge (m3/s)', fontsize=fontsize)
    ax.yaxis.set_major_locator(MultipleLocator(300))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(MultipleLocator(100))
    plt.grid(True)
    #ax.yaxis.grid(b=True, which='minor', color='lightgrey', linestyle='-')
    ax.axhline(-200, color='k',lw=0.8, linestyle='dashed')
    ax.axhline(200, color='k', lw=0.8, linestyle='dashed')
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.yaxis.grid(b=True, which='minor', color='lightgrey', linestyle='-')
    ax.tick_params(axis='y', labelsize=fontsize)

    ax = axs[2]
    ax.set_ylim(-100, 250)
    ax.set_ylabel('% relative error', fontsize=fontsize)
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    ax.axhline(-50, color='k',lw=0.8, linestyle='dashed')
    ax.axhline(50, color='k', lw=0.8, linestyle='dashed')
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.yaxis.grid(b=True, which='minor', color='lightgrey', linestyle='-')
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)


    list_percentage=[18, 20, 22]
    listcolor = ['red', 'darkgreen', 'gold', 'darkgreen', 'darkred', 'coral']
    listcolorbis = ['darkgreen','darkred', 'indigo']
    lw=0.8
    alpha=0.5

    for p in range(len(list_percentage)):
        percentage=list_percentage[p]
        print("p and corresponding percentage", p, percentage)
        liste2= [x * percentage/100 for x in liste_sontay]
        diff_2Q=np.subtract(np.array(liste2),np.array(Q_moy_jour_TT)) #map(operator.sub, liste2, Q_moy_jour)
        rel_2Q=100*(diff_2Q/Q_moy_jour_TT)
        #avec le lag :
        liste2_lag= percentage/100*df_new['Son Tay']
        diff_2Q_lag=liste2_lag-df_new['Trung Trang']
        #print('df_new', df_new)
        rel_2Q_lag=100*(diff_2Q_lag/df_new['Trung Trang'])
        #print('l2 lag', diff_2Q_lag, diff_2Q)
        #print(Q_moy_jour_TT)
        #print('Min q moy TT et ST', np.min(Q_moy_jour_TT), np.min(liste_sontay))
        #print('relative error', np.max(rel_2Q), np.where(rel_2Q==np.max(rel_2Q))[0][0],
        #      liste2[np.where(rel_2Q==np.max(rel_2Q))[0][0]], Q_moy_jour_TT[np.where(rel_2Q==np.max(rel_2Q))[0][0]])
        ax=axs[0]
        t3 = ax.plot(Date_unique, liste2, color=listcolor[p], linewidth=lw, alpha=alpha,label=str(percentage) + '% of Son Tay')

        if not lag_calcul:
            ax=axs[1]
            #correlation=np.corrcoef(Q_moy_jour_TT, liste2)
            #print('correlation', correlation)
            t1 = ax.plot(Date_unique, diff_2Q, color=listcolor[p], linewidth=lw,alpha=alpha, label='mean value : '+str(np.round(np.nanmean(diff_2Q),4))+' STD '+str(np.round(np.std(diff_2Q),4)))#, label= 'correlation :'+str(correlation))
            ax.legend(loc='lower left', fontsize=fontsize-3)
            ax=axs[2]
            t2 = ax.plot(Date_unique, rel_2Q, color=listcolor[p],linewidth=lw, alpha=alpha, label='mean value : '+str(np.round(np.nanmean(rel_2Q),4))+' STD '+str(np.round(np.std(rel_2Q),4)))
            ax.legend(loc='lower left', fontsize=fontsize-3)

        if lag_calcul>=1 :
            axs[0].plot(Date_unique, liste2_lag, color=listcolorbis[p], linewidth=lw-0.1, alpha=alpha,
                          label=str(percentage) + '% of Son Tay, 1d lag')
            axs[1].plot(Date_unique, diff_2Q_lag, color=listcolorbis[p], linewidth=lw, alpha=alpha, label='mean value : '+str(np.round(np.nanmean(diff_2Q_lag),4))+' STD '+str(np.round(np.std(diff_2Q_lag),4)))
            axs[1].legend(loc='lower left', fontsize=fontsize - 3)
            axs[2].plot(Date_unique, rel_2Q_lag, color=listcolorbis[p], linewidth=lw, alpha=alpha, label='mean value : '+str(np.round(np.nanmean(rel_2Q_lag),4))+' STD '+str(np.round(np.std(rel_2Q_lag),4)))
            axs[2].legend(loc='upper right', fontsize=fontsize-3)

    axs[0].legend(loc='upper left', fontsize=fontsize-3)
    #axs[1].legend(loc='upper right', fontsize=fontsize - 3)
    #axs[2].legend(loc='upper right', fontsize=fontsize - 3)


    rep2='/home/penicaud/Documents/Data/Décharge_waterlevel/'
    outfile=rep2+'Décharge_TT_SonTay_'+str(percentage)+'%'
    if lag_calcul:
        outfile=outfile+'_+1daylag_'
    outfile=outfile+'.png'
    plt.savefig(outfile, format='png', dpi=600)

else : #on va regarder les autres rivières et comprendre leur %
    sys.exit(1)



sys.exit(1)
################################################################################################################
################################################################################################################

#Code qui fonctionne pour faire une image de  2 lignes avec en haut le débit de ST de TT et le W% de ST. En dessous la différence entre les 2 pui une erreur rel : diff/debit à TT en %

percentage=20
liste2= [x * percentage/100 for x in liste_sontay]
diff_2Q=np.subtract(np.array(liste2),np.array(Q_moy_jour_TT)) #map(operator.sub, liste2, Q_moy_jour)
#print(diff_2Q)
rel_2Q=100*(diff_2Q/Q_moy_jour_TT)
print(Q_moy_jour_TT)
print('Min q moy TT et ST', np.min(Q_moy_jour_TT), np.min(liste_sontay))
print('relative error', np.max(rel_2Q), np.where(rel_2Q==np.max(rel_2Q))[0][0],
      liste2[np.where(rel_2Q==np.max(rel_2Q))[0][0]], Q_moy_jour_TT[np.where(rel_2Q==np.max(rel_2Q))[0][0]])
fontsize=11

fig, axs = plt.subplots(ncols=1, nrows=2)  # , ncols=2)#fonctionne avec plot:share=1,left=3 , right=5,bottom=5,top=7,wspace=10, hspace=5)
fig.suptitle('Daily Discharge')
ax=axs[0]
#ax.set_xlim(26, 32.1)  # (25,31)
# t1 = ax.scatter(T_LOG, -depth_LOG, alpha=0.8, marker='x', color='red', label='CTD LOG')
# t1=ax.plot(T_LOG, -depth_LOG, color='red', label='CTD LOG '+hour_LOG+'h'+mn_LOG )  # color=color[i], label=hour+'h'+mn)
# t2 = ax.scatter(T_imer, -depth_imer, alpha=0.8, marker='x', color='gold', label='CTD IMER')
t1 = ax.plot(Date_unique, Q_moy_jour_TT, color='gold', label='Trung Trang')
t2 = ax.plot(Date_unique, liste_sontay , color='blue', label='Son Tay')
t3 = ax.plot(Date_unique, liste2 , color='brown', label=str(percentage)+'% of Son Tay') #linestyle='dotted',
#ax.set_xlabel('Date', fontsize=fontsize)  # ('Conservative Temperature (°C)', fontsize=fontsize)
ax.set_ylabel('Discharge (m3/s)', fontsize=fontsize)
ax.yaxis.set_major_locator(MultipleLocator(5000))
#ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.yaxis.set_minor_locator(MultipleLocator(2500))
ax.yaxis.grid(b=True, which='minor', color='lightgrey', linestyle='-')
ax.legend(loc='upper left', fontsize=fontsize - 3)

ax=axs[1]
ax.set_ylim(-1250, 1250)
t1 = ax.plot(Date_unique, diff_2Q, color='grey', label='absolute difference')
t2 = ax.plot(Date_unique, rel_2Q, color='red', label='relative error (%)')
ax.set_xlabel('Date', fontsize=fontsize)  # ('Conservative Temperature (°C)', fontsize=fontsize)
ax.set_ylabel('Discharge (m3/s)', fontsize=fontsize)
ax.yaxis.set_major_locator(MultipleLocator(500))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.yaxis.set_minor_locator(MultipleLocator(250))
ax.legend(loc='upper right', fontsize=fontsize - 3)
plt.grid(True)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
ax.yaxis.grid(b=True, which='minor', color='lightgrey', linestyle='-')

rep2='/home/penicaud/Documents/Data/Décharge_waterlevel/'
outfile=rep2+'Décharge_TT_SonTay_'+str(percentage)+'%.png'
if save :
    plt.savefig(outfile, format='png', dpi=600)