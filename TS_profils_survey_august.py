#04/07/2022 faire profils des TS a partir des fichiers excels /csv de la petite CTD
import csv
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression, TheilSenRegressor
#from sklearn.linear_model import RANSACRegressor
#import csv
import sys, os, glob
#import xarray as xr
#import matplotlib.colors as mcolors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import gsw
import random
import pickle
from itertools import zip_longest
from matplotlib.lines import Line2D

#2 versions : day by day or all data

#CC1238004_20220604_050128
year='2022'
month= '08'
alldayfig=0 #1 to plot all the data on one only figure
transectfig=1 #1 to plot transect by transect

if month=='06':
    rep='/home/penicaud/Documents/Data/CTD/Survey_June/CTD_LOG/'
    day = ['16','17','18']
    if transectfig:
        transect = ['SFJ'] # ['TJ1', 'TJ2', 'TJ3', 'TJ4', 'SFJ']  # better to put the transect at the end, to have the colorbar adapted, because many measurements
elif month=='08':
    rep='/home/penicaud/Documents/Data/CTD/Survey_August/CTD_LOG/'
    if alldayfig:
        day = ['10', '11', '12']#, '13']  # '11','12','13', '14']
    else :
        day = ['10','11', '12','13']
        if transectfig:
            transect= ['SF_24'] # ['TA1', 'TA2', 'TA3', 'TA4', 'SF1', 'SF_24'] #better to put the transect at the end, to have the colorbar adapted, because many measurements
            #transect=['SF_24']
#TODO : make the code more robust for the colors : colors 5 and 6 are adapted to the values of the fixed stations (more colors availables)

init= 'CC1238004_'
suffixe = '.csv'

station = 1 #if add the name of the station o the legend :
conservative_param=0 # CT and SA
colorful=0 #0 if figure in shades of grey ; 1 if colorful
diagramTS=0 #1 : give the TS diagram
diagramTS1fig=0 #all in 1 fig
fontsize=10
maxdepth=[] #list of all the max depth of each profil
ssalinity=[] #surface salinity (choose the 1st or 2d point of each profil
date=[] #record the date of the CTD profil
tempmax, tempmin, sal, density, N =[],[],[],[],[]
var='TSDN2' #TSDN2 : 4 graphs of T , S , D and N2. IF anything else, 3 graphs, T, S , N2

################################################          COLORS         #################################""

if colorful==0:
    print('colorful=0, we are in grey')
    cmap=plt.cm.binary #grey
    #cmap=plt.cm.hsv_r
    # extracting all colors
    color = [cmap(i) for i in range(cmap.N)]
    #color=color[20::20] #good for the fig with all the profiles of T and S of the 18/06 in shades of grey
    # #because the colors go from white to black, choose only from a certain phase and with a step so that we can see some differences
    #color = color[30::25] #good for hsv_r from purple - green -red on 6 curves
    #color1 = color[5::17] #good for hsv_r from purple - green -red on 6 curves
    print('len color init', len(color))
    color= color[20:-1]
    len_color=len(color)
    #color=[color,color,color,color,color,color]
    print('len color 2e', len_color)


elif colorful:
    if transectfig :
        step=30
    else :
        step=5
    print('step color', step)

    cmap=plt.cm.winter#blue
    color1 = [cmap(i) for i in range(cmap.N)]
    color1 = color1[::step+20]

    cmap=plt.cm.autumn#red
    color2 = [cmap(i) for i in range(cmap.N)]
    color2 = color2[::step]

    cmap=plt.cm.summer#green
    color3 = [cmap(i) for i in range(cmap.N)]
    color3 = color3[::15]

    cmap=plt.cm.spring#pink
    color4 = [cmap(i) for i in range(cmap.N)]
    color4 = color4[::step]

    cmap=plt.cm.copper_r#copper if _r, from orange to black brown
    color5 = [cmap(i) for i in range(cmap.N)]
    color5 = color5[::10]

    cmap=plt.cm.hot#brown red yellow
    color6 = [cmap(i) for i in range(cmap.N)]
    color6 = color6[::8]


    if alldayfig==1:
        color=[color1, color2, color3, color3]
    elif transectfig:
        color=[color1, color2, color3, color4, color5, color6] #color 5 and 6 adapted to fixed stations :
else :
    print('PROBLEM WITH COLORFUL : choose 0 or 1')
    sys.exit(1)

#######################################################################################################"

if diagramTS :
    fig, ax = plt.subplots()  # fonctionne avec plot : share=1,left=3 , right=5, bottom=5 , top=7, wspace=10, hspace=5)
    if diagramTS1fig:

        fig.suptitle('All data')
    ax.set_xlabel('Conservative Temperature (°C)', fontsize=fontsize)
    ax.set_ylabel('Absolute Salinity (g/kg)', fontsize=fontsize)
    ax.set_xlim(28.1,32.1)#(27, 31)
    ax.set_ylim(0,32.1)#(0, 31)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    # Add the density lines to the graph
    ################################
    # gsw use conservative temperature CT and SA absolute salinity :
    # Sea pressure ? 1013.25 en moyenne, voir si ca fit
    tempL = np.linspace(27, 33, 100)
    salL = np.linspace(0, 33, 100)
    Tg, Sg = np.meshgrid(tempL, salL)
    sigma_theta = gsw.sigma0(Sg, Tg)
    cs = ax.contour(Tg, Sg, sigma_theta, colors='grey', zorder=1)  # , vmin=0, vmax=25)
    cl = plt.clabel(cs, fontsize=10, inline=True, fmt='% .1f')
    ###############################



# All days with profiles of T S on subplots
# fig, axs = plt.subplots(
#     ncols=3)  # fonctionne avec plot : share=1,left=3 , right=5, bottom=5 , top=7, wspace=10, hspace=5)
# fig.suptitle(month + '/' + year)
# ax = axs[0]
# ax.set_xlim(29, 32.5)  # (28.75, 30)  # (25,31)
# ax.set_ylim(-18, 0)
# # p1 = ax.scatter(CT, -data["Depth (Meter)"],alpha=0.8, marker='x', color='red')#, label='GFF')
# if conservative_param:
#     ax.set_xlabel('Conservative Temperature (°C)', fontsize=fontsize)
# elif conservative_param == 0:
#     ax.set_xlabel('Temperature (°C)', fontsize=fontsize)
# ax.set_ylabel('Depth (m)', fontsize=fontsize)
# ax.xaxis.set_major_locator(MultipleLocator(1))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# # ax.xaxis.set_minor_locator(MultipleLocator(0.5))
#
# ax = axs[1]
# ax.set_xlim(-2, 30)
# ax.set_ylim(-18, 0)
# # p1 = ax.scatter(SA, -data["Depth (Meter)"],alpha=0.8, marker='x', color='blue')#, label='GFF')
# # color=color[i] , label=hour+'h'+mn)#, label='GFF')
# if conservative_param:
#     ax.set_xlabel('Absolute Salinity (g/kg)', fontsize=fontsize)
# elif conservative_param == 0:
#     ax.set_xlabel('Salinity (PSU)', fontsize=fontsize)
# # ax.set_ylabel('Depth (m)', fontsize=fontsize)
# ax.xaxis.set_major_locator(MultipleLocator(10))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# ax.xaxis.set_minor_locator(MultipleLocator(5))
# # ax.legend(loc='lower right')
#
# ax = axs[2]
# # ax.set_xlim(990, 1020)
# ax.set_xlim(0,0.3)
# ax.set_ylim(-18, 0)
# # p1 = ax.scatter(D, -data["Depth (Meter)"],alpha=0.8, marker='x', color='k')#, label='GFF')
# # color=color[i],label=hour + 'h' + mn)  # , label='GFF')
# # ax.set_xlabel('Density (kg/m\u00b3from matplotlib.lines import Line2D)', fontsize=fontsize)
# # ax.set_xlabel('N\u00b2 ($s{-2}$)', fontsize=fontsize)
# ax.set_xlabel(r'$N^2 (s^{-2})$', fontsize=fontsize)
# # ax.set_ylabel('Depth (m)', fontsize=fontsize)
# ax.xaxis.set_major_locator(MultipleLocator(0.10))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.xaxis.set_minor_locator(MultipleLocator(0.05))
# # ax.legend(loc='lower right')
# # set the spacing between subplots
#
# plt.subplots_adjust(left=0.11,
#                     bottom=0.1,
#                     right=0.95,
#                     top=0.9,
#                     wspace=0.4,
#                     hspace=0.4)
#

i1=-1 #manage the colors day by day

if transectfig==0 :
    for d in day:
        i1=i1+1
        # #Figure day by day
        # fig2, ax2 = plt.subplots()  # fonctionne avec plot : share=1,left=3 , right=5, bottom=5 , top=7, wspace=10, hspace=5)
        # fig2.suptitle(d+'/'+month+'/'+year)
        # ax2.set_xlabel('Conservative Temperature (°C)', fontsize=fontsize)
        # ax2.set_ylabel('Absolute Salinity (g/kg)', fontsize=fontsize)
        # ax.set_xlim(28.1, 32.1)  # (27, 31)
        # ax.set_ylim(0.0,32.1)  # (0, 31)
        # ax2.xaxis.set_major_locator(MultipleLocator(1))
        # ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        # ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
        # ax2.yaxis.set_major_locator(MultipleLocator(5))
        # ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        # ax2.yaxis.set_minor_locator(MultipleLocator(1))

        # Add the density lines to the graph
        ################################
        ##gsw use conservative temperature CT and SA absolute salinity :
        ##Sea pressure ? 1013.25 en moyenne, voir si ca fit
        # tempL = np.linspace(28.1,32.1,100)#(27, 33, 100)
        # salL = np.linspace(0, 32.1, 100)
        # Tg, Sg = np.meshgrid(tempL, salL)
        # sigma_theta = gsw.sigma0(Sg, Tg)
        # cs = ax.contour(Tg, Sg, sigma_theta, colors='grey', zorder=1)  # , vmin=0, vmax=25)
        # #cs2 = ax2.contour(Tg, Sg, sigma_theta, colors='grey', zorder=1)  # , vmin=0, vmax=25)
        # cl = plt.clabel(cs, fontsize=10, inline=True, fmt='% .1f')
        # #cl2 = plt.clabel(cs2, fontsize=10, inline=True, fmt='% .1f')
        ##############################

        # #Day by day with several profiles of T S on subplots
        fig, axs = plt.subplots(
            ncols=3)  # fonctionne avec plot : share=1,left=3 , right=5, bottom=5 , top=7, wspace=10, hspace=5)
        fig.suptitle(d+'/'+month + '/' + year)
        ax = axs[0]
        #ax.set_xlim(29, 32.5)  # (28.75, 30)  # (25,31)
        #ax.set_ylim(-18, 0)
        # p1 = ax.scatter(CT, -data["Depth (Meter)"],alpha=0.8, marker='x', color='red')#, label='GFF')
        if conservative_param:
            ax.set_xlabel('Conservative Temperature (°C)', fontsize=fontsize)
        elif conservative_param==0:
            ax.set_xlabel('Temperature (°C)', fontsize=fontsize)
        ax.set_ylabel('Depth (m)', fontsize=fontsize)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # ax.xaxis.set_minor_locator(MultipleLocator(0.5))

        ax = axs[1]
        ax.set_xlim(-2, 30)
        #ax.set_ylim(-18, 0)
        # p1 = ax.scatter(SA, -data["Depth (Meter)"],alpha=0.8, marker='x', color='blue')#, label='GFF')
        # color=color[i] , label=hour+'h'+mn)#, label='GFF')
        if conservative_param:
            ax.set_xlabel('Absolute Salinity (g/kg)', fontsize=fontsize)
        elif conservative_param == 0:
            ax.set_xlabel('Salinity (PSU)', fontsize=fontsize)
        # ax.set_ylabel('Depth (m)', fontsize=fontsize)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        # ax.legend(loc='lower right')

        ax = axs[2]
        #ax.set_xlim(990, 1020)
        #ax.set_xlim(-0.01, 0.3)
        #ax.set_ylim(-18, 0)
        # p1 = ax.scatter(D, -data["Depth (Meter)"],alpha=0.8, marker='x', color='k')#, label='GFF')
        # color=color[i],label=hour + 'h' + mn)  # , label='GFF')
        # ax.set_xlabel('Density (kg/m\u00b3from matplotlib.lines import Line2D)', fontsize=fontsize)
        # ax.set_xlabel('N\u00b2 ($s{-2}$)', fontsize=fontsize)
        ax.set_xlabel(r'$N^2 (s^{-2})$', fontsize=fontsize)
        # ax.set_ylabel('Depth (m)', fontsize=fontsize)
        ax.xaxis.set_major_locator(MultipleLocator(0.10))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        # ax.legend(loc='lower right')
        # set the spacing between subplots

        plt.subplots_adjust(left=0.11,
                            bottom=0.1,
                            right=0.95,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)


        i = 0  # to select the color of the table
        liste=glob.glob(rep+init+year+month+d+'*'+suffixe)     #Go and find all the files
        #For one figure to gather the T and S of one day, specially for the fixed station
        #fig, axs = plt.subplots(ncols=2)#fonctionne avec plot:share=1,left=3 , right=5,bottom=5,top=7,wspace=10, hspace=5)
        #fig.suptitle(d+'/'+month+'/'+year)
        ssalinity=[]
        date=[]
        #print('ssalinity', ssalinity)
        liste2=[] #list to print only certain files : used for the graph with the 10mn files
        #ind=0
        nb_station=len(sorted(liste))
        print('nb station', nb_station)
        if month=='06' and day=='18':
            minsec=['4006','5025','0010','0853','1744','2943','4041','5436','0410','1322'] #selected min+sec to put on a graph
        for f in sorted(liste):
            #make a graph for every 10 mn from 17h30 to 19h. 17h40, 17h50, 18h00, 18h08, 18h17, 18h29, 18h40, 18h54,  19h04
            print(f)
            # if station :
            #     hour = str(int(f[-14:-12]) + 7)#(f[-10:-8]) + 7)  # To have the local time UTC+7
            #     mn = f[-12:-10] #[-8:-6] without the additional _A01 at the end of the files # minutes
            #     ms=f[-12:-8] #[-8:-6]#min and sec to compare to minsec
            #     sec=f[-10:-8] #f=
            #     sta=f[-7:-3]
            hour = str(int((f[-10:-8]) + 7)  )# To have the local time UTC+7
            mn = f[-8:-6] #without the additional _A01 at the end of the files # minutes
            ms = f[-8:-4]#min and sec to compare to minsec
            sec = f[-6:-4]  # f=
            #To create a graph with only certain profiles, uncomment (for every 10mn profiles of the 18/06 on the fixed
            # station for example)
            # if int(hour)>=17 : #and any(x==mn for mn in [40,50,00,08,17,29,40,54,04]) :
            #     for m in minsec:
            #         if ms==m :
            #             print('heure', hour)
            #             liste2.append(f)
            #             print('liste2 add', len(liste2), liste2)
            #             print('hour', hour, mn)
            #        rest to indent to have one figure with several profiles corresponding to the double if selection above

            col_list = ["Depth (Meter)", "Temperature (Celsius)", "Conductivity (MicroSiemens per Centimeter)",
            "Salinity (Practical Salinity Scale)", "Density (Kilograms per Cubic Meter)"]
            data = pd.read_csv(f, skiprows=28, usecols=col_list)#lambda x : x > 0 and x <= 27 )#, usecols=col_list)
            df=pd.DataFrame(data, columns=['Depth (Meter)', "Temperature (Celsius)",
                                           "Salinity (Practical Salinity Scale)",
                                           "Density (Kilograms per Cubic Meter)"])
            depth=df['Depth (Meter)']
            T=df["Temperature (Celsius)"]
            S=df["Salinity (Practical Salinity Scale)"]
            D=df["Density (Kilograms per Cubic Meter)"]
            Depth=df["Depth (Meter)"]

            # Calcul de N2
            p = 0  # 1013.25
            SA = gsw.SA_from_SP(S, p, 106.5, 20.5)
            CT = gsw.CT_from_t(SA, T.values, p)
            PT = gsw.pt0_from_t(SA, T.values, p)
            p_n2 = np.arange(0, len(Depth) * 0.001013, 0.001013, dtype=float)
            print('len pn2', len(p_n2))
            # tableau qui va de 0 à la surface à la pression de profondeur donnée par len(Depth) en dm (car avec la ctd imer,
            # une mesure tous les 0.1m, len Depth donne le nombre de mesures si 54 = 5.4m de prof) *0.001013 dbar car on perd
            # un atm tous les 10m : 0.001013 db tous les 0.1m
            # p_n2=p*p_n2
            [N2, p_mid] = gsw.Nsquared(SA, CT, p_n2, 20.5)
            # ATTENTION §§§ len(p_mid) et len(N2)= len de tous le reste -1 ==> PQ ? Rajouter pour éviter décalage
            N2_bis = np.zeros(len(SA), dtype=float)
            N2_bis[0:len(SA) - 1] = N2_bis[0:len(SA) - 1] + N2  # BIDOUILLE pour ajoueter un zero à la fin
            N2_bis = pd.DataFrame(N2_bis, columns=['N2 column'])
            print('N2 ok ', len(N2_bis))

            #print('p', p)
            #print('N2', N2, p_mid)

            # print('SA-S', 'max', np.max(abs(SA-S)))
            # print('CT-T', 'max', np.max(abs(CT-T)))
            # print('PT-T', len(PT-T), 'max', np.max(abs(PT-T)))

            #print(depth.max())
            maxdepth.append(depth.max()) #table of the deepest depth of each measurement
            ssalinity.append(S[1]) #choice of the 2d point not 1st (0) : to defend
            sal.append(S.max())
            tempmax.append(CT.max())
            tempmin.append(CT.min())
            density.append(D.min())
            density.append(D.max())
            N.append(N2.max())
            date.append(hour+':'+mn+":"+sec)

            #diagram TS
            #im = ax.scatter( T ,S, c=Depth, cmap=plt.cm.jet, vmin=0, vmax=15, alpha=0.5)#color='k' )#color=color[i], label=hour+'h'+mn)
            #im2 = ax2.scatter( T ,S, c=Depth, cmap=plt.cm.jet, vmin=0, vmax=15, alpha=0.5)#color='k' )#color=color[i], label=hour+'h'+mn)
            #Diagram with the CT and SA values
            #im = ax.scatter(CT ,SA, c=Depth, cmap=plt.cm.jet, vmin=0, vmax=15, alpha=0.5)#color='k' )#color=color[i], label=hour+'h'+mn)
            #im2 = ax2.scatter(CT ,SA, c=Depth, cmap=plt.cm.jet, vmin=0, vmax=15, alpha=0.5)#color='k' )#color=color[i], label=hour+'h'+mn)

        #plt.show()
        #cbar2 = plt.colorbar(im2, label='depth (m)')  # , ax=ax, ticks=1)
        #fig2.savefig('TS_diagram_'+year+month+d+'_CT_SA.png', format='png')
        #plt.show()
    #for all data
    #cbar = plt.colorbar(im, label='depth (m)')  # , ax=ax, ticks=1)
    #fig.savefig('TS_diagram_all_data_SA_CT.png', format='png')
    #plt.show()

            if colorful:
                colorplot=color[i1][i]
            else :
                colorplot=color[i*(int(nb_station/len_color))]

            ax = axs[0]
            #p1 = ax.plot(CT, -data["Depth (Meter)"], color=color[i], label=hour + 'h' + mn)
            if alldayfig:
                label=d+'/'+month
            else :
                if station==1:
                    if t!='SF1' and t!='SF_24':
                        label=sta
                    else :
                        label = hour + 'h' + mn
                else :
                    label=hour+'h'+mn
            if conservative_param :
                p1 = ax.plot(CT, -data["Depth (Meter)"], color=colorplot, label=label)
            elif conservative_param==0:
                p1 = ax.plot(T, -data["Depth (Meter)"], color=colorplot, label=label )#, label=hour + 'h' + mn)

            ax = axs[1]
            if conservative_param :
                p2 = ax.plot(SA, -data["Depth (Meter)"], color=colorplot, label=label )#label=hour + 'h' + mn)
            elif conservative_param==0:
                p2 = ax.plot(S, -data["Depth (Meter)"], color=colorplot, label=label )#label=hour + 'h' + mn)
            if alldayfig==0:
                plt.legend(handles=p2, ncol=3, fontsize='xx-small', loc='upper right')

            ax = axs[2]
            p3 = ax.plot(N2, -data["Depth (Meter)"][0:-1], color=colorplot )#,label=hour + 'h' + mn)  # N2 has a lengh = other data-1
            # ax = axs[2]
            # p3 = ax.plot(data["Density (Kilograms per Cubic Meter)"], -data["Depth (Meter)"], color=color[i]
            #              ,label=hour + 'h' + mn)  # , label='GFF')
            i = i + 1
            # for single plots
            # plt.show()
            # fig.savefig('TS_adjust_' + d + month + year + '_'+hour+mn+sec+ '.png')

            # if int(hour)>=17 : #and any(x==mn for mn in [40,50,00,08,17,29,40,54,04]) :
            #      for m in minsec:
            #          if ms==m :
            #              ax = axs[0]
            #              p1 = ax.plot(CT, -data["Depth (Meter)"], color=color[i], label=hour+'h'+mn)
            #
            #              ax = axs[1]
            #              p2 = ax.plot(SA, -data["Depth (Meter)"], color=color[i], label=hour+'h'+mn)
            #              plt.legend(ncol=3, fontsize='xx-small', loc='upper right')
            #
            #              ax = axs[2]
            #              p3 =ax.plot(N2 , -data["Depth (Meter)"][0:-1], color=color[i], label=hour+'h'+mn) #N2 has a lengh = other data-1
            #              # ax = axs[2]
            #              # p3 = ax.plot(data["Density (Kilograms per Cubic Meter)"], -data["Depth (Meter)"], color=color[i]
            #              #              ,label=hour + 'h' + mn)  # , label='GFF')
            #              i = i + 1
            #              # for single plots
            #              # plt.show()
            #              # fig.savefig('TS_adjust_' + d + month + year + '_'+hour+mn+sec+ '.png')
        #fig.savefig('CT_SA_N_10mn_profiles'+d+month+year+'.png')
        #plt.show()
        if conservative_param:
            if colorful :
                fig.savefig('CT_SA_N2_'+d+month+year+'.png') #for single plots, station by station
            elif colorful==0:
                fig.savefig('CT_SA_N2_grey_' + d + month + year + '.png')  # for single plots, station by station
        elif conservative_param==0:
            if colorful :
                fig.savefig('T_S_N_'+d+month+year+'.png', format='png')
            elif colorful==0:
                fig.savefig('T_S_N_grey_' + d + month + year + '.png', format='png')

    if alldayfig:
        #LEGEND for the plot of all day
        colors=['blue','red', 'green']
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
        plt.legend(handles=lines, labels=['10/08', '11/08', '12/08'], fontsize='small')
        fig.savefig('T_S_N_allday_surveyaugust_shades.png', format='png')
        #plt.show()

elif transectfig :
    for t in transect:
        i1 = i1 + 1
        rep2 = rep+t+'/colocalise_station/'
        print('rep2', rep2)
        if month == '06':
            print('month june', t)
            t_min=27
            t_max=31
            s_max=31
            N2_min=-10
            N2_max=80
            N2_ecart=20
            Dens_min=990
            Dens_max=1050
            Dens_ecart=25
            d_max= -11 # -18 # 17/07/2023

            if t=='TJ1':
                d='16'
                sta_deb=1
            elif t=='TJ2':
                d='16'
                sta_deb=11
            elif t=='TJ3':
                d = '17'
                sta_deb=18
            elif t=='TJ4':
                d='17'
                sta_deb=29
            elif t=='SFJ' :
                d='18'
                sta_deb=1

        elif month == '08':
            t_min = 28.2
            t_max = 32.5
            s_max = 30
            N2_min = -0.03
            N2_max = 0.3
            N2_ecart=0.1
            Dens_min = 900
            Dens_max = 1500
            Dens_ecart = 100
            d_max = -12 # -18  # 17/07/2023

            if t == 'TA1':
                d = '10'
                sta_deb = 1
            elif t == 'TA2':
                d = '10'
                sta_deb = 6
            elif t == 'SF1':
                d = '10'
                sta_deb = 13
            elif t == 'TA3':
                d = '11'
                sta_deb = 26
            elif t == 'TA4':
                d = '12'
                sta_deb = 41
            elif t == 'SF_24':
                d = '12-13'
                sta_deb = 1

        else :
            print('PROBLEM with transect')
            sys.exit(1)



        # #Figure day by day
        # fig2, ax2 = plt.subplots()  # fonctionne avec plot : share=1,left=3 , right=5, bottom=5 , top=7, wspace=10, hspace=5)
        # fig2.suptitle(d+'/'+month+'/'+year)
        # ax2.set_xlabel('Conservative Temperature (°C)', fontsize=fontsize)
        # ax2.set_ylabel('Absolute Salinity (g/kg)', fontsize=fontsize)
        # ax.set_xlim(28.1, 32.1)  # (27, 31)
        # ax.set_ylim(0.0,32.1)  # (0, 31)
        # ax2.xaxis.set_major_locator(MultipleLocator(1))
        # ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        # ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
        # ax2.yaxis.set_major_locator(MultipleLocator(5))
        # ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        # ax2.yaxis.set_minor_locator(MultipleLocator(1))

        # Add the density lines to the graph
        ################################
        ##gsw use conservative temperature CT and SA absolute salinity :
        ##Sea pressure ? 1013.25 en moyenne, voir si ca fit
        # tempL = np.linspace(28.1,32.1,100)#(27, 33, 100)
        # salL = np.linspace(0, 32.1, 100)
        # Tg, Sg = np.meshgrid(tempL, salL)
        # sigma_theta = gsw.sigma0(Sg, Tg)
        # cs = ax.contour(Tg, Sg, sigma_theta, colors='grey', zorder=1)  # , vmin=0, vmax=25)
        # #cs2 = ax2.contour(Tg, Sg, sigma_theta, colors='grey', zorder=1)  # , vmin=0, vmax=25)
        # cl = plt.clabel(cs, fontsize=10, inline=True, fmt='% .1f')
        # #cl2 = plt.clabel(cs2, fontsize=10, inline=True, fmt='% .1f')
        ##############################

        # #Day by day with several profiles of T S on subplots
        fig, axs = plt.subplots(ncols=4, figsize = (9,5))#fonctionne avec plot:share=1,left=3,right=5,bottom=5,top=7,wspace=10,hspace=5)
        fig.suptitle(t+', '+d + '/' + month + '/' + year)
        ax = axs[0]
        ax.set_xlim(t_min, 29.5 ) # t_max)  # (28.75, 30)  # (25,31)
        ax.set_ylim(d_max, 0)
        # p1 = ax.scatter(CT, -data["Depth (Meter)"],alpha=0.8, marker='x', color='red')#, label='GFF')
        if conservative_param:
            ax.set_xlabel('Conservative Temperature (°C)', fontsize=fontsize)
        elif conservative_param == 0:
            ax.set_xlabel('Temperature (°C)', fontsize=fontsize)
        ax.set_ylabel('Depth (m)', fontsize=fontsize)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.d'))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))

        ax = axs[1]
        ax.set_xlim(-2, 30)
        ax.set_ylim(d_max, 0)
        # p1 = ax.scatter(SA, -data["Depth (Meter)"],alpha=0.8, marker='x', color='blue')#, label='GFF')
        # color=color[i] , label=hour+'h'+mn)#, label='GFF')
        if conservative_param:
            ax.set_xlabel('Absolute Salinity (g/kg)', fontsize=fontsize)
        elif conservative_param == 0:
            ax.set_xlabel('Salinity (PSU)', fontsize=fontsize)
        # ax.set_ylabel('Depth (m)', fontsize=fontsize)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        # ax.legend(loc='lower right')
        if var =='TSDN2':
            ax = axs[2]
            # ax.set_xlim(990, 1020)
            ax.set_xlim(Dens_min, 1025) # Dens_max)
            ax.set_ylim(d_max, 0)
            # p1 = ax.scatter(D, -data["Depth (Meter)"],alpha=0.8, marker='x', color='k')#, label='GFF')
            # color=color[i],label=hour + 'h' + mn)  # , label='GFF')
            # ax.set_xlabel('Density (kg/m\u00b3from matplotlib.lines import Line2D)', fontsize=fontsize)
            # ax.set_xlabel('N\u00b2 ($s{-2}$)', fontsize=fontsize)
            ax.set_xlabel(' Density (g/kg)', fontsize=fontsize)
            # ax.set_ylabel('Depth (m)', fontsize=fontsize)
            ax.xaxis.set_major_locator(MultipleLocator(Dens_ecart))
            ax.xaxis.set_minor_locator(MultipleLocator(Dens_ecart/2))
            # ax.legend(loc='lower right')
            # set the spacing between subplots

            ax = axs[3]
            # ax.set_xlim(990, 1020)
            ax.set_xlim(N2_min, N2_max)
            ax.set_ylim(d_max, 0)
            # p1 = ax.scatter(D, -data["Depth (Meter)"],alpha=0.8, marker='x', color='k')#, label='GFF')
            # color=color[i],label=hour + 'h' + mn)  # , label='GFF')
            # ax.set_xlabel('Density (kg/m\u00b3from matplotlib.lines import Line2D)', fontsize=fontsize)
            # ax.set_xlabel('N\u00b2 ($s{-2}$)', fontsize=fontsize)
            ax.set_xlabel(r'$N^2  (s^{-2})$', fontsize=fontsize)
            # ax.set_ylabel('Depth (m)', fontsize=fontsize)
            ax.xaxis.set_major_locator(MultipleLocator(N2_ecart))
            if month == '08':
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # else :
            #    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(N2_ecart / 2))
            # ax.legend(loc='lower right')
            # set the spacing between subplots

        else :
            ax = axs[2]
            # ax.set_xlim(990, 1020)
            ax.set_xlim(N2_min, N2_max)
            ax.set_ylim(d_max, 0)
            # p1 = ax.scatter(D, -data["Depth (Meter)"],alpha=0.8, marker='x', color='k')#, label='GFF')
            # color=color[i],label=hour + 'h' + mn)  # , label='GFF')
            # ax.set_xlabel('Density (kg/m\u00b3from matplotlib.lines import Line2D)', fontsize=fontsize)
            # ax.set_xlabel('N\u00b2 ($s{-2}$)', fontsize=fontsize)
            ax.set_xlabel(r'$N^2  (s^{-2})$', fontsize=fontsize)
            # ax.set_ylabel('Depth (m)', fontsize=fontsize)
            ax.xaxis.set_major_locator(MultipleLocator(N2_ecart))
            if month == '08':
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # else :
            #    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(N2_ecart / 2))
            # ax.legend(loc='lower right')
            # set the spacing between subplots

        plt.subplots_adjust(left=0.11,
                            bottom=0.1,
                            right=0.95,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)

        i = 0  # to select the color of the table
        liste = glob.glob(rep2 + init + year + month + '*' + suffixe)  # Go and find all the files
        # For one figure to gather the T and S of one day, specially for the fixed station
        # fig, axs = plt.subplots(ncols=2)#fonctionne avec plot:share=1,left=3 , right=5,bottom=5,top=7,wspace=10, hspace=5)
        # fig.suptitle(d+'/'+month+'/'+year)
        ssalinity = []
        date = []
        # print('ssalinity', ssalinity)
        liste2 = []  # list to print only certain files : used for the graph with the 10mn files
        # ind=0
        nb_station=len(sorted(liste))

        if month == '06' and day == '18':
            minsec = ['4006', '5025', '0010', '0853', '1744', '2943', '4041', '5436', '0410',
                      '1322']  # selected min+sec to put on a graph
        for f in sorted(liste):
            # make a graph for every 10 mn from 17h30 to 19h. 17h40, 17h50, 18h00, 18h08, 18h17, 18h29, 18h40, 18h54,  19h04
            print(f)
            if (int(f[-10:-8]) + 7)<24:
                hour = str(int(f[-10:-8]) + 7)  # To have the local time UTC+7
            elif (int(f[-10:-8]) + 7)>=24:
                hour= (int(f[-10:-8]) + 7) % 24
                hour=str(hour)
            mn = f[-8:-6]  # minutes
            ms = f[-8:-4]  # min and sec to compare to minsec
            sec = f[-6:-4]
            # To create a graph with only certain profiles, uncomment (for every 10mn profiles of the 18/06 on the fixed
            # station for example)
            # if int(hour)>=17 : #and any(x==mn for mn in [40,50,00,08,17,29,40,54,04]) :
            #     for m in minsec:
            #         if ms==m :
            #             print('heure', hour)
            #             liste2.append(f)
            #             print('liste2 add', len(liste2), liste2)
            #             print('hour', hour, mn)
            #        rest to indent to have one figure with several profiles corresponding to the double if selection above


            #TO name the station instead of the time as label
            if t == 'SF_24':
                sta = 'AF' + str(sta_deb)
            elif t == 'SFJ':
                sta ='JF'+str(sta_deb)
            else:
                if month =='06':
                    sta='J'+str(sta_deb)
                elif month =='08':
                    sta = 'A' + str(sta_deb)



            col_list = ["Depth (Meter)", "Temperature (Celsius)", "Conductivity (MicroSiemens per Centimeter)",
                        "Salinity (Practical Salinity Scale)", "Density (Kilograms per Cubic Meter)"]
            data = pd.read_csv(f, skiprows=28, usecols=col_list)  # lambda x : x > 0 and x <= 27 )#, usecols=col_list)
            df = pd.DataFrame(data, columns=['Depth (Meter)', "Temperature (Celsius)",
                                             "Salinity (Practical Salinity Scale)",
                                             "Density (Kilograms per Cubic Meter)"])
            depth = df['Depth (Meter)']
            T = df["Temperature (Celsius)"]
            S = df["Salinity (Practical Salinity Scale)"]
            D = df["Density (Kilograms per Cubic Meter)"]
            Depth = df["Depth (Meter)"]
            print('Density', len(D), D)
            print('Depth', len(Depth), Depth)

            # Calcul de N2
            p = 0  # 1013.25
            SA = gsw.SA_from_SP(S, p, 106.5, 20.5)
            CT = gsw.CT_from_t(SA, T.values, p)
            PT = gsw.pt0_from_t(SA, T.values, p)
            p_n2 = np.arange(0, len(Depth) * 0.001013, 0.001013, dtype=float)
            print('len pn2', len(p_n2))
            # tableau qui va de 0 à la surface à la pression de profondeur donnée par len(Depth) en dm (car avec la ctd imer,
            # une mesure tous les 0.1m, len Depth donne le nombre de mesures si 54 = 5.4m de prof) *0.001013 dbar car on perd
            # un atm tous les 10m : 0.001013 db tous les 0.1m
            # p_n2=p*p_n2
            [N2, p_mid] = gsw.Nsquared(SA, CT, p_n2, 20.5)
            # ATTENTION §§§ len(p_mid) et len(N2)= len de tous le reste -1 ==> PQ ? Rajouter pour éviter décalage
            N2_bis = np.zeros(len(SA), dtype=float)
            N2_bis[0:len(SA) - 1] = N2_bis[0:len(SA) - 1] + N2  # BIDOUILLE pour ajoueter un zero à la fin
            N2_bis = pd.DataFrame(N2_bis, columns=['N2 column'])
            print('N2 ok ', len(N2_bis))

            #p = 0  # 1013.25
            #SA = gsw.SA_from_SP(S, p, 106.5, 20.5)
            #print('SA', SA)
            #CT = gsw.CT_from_t(SA, T, p)
            #print('CT', CT)
            #PT = gsw.pt0_from_t(SA, T, p)
            #[N2, p_mid] = gsw.Nsquared(SA, CT, p, 20.5)

            # print('p', p)
            # print('N2', N2, p_mid)

            # print('SA-S', 'max', np.max(abs(SA-S)))
            # print('CT-T', 'max', np.max(abs(CT-T)))
            # print('PT-T', len(PT-T), 'max', np.max(abs(PT-T)))

            # print(depth.max())
            maxdepth.append(depth.max())  # table of the deepest depth of each measurement
            ssalinity.append(S[1])  # choice of the 2d point not 1st (0) : to defend
            sal.append(S.max())
            tempmax.append(CT.max())
            tempmin.append(CT.min())
            density.append(D.min())
            density.append(D.max())
            N.append(N2.max())
            date.append(hour + ':' + mn + ":" + sec)

            if diagramTS:
                im = ax.scatter( T ,S, c=Depth, cmap=plt.cm.jet, vmin=0, vmax=15, alpha=0.5)#color='k' )#color=color[i], label=hour+'h'+mn)
                im2 = ax2.scatter( T ,S, c=Depth, cmap=plt.cm.jet, vmin=0, vmax=15, alpha=0.5)#color='k' )#color=color[i], label=hour+'h'+mn)
                #Diagram with the CT and SA values
                im = ax.scatter(CT ,SA, c=Depth, cmap=plt.cm.jet, vmin=0, vmax=15, alpha=0.5)#color='k' )#color=color[i], label=hour+'h'+mn)
                im2 = ax2.scatter(CT ,SA, c=Depth, cmap=plt.cm.jet, vmin=0, vmax=15, alpha=0.5)#color='k' )#color=color[i], label=hour+'h'+mn)

                #plt.show()
                if not diagramTS1fig:
                    cbar2 = plt.colorbar(im2, label='depth (m)')  # , ax=ax, ticks=1)
                    fig2.savefig('TS_diagram_'+year+month+d+'_CT_SA.png', format='png')
                    #plt.show()
                elif diagramTS1fig:
                    cbar = plt.colorbar(im, label='depth (m)')  # , ax=ax, ticks=1)
                    fig.savefig('TS_diagram_all_data_SA_CT'+month+year+'.png', format='png')
                    #plt.show()
                

            if colorful:
                colorplot=color[i1][i]
            else :
                colorplot=color[i*(int(len_color/nb_station))]
            ax = axs[0]
            # p1 = ax.plot(CT, -data["Depth (Meter)"], color=color[i], label=hour + 'h' + mn)
            if alldayfig:
                label = d + '/' + month
            else:
                if station==1:
                    if t!='SF1' and t!='SF_24' and t!='SFJ':
                        label=sta
                    else :
                        label = hour + 'h' + mn
                else :
                    label = hour + 'h' + mn
            if conservative_param:
                p1 = ax.plot(CT, -data["Depth (Meter)"], color=colorplot, label=label)
            elif conservative_param == 0:
                p1 = ax.plot(T, -data["Depth (Meter)"], color=colorplot, label=label)

            ax = axs[1]
            if conservative_param:
                p2 = ax.plot(SA, -data["Depth (Meter)"], color=colorplot, label=label)
            elif conservative_param == 0:
                p2 = ax.plot(S, -data["Depth (Meter)"], color=colorplot, label=label)
            #if alldayfig==0:

            if var=='TSDN2':
                ax = axs[2]
                p3 = ax.plot(D, -data["Depth (Meter)"],
                             color=colorplot, label=label)  # N2 has a lengh = other data-1

                ax = axs[3]
                p3 = ax.plot(N2, -data["Depth (Meter)"][0:-1],
                             color=colorplot, label=label)  # N2 has a lengh = other data-1
                ax.legend(ncol=1, fontsize='xx-small', loc='upper right')
            else :
                ax = axs[2]
                p3 = ax.plot(N2, -data["Depth (Meter)"][0:-1],
                             color=colorplot, label=label)# N2 has a lengh = other data-1
                ax.legend(ncol=1, fontsize='xx-small', loc='upper right')

            sta_deb=sta_deb+1
            # ax = axs[2]
            # p3 = ax.plot(data["Density (Kilograms per Cubic Meter)"], -data["Depth (Meter)"], color=color[i]
            #              ,label=hour + 'h' + mn)  # , label='GFF')
            i = i + 1
            # for single plots
            # plt.show()
            # fig.savefig('TS_adjust_' + d + month + year + '_'+hour+mn+sec+ '.png')

            # if int(hour)>=17 : #and any(x==mn for mn in [40,50,00,08,17,29,40,54,04]) :
            #      for m in minsec:
            #          if ms==m :
            #              ax = axs[0]
            #              p1 = ax.plot(CT, -data["Depth (Meter)"], color=color[i], label=hour+'h'+mn)
            #
            #              ax = axs[1]
            #              p2 = ax.plot(SA, -data["Depth (Meter)"], color=color[i], label=hour+'h'+mn)
            #              plt.legend(ncol=3, fontsize='xx-small', loc='upper right')
            #
            #              ax = axs[2]
            #              p3 =ax.plot(N2 , -data["Depth (Meter)"][0:-1], color=color[i], label=hour+'h'+mn) #N2 has a lengh = other data-1
            #              # ax = axs[2]
            #              # p3 = ax.plot(data["Density (Kilograms per Cubic Meter)"], -data["Depth (Meter)"], color=color[i]
            #              #              ,label=hour + 'h' + mn)  # , label='GFF')
            #              i = i + 1
            #              # for single plots
            #              # plt.show()
            #              # fig.savefig('TS_adjust_' + d + month + year + '_'+hour+mn+sec+ '.png')
        # fig.savefig('CT_SA_N_10mn_profiles'+d+month+year+'.png')
        #plt.show()
        if alldayfig==0:
            if conservative_param:
                if var=='TSDN2':
                    outfile='CT_SA_D_N2_'
                else :
                    outfile='CT_SA_N2_'
            elif conservative_param==0:
                if var =='TSDN2':
                    outfile='T_S_D_N2_'
                else :
                    outfile='T_S_N2_'
            if colorful :
                outfile= outfile+'colored_'
            elif colorful==0:
                outfile=outfile+'grey_'
            outfile=outfile+t+'_'+ d + month + year + '.png'
            fig.savefig(outfile, format='png')
            print('fig saved at ', outfile)


    if alldayfig:
        # LEGEND for the plot of all day
        colors = ['blue', 'red', 'green']
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
        plt.legend(handles=lines, labels=['10/08', '11/08', '12/08'], fontsize='small')
        fig.savefig('T_S_N2_survey'+month+'_'+t+'.png', format='png')
        #plt.show()

#         sys.exit(1)
#         #figure TS
#         #Figure for one fig per station
#         # fig, axs = plt.subplots(ncols=3)# fonctionne avec plot : share=1,left=3 , right=5, bottom=5 , top=7, wspace=10, hspace=5)
#         # fig.suptitle(d+'/'+month+'/'+year+' '+hour+'h'+mn)
#         ax = axs[0]
#         ax.set_xlim(28,32.1) #(25,31)
#         ax.set_ylim(-16.5, 0)
#         #p1 = ax.scatter(CT, -data["Depth (Meter)"],alpha=0.8, marker='x', color='red')#, label='GFF')
#         p1 = ax.plot(CT, -data["Depth (Meter)"], color='red' )#color=color[i], label=hour+'h'+mn)
#         ax.set_xlabel('Conservative Temperature (°C)', fontsize=fontsize)
#         ax.set_ylabel('Depth (m)', fontsize=fontsize)
#         ax.xaxis.set_major_locator(MultipleLocator(1))
#         ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#         ax.xaxis.set_minor_locator(MultipleLocator(0.5))
#
#         ax = axs[1]
#         ax.set_xlim(0,31.5)
#         ax.set_ylim(-16.5, 0)
#         #p1 = ax.scatter(SA, -data["Depth (Meter)"],alpha=0.8, marker='x', color='blue')#, label='GFF')
#         p2 = ax.plot(SA, -data["Depth (Meter)"], color='blue' )
#         #color=color[i] , label=hour+'h'+mn)#, label='GFF')
#         ax.set_xlabel('Absolute Salinity (g/kg)', fontsize=fontsize)
#         #ax.set_ylabel('Depth (m)', fontsize=fontsize)
#         ax.xaxis.set_major_locator(MultipleLocator(10))
#         ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#         ax.xaxis.set_minor_locator(MultipleLocator(5))
#         # ax.legend(loc='lower right')
#
#         ax = axs[2]
#         ax.set_xlim(990,1020)
#         ax.set_ylim(-16.5, 0)
#         #p1 = ax.scatter(D, -data["Depth (Meter)"],alpha=0.8, marker='x', color='k')#, label='GFF')
#         p3 = ax.plot(data["Density (Kilograms per Cubic Meter)"], -data["Depth (Meter)"], color='k')
#         #color=color[i],label=hour + 'h' + mn)  # , label='GFF')
#         ax.set_xlabel('Density (kg/m\u00b3)', fontsize=fontsize)
#         #ax.set_ylabel('Depth (m)', fontsize=fontsize)
#         ax.xaxis.set_major_locator(MultipleLocator(10))
#         ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#         ax.xaxis.set_minor_locator(MultipleLocator(5))
#         # ax.legend(loc='lower right')
#         # set the spacing between subplots
#
#         plt.subplots_adjust(left=0.11,
#                         bottom=0.1,
#                         right=0.95,
#                         top=0.9,
#                         wspace=0.4,
#                         hspace=0.4)
#
#         #plt.legend(ncol=3, fontsize='xx-small')
#         i=i+1
#         #for single plots
#         #plt.show()
#         #fig.savefig('TS_adjust_' + d + month + year + '_'+hour+mn+sec+ '.png')
#
#         plt.show()
#         #fig.savefig('CT_SA_D_'+d+month+year+'_'+hour+mn+sec+'.png') #for single plots, station by station
#
# #
#         # Extraire données de surface : moyennes des premiers points sur les 2m ? Juste les 2 premiers points ? Le min des 2/3 premiers points ?
#         #print('salinity', type(ssalinity), len(ssalinity), ssalinity)
#         #record all the salinity of the 2d point of salinity profiles
#         #file='Surface_salinity_' + d + month + year
#         #print('f', file)
#         #l=[date,ssalinity]
#         #export_data = zip_longest(*l, fillvalue='')
#         #with open(file+'.csv', 'w', encoding="ISO-8859-1", newline='') as f:
#         #    pickle.dump(ssalinity, f)
#         #    write = csv.writer(f)
#         #    write.writerow(('Time', 'Surface Salinity'))
#         #    write.writerows(export_data)

# print('max depth' , np.nanmax(maxdepth)) #give the maximum depth over the whole period
# print('min depth' , np.nanmin(maxdepth)) #give the min depth over the whole period
# print('N max', np.nanmin(N), np.nanmax(N))
# print('T' , np.nanmin(tempmin), np.nanmax(tempmax))
# print('S' , np.nanmin(sal), np.nanmax(sal))
# print('density', np.nanmin(density), np.nanmax(density))