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

#2 versions : day by day or all data

#CC1238004_20220604_050128
rep='/home/penicaud/Documents/Data/CTD/Survey_june_CTD/'
init= 'CC1238004_'
year='2022'
month= '06'
day= ['18'] # ['16','17','18']
suffixe = '.csv'

fontsize=13
maxdepth=[] #list of all the max depth of each profil
ssalinity=[] #surface salinity (choose the 1st or 2d point of each profil
date=[] #record the date of the CTD profil
tempmax, tempmin, sal, density =[],[],[],[]

cmap=plt.cm.binary #grey
#cmap=plt.cm.hsv_r
#15 ==> 90
# extracting all colors
color = [cmap(i) for i in range(cmap.N)]
color=color[15::10] #good for the fig with all the profiles of T and S of the 18/06 in shades of grey
# #because the colors go from white to black, choose only from a certain phase and with a step so that we can see some differences
#color = color[30::25] #good for hsv_r from purple - green -red on 6 curves

# making first color entry  grey
#color[0] = (.5, .5, .5, 1.0)


#number_of_colors = 100
#color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
#             for i in range(number_of_colors)]
#FIND THE MISSING LINES
#To do one time, if the colors are suitable, in order to always have the same colors between graphs
#with open('list_100colors.pkl', 'wb') as f:
#    pickle.dump(color, f)
#print(color)

#To load the color list file
#with open('list_100colors.pkl', 'rb') as f:
#    color = pickle.load(f)

# #diagram TS
fig, ax = plt.subplots()  # fonctionne avec plot : share=1,left=3 , right=5, bottom=5 , top=7, wspace=10, hspace=5)
fig.suptitle('All data')
ax.set_xlabel('Conservative Temperature (°C)', fontsize=fontsize)
ax.set_ylabel('Absolute Salinity (g/kg)', fontsize=fontsize)
ax.set_xlim(28,32.1)#(27, 31)
ax.set_ylim(0,32.1)#(0, 31)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.yaxis.set_minor_locator(MultipleLocator(1))
#
# # Add the density lines to the graph
# ################################
# # gsw use conservative temperature CT and SA absolute salinity :
# # Sea pressure ? 1013.25 en moyenne, voir si ca fit
# tempL = np.linspace(27, 33, 100)
# salL = np.linspace(0, 33, 100)
# Tg, Sg = np.meshgrid(tempL, salL)
# sigma_theta = gsw.sigma0(Sg, Tg)
# cs = ax.contour(Tg, Sg, sigma_theta, colors='grey', zorder=1)  # , vmin=0, vmax=25)
# cl = plt.clabel(cs, fontsize=10, inline=True, fmt='% .1f')
# ###############################

for d in day:
    #Figure day by day
    fig2, ax2 = plt.subplots()  # fonctionne avec plot : share=1,left=3 , right=5, bottom=5 , top=7, wspace=10, hspace=5)
    fig2.suptitle(d+'/'+month+'/'+year)
    ax2.set_xlabel('Temperature (°C)', fontsize=fontsize)
    ax2.set_ylabel('Salinity (PS)', fontsize=fontsize)
    ax.set_xlim(28, 32.1)  # (27, 31)
    ax.set_ylim(0, 32.1)  # (0, 31)
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))

    # Add the density lines to the graph
    ################################
    ##gsw use conservative temperature CT and SA absolute salinity :
    ##Sea pressure ? 1013.25 en moyenne, voir si ca fit
    tempL = np.linspace(27, 33, 100)
    salL = np.linspace(0, 33, 100)
    Tg, Sg = np.meshgrid(tempL, salL)
    sigma_theta = gsw.sigma0(Sg, Tg)
    cs = ax.contour(Tg, Sg, sigma_theta, colors='grey', zorder=1)  # , vmin=0, vmax=25)
    cs2 = ax2.contour(Tg, Sg, sigma_theta, colors='grey', zorder=1)  # , vmin=0, vmax=25)
    cl = plt.clabel(cs, fontsize=10, inline=True, fmt='% .1f')
    cl2 = plt.clabel(cs2, fontsize=10, inline=True, fmt='% .1f')
    ##############################

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
    minsec=['4006','5025','0010','0853','1744','2943','4041','5436','0410','1322'] #selected min+sec to put on a graph
    for f in sorted(liste):
        #make a graph for every 10 mn from 17h30 to 19h. 17h40, 17h50, 18h00, 18h08, 18h17, 18h29, 18h40, 18h54,  19h04
        #print(f)
        hour = str(int(f[-10:-8]) + 7)  # To have the local time UTC+7
        mn = f[-8:-6]  # minutes
        ms=f[-8:-4] #min and sec to compare to minsec
        sec=f[-6:-4]
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

        p=0#1013.25
        SA=gsw.SA_from_SP(S,p, 106.5, 20.5)
        CT=gsw.CT_from_t(SA,T,p)
        PT=gsw.pt0_from_t(SA, T, p)



        print('SA-S', 'max', np.max(abs(SA-S)))
        print('CT-T', 'max', np.max(abs(CT-T)))
        print('PT-T', len(PT-T), 'max', np.max(abs(PT-T)))
        #sys.exit(1)

        #print(depth.max())
        maxdepth.append(depth.max()) #table of the deepest depth of each measurement
        ssalinity.append(S[1]) #choice of the 2d point not 1st (0) : to defend
        sal.append(S.max())
        tempmax.append(T.max())
        tempmin.append(T.min())
        density.append(D.min())
        density.append(D.max())
        date.append(hour+':'+mn+":"+sec)


        #diagram TS
        #im = ax.scatter( T ,S, c=Depth, cmap=plt.cm.jet, vmin=0, vmax=15, alpha=0.5)#color='k' )#color=color[i], label=hour+'h'+mn)
        #im2 = ax2.scatter( T ,S, c=Depth, cmap=plt.cm.jet, vmin=0, vmax=15, alpha=0.5)#color='k' )#color=color[i], label=hour+'h'+mn)
        #Diagram with the CT and SA values
        im = ax.scatter(CT ,SA, c=Depth, cmap=plt.cm.jet, vmin=0, vmax=15, alpha=0.5)#color='k' )#color=color[i], label=hour+'h'+mn)
        im2 = ax2.scatter(CT ,SA, c=Depth, cmap=plt.cm.jet, vmin=0, vmax=15, alpha=0.5)#color='k' )#color=color[i], label=hour+'h'+mn)

    #plt.show()
    cbar2 = plt.colorbar(im2, label='depth (m)')  # , ax=ax, ticks=1)
    fig2.savefig('TS_diagram_'+year+month+d+'_CT_SA.png', format='png')
    plt.show()
#for all data
#cbar= plt.colorbar(im, label='depth (m)')#, ax=ax, ticks=1)
#fig.savefig('TS_diagram_all_data_SA_CT.png', format='png')
plt.show()

#         #figure TS
#         #Figure for one fig per station
#         fig, axs = plt.subplots(ncols=3)# fonctionne avec plot : share=1,left=3 , right=5, bottom=5 , top=7, wspace=10, hspace=5)
#         fig.suptitle(d+'/'+month+'/'+year+' '+hour+'h'+mn)
#         ax = axs[0]
#         ax.set_xlim(25,31)
#         ax.set_ylim(-16.5, 0)
#         p1 = ax.scatter(data["Temperature (Celsius)"], -data["Depth (Meter)"],alpha=0.8, marker='x', color='red')#, label='GFF')
#         p1 = ax.plot( data["Temperature (Celsius)"], -data["Depth (Meter)"], color='red' )#color=color[i], label=hour+'h'+mn)
#         ax.set_xlabel('Temperature (°C)', fontsize=fontsize)
#         ax.set_ylabel('Depth (m)', fontsize=fontsize)
#         ax.xaxis.set_major_locator(MultipleLocator(1))
#         ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#         ax.xaxis.set_minor_locator(MultipleLocator(0.5))
#
#         ax = axs[1]
#         ax.set_xlim(0,30.5)
#         ax.set_ylim(-16.5, 0)
#         p1 = ax.scatter(data["Salinity (Practical Salinity Scale)"], -data["Depth (Meter)"],alpha=0.8, marker='x', color='blue')#, label='GFF')
#         p2 = ax.plot( data["Salinity (Practical Salinity Scale)"], -data["Depth (Meter)"], color='blue' )
#         #color=color[i] , label=hour+'h'+mn)#, label='GFF')
#         ax.set_xlabel('Salinity (PS)', fontsize=fontsize)
#         #ax.set_ylabel('Depth (m)', fontsize=fontsize)
#         ax.xaxis.set_major_locator(MultipleLocator(10))
#         ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#         ax.xaxis.set_minor_locator(MultipleLocator(5))
#         # ax.legend(loc='lower right')
#
#         ax = axs[2]
#         ax.set_xlim(990,1020)
#         ax.set_ylim(-16.5, 0)
#         p1 = ax.scatter(D, -data["Depth (Meter)"],alpha=0.8, marker='x', color='k')#, label='GFF')
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
#         #plt.show()
#         #fig.savefig('TS_'+d+month+year+'.png') #for single plots, station by station
#
#
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
#
# print('max depth' , np.nanmax(maxdepth)) #give the maximum depth over the whole period
# print('min depth' , np.nanmin(maxdepth)) #give the min depth over the whole period
# print('T' , np.nanmin(tempmin), np.nanmax(tempmax))
# print('S' , np.nanmin(sal), np.nanmax(sal))
#print('density', np.nanmin(density), np.nanmax(density))
