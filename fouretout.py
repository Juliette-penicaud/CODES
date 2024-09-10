#18/07/2022 Check
import csv
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression, TheilSenRegressor
#from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
#import csv
import sys, os, glob
#import xarray as xr
#import matplotlib.colors as mcolors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import gsw
import random
import pickle
from itertools import zip_longest



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

#COULEURS : crée une palette de gris ou autres à partir de colormaps
cmap=plt.cm.binary #grey
#cmap=plt.cm.hsv_r
# extracting all colors
color = [cmap(i) for i in range(cmap.N)]
color=color[15::10] #good for the fig with all the profiles of T and S of the 18/06 in shades of grey
# #because the colors go from white to black, choose only from a certain phase and with a step so that we can see some differences
#color = color[30::25] #good for hsv_r from purple - green -red on 6 curves

#couleurs : create a table of random colors and save it useful to re use
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

#Pour avoir l'heure /100 et non pas /60, ATTENTION, le calcul de T2 donne le pourcentage de marée, regarder juste les mn2
for t in time:
    print('t', t)
    if t>=6 and t<18:
        mn=t-int(t) #only mn part left
        mn2=mn*100/60
        t2=(int(t)+mn2-6)/12 *100 #HT=100%, LT=0%
        #t2="{:.2f}".format(t2)
        t2=round(t2,2)
    elif t==18.0:
        t2=100.00
    elif t==6.0:
        t2=0.00
    elif t>=18 or t<6 :
        if t<=23 and t>18:
            mn = t - int(t)  # only mn part left
            mn2 = mn * 100 / 60
            t2=-(int(t)+mn2-18)/12 *100
            t2 = round(t2, 2)

        else:
            print('special t', t)
            mn = t - int(t)  # only mn part left
            mn2 = mn * 100 / 60
            t2=-(24+int(t)+mn2-18)/12 *100
            #t2 = "{:.2f}".format(t2)
            t2 = round(t2, 2)

            #convention: negative, LT--> -100% but LT and HT are calculated in the positive convention
    else:
        print('PROBLEM to define t')

    print('t2' , t2, '%')
    percentage.append(t2)




#Select the index of the line corresponding to a several condition on different columns of df and insert a new column in df
type_hour = ['hour_LOG', 'hour_IMER']
for t_hour in type_hour:
    print('type hour', type_hour)
    if t_hour == 'hour_LOG':
        name_num = 'num_LOG'
    elif t_hour == 'hour_IMER':
        name_num = 'num_IMER'
    num = []
    for h in df2[t_hour]:
        hour_df2 = int(h[0:2])
        mn_df2 = int(h[3:5])
        sec_df2 = 0
        # hour_df2=math.trunc(h)
        # mn_df2=((h-int(h))*100)
        # sec_df2=0
        # print('hour', h,  h-int(h), hour_df2, mn_df2)
        # mn_df2=math.trunc((np.around(h,2)-math.trunc(h))*100)
        print('hour', h, hour_df2, mn_df2)
        condition = ((df.HH == hour_df2) & (df.MM == mn_df2) & (df.SS == sec_df2))
        l = df.index[condition]
        l = l[0] + 1  # +1 because it gives the index and indexing begins by 0
        print(l)
        num.append(l)

    df2[name_num] = num
    print(df2)

#Pandas, select only some rows of a dataframe for a new df
station_to_plot = ['S9', 'S10', 'S12', 'S15', 'S17', 'S19', 'S21', 'S23']
data_to_plot = df2[(df2['stations'].isin(station_to_plot))]
print('data to plot', data_to_plot)



############################### find last non nan value of an array
last_val=(~np.isnan(vx)).cumsum(1).argmax(1) #last value non nan : to remove. WORKS ONLY FOR VX, shape [[]]
lasy_val = np.argwhere(np.isnan(vxmoy))[0] #Ok for simple [] array


# ordonne une liste selon les valeurs entiere en skppant char : S1 S12 S3 S4 S34 S5 devient S1 S3 S4 S5 S12 S35
list_sheet_station = sorted(list_sheet_station, key=lambda s: int(re.search(r'\d+', s).group()))
