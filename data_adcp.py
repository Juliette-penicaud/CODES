#27/07/2022 Treat ADCP DATA
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cmcrameri as cmc
import scipy.signal as signal
import math
#import matplotlib
#matplotlib.use('TkAgg')
#from adcploader import *
from mpl_toolkits import mplot3d

# 28/09/23 : addition of other parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
filter = True
horizontal_line = False
plot_two = 'direction' # direction or profiles
unit = 'm/s'
#LIST OF variables : station, station2 same but with '' inbetween to plot on the 2d graph, profilevertical : the number of the rows of the table

#Compare averages of the profiles (no avg, 5sec, 10 sec, 15sec)
day='18'
month='06'
year='2022'
if month=='06':
    survey='Survey_June'
elif month=='08':
    survey='Survey_June'
elif month=='10':
    survey='Survey_Octobre'
else :
    print('PROBLEME SURVEY')
    sys.exit(1)
file='/home/penicaud/Documents/Data/ADCP/'+survey+'/'+day+month+'_alldata_BT.csv'
i=1
col_list = ["HH", "MM", "SS",]

for i in range(1,32):
    string="Mag, mm/s, "+str(i)
    col_list.append(string)
for i in range(1,32):
    string="Dir, deg, "+str(i)
    col_list.append(string)
for i in range(1,32):
    string="Eas, mm/s, "+str(i)
    col_list.append(string)
for i in range(1,32):
    string="Nor, mm/s, "+str(i)
    col_list.append(string)

print('file ', file)
#data = pd.read_csv(file, skiprows=11, low_memory=False, sep=',', usecols=col_list) #if it is the file no BT (bottom track)
data = pd.read_csv(file, skiprows=11, low_memory=False, sep=' ', usecols=col_list) #sep = ' ' if it is the file with BT
df = pd.DataFrame(data)

#print(data)
hour = df["HH"]
mn = df["MM"]
sec = df["SS"]

mag,dir=[],[]
for i in range(1,32):
    string1="Mag, mm/s, "+str(i)
    string2="Dir, deg, "+str(i)
    mag.append(string1)
    dir.append(string2)


magnitude=pd.DataFrame(data, columns=mag)
magnitude2=magnitude.transpose()
magnitude2 = magnitude2.iloc[::-1]
direction=pd.DataFrame(data, columns=dir)
direction2=direction.transpose()
direction2 = direction2.iloc[::-1]

cmap=plt.cm.jet
cmap = cmc.cm.hawaii_r
cmap2=plt.cm.plasma
fontsize=12

lim1=7#nb of the limit bin between layer 1 and 2
lim2=17# nb of the limit bin between layer 2 and 3 #17 contain every intense bins, 15 around 80 % (a vue de nez)
dim='2d' #or '3d'
sta=''#S9-S24'#'random' or 'S9-S24'
moy=180 #EVEN NUMBER #nb of seconds and of profiles we want to average
CTD_to_observe='LOG' #'LOG' or 'IMER'

var='all data'
#X axis
if var=='all data' :
    x=np.linspace(0,22948,6, dtype=int)
    fixe3 = 1850
    fixe1 = 6885
    fixe2 = 16064
    fixedir1 = 12362
    fixedir2 = 7742
elif var=='salt':
    if sta=='random':
        x=np.linspace(16064, 22948, 6, dtype=int) #from 17h39
        # Determine the new values of x2, for the profiles A B C D ..
        x2 = np.copy(x)
        x2[0] = 16382
        x2[-1] = 22382
        print('x2', x2)
    elif sta=='S9-S24':
        x=np.linspace(15182, 22948, 6, dtype=int) #from 17h25,
    else :
        x = [0]
else :
    print('Problem')

time=[]
for xi in x :
    if hour[xi]<10 :
        h='0'+str(hour[xi])
    else :
        h=str(hour[xi])
    if mn[xi]<10 :
        m = '0' + str(mn[xi])
    else :
        m =str(mn[xi])
    if sec[xi]<10:
        s = '0' + str(sec[xi])
    else:
        s= str(sec[xi])
    time.append(h+':'+m+':'+s)
    #print('hour ', hour[xi], min[xi], sec[xi])
print('time', time)

#Axis y
y = np.linspace(0, 31, 3)
print('y', y)
depth=np.around(np.arange(-10.2,-0.3,0.3), decimals=1)
print(depth)
print('depth', depth[::16])
#yplot = np.linspace(0,32, 9)
yplot=np.arange(1,32,1)
print('yplot', yplot)

#parameter for figures
n=1 #indicator for quiver plot legend
labelcolor='black'
incr=0

#TODO : attention on peut etre ne 2D avec des stations
if dim=='2d':
    fig,axs = plt.subplots(nrows=2, sharex=True,  constrained_layout=True)
    ax = axs[0]
    posx=[0.09, 0.25, 0.45, 0.65, 0.85 , 0.97] #position for the 2d graph for the legend of the averaged layer
    posy=[0.05, 0.28, 0.65]
elif dim=='3d':
    fig=plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    if sta=='random':
        posx3dgreen=np.arange(0.05,3.4,0.6)
        posx3dblue=np.arange(-0.4,3.4,0.6)
        posx3dred=np.arange(-0.53,3.4,0.6)
        station = ['A', 'B', 'C', 'D', 'E', 'F']
        station2=station.copy() #to avoid changing list staiton
        station2.insert(0,'')
        i=1
        while i < 2*len(station):
            station2.insert(i, '')
            i += 2    #station2.insert(0::2, '')
        profilevertical=x2 #determine the x axis for the 1st plot
    elif sta=='S9-S24':
        print('S9-S24')
        df2= pd.DataFrame({'stations' : ['S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15','S16', 'S17', 'S18',
                                       'S19', 'S20', 'S21', 'S22', 'S23', 'S24'] ,
                          'hour_LOG' :['17.30','17.54','18.00','18.05','18.08','18.13','18.17','18.26','18.29','18.40',
                                       '18.46','18.54','19.04','19.13','19.20','19.29'],
                          'hour_IMER': ['17.31','17.53','17.58','18.03','18.07','18.12','18.18','18.25','18.30','18.39',
                                        '18.45','18.51','18.57','19.08','19.21','19.31']
                          })
        #TODO : retrouver les numeros des ensembles grace aux horaires. check where ==HH and MM ==

        type_hour= ['hour_LOG', 'hour_IMER']
        for t_hour in type_hour:
            print('type hour', type_hour)
            if t_hour=='hour_LOG':
                name_num='num_LOG'
            elif t_hour=='hour_IMER':
                name_num='num_IMER'
            num = []
            for h in df2[t_hour]:
                hour_df2=int(h[0:2])
                mn_df2=int(h[3:5])
                sec_df2=0
                # hour_df2=math.trunc(h)
                # mn_df2=((h-int(h))*100)
                # sec_df2=0
                # print('hour', h,  h-int(h), hour_df2, mn_df2)
                # mn_df2=math.trunc((np.around(h,2)-math.trunc(h))*100)
                condition = ((df.HH==hour_df2) & (df.MM==mn_df2) & (df.SS==sec_df2))
                l=df.index[condition]
                l=l[0]+1 #+1 because it gives the index and indexing begins by 0
                #print('hour', h, hour_df2, mn_df2, l)
                num.append(l)
            df2[name_num]=num
        print(df2)
        station=['S9', 'S10', 'S12', 'S15', 'S17', 'S19', 'S21', 'S23', 'S24']
        station2=station.copy()
        station2.insert(0,'')
        i=1
        while i < 2*len(station):
            station2.insert(i, '')
            i += 2
        data_to_plot=df2[(df2['stations'].isin(station))]
        print('data to plot', data_to_plot)
        if CTD_to_observe=='LOG':
            profilevertical=data_to_plot['num_LOG']
        elif CTD_to_observe=='IMER':
            profilevertical=data_to_plot['num_IMER']
        print('profile vertical', CTD_to_observe, profilevertical)
    else :
        print('No station printed')

#1st plot of the magnitude of current
fig.suptitle('Current magnitude and velocity profiles')
disp='Magnitude'
if unit == 'm/s' :
    disp = disp + ' (m/s)'
else :
    disp = disp + ' (mm/s)'
ax.set_ylabel('Depth (m)', fontsize=fontsize-2)
ax.set_ylim(0, 31)
ax.set_yticks(y)
ax.set_yticklabels(labels=depth[::16], fontsize=fontsize-2)
#ax.set_xlabel('Time', fontsize=fontsize)
ax.set_xlim(np.min(x),np.max(x))
ax.set_xticks(x)
ax.set_xticklabels(labels=time, fontsize=fontsize-4)
#ax.set_xlim(0, np.max(x) + 1)
#ax.set_xticks(ticks=x)  # , labels=stations)
#ax.set_xticklabels(stations)  # , minor=True)
if filter:
    variable = signal.medfilt2d(magnitude2, 5)  # calcul de la turbidité avec filter median 5
else:
    variable = magnitude2
if unit == 'm/s':
    variable = variable / 1000
vmax = 1000
if unit == 'm/s':
    vmax = vmax / 1000
p1=ax.pcolormesh(variable, cmap=cmap, vmin=0, vmax=vmax)
# linev3=ax.axvline(fixe3,color='k')
# #lineh3=ax.axhline(20, xmin=0, xmax=0.1, color='k')
# linev1=ax.axvline(fixe1, color='k')
# linev2=ax.axvline(fixe2, color='k')
if horizontal_line :
    lineh2=ax.axhline(lim1,xmin=0, xmax=1, color='k') #17 contain every intense bins, 15 around 80 % (a vue de nez)
    lineh3=ax.axhline(lim2, xmin=0, xmax=1, color='k')
cbar = plt.colorbar(p1, label=disp, ax=ax)  # , ticks=1)#ax=ax
cbar.ax.tick_params(labelsize=8)

if sta=='S9-S24' or sta == 'random' :
    #Plot the vertical profiles only
    xver=0 #increment only for the list of "stations"
    for xvertical in profilevertical:
        s = station[xver]
        ax.axvline(xvertical, color='black')
        ax.text(xvertical-5, y=np.max(yplot)+1, s=s, color='k', fontsize=fontsize-2)
        xver=xver+1




#Parameters for the 2d plot
limvel=0.6 #for values in m/s, every limvel, a new profile on the 3d plot
length=0.001 #for values in m/s, length of the arrow
val=0.3 #Value of the unit
labelpad = 3 #distance between ticks and the axis

if dim=='2d':
    if plot_two == 'profile' :
        ax = axs[1]
        disp=' quiver plot '
        ax.set_ylabel('Depth (m)', fontsize=fontsize)
        ax.set_ylim(0, 35)
        ax.set_yticks(y)
        ax.set_yticklabels(labels=depth[::16])
        ax.set_xlabel('Time', fontsize=fontsize)
        ax.set_xlim(np.min(x),np.max(x))
        ax.set_xticks(x)
        ax.set_xticklabels(labels=time, fontsize=fontsize-4)
        if horizontal_line :
            lineh2=ax.axhline(lim1,xmin=0, xmax=1, color='k')
            lineh3=ax.axhline(lim2, xmin=0, xmax=1, color='k')
        # cbar = plt.colorbar(p2, label=disp, ax=ax )#, ticks=60)#ax=ax
        # cbar.ax.tick_params(labelsize=8)
    elif plot_two == 'direction':
        # 2d plot with the direction in degrees
        ax = axs[1]
        disp='Direction (degree)'
        if filter :
            variable2 = signal.medfilt2d(direction2,5)
        else :
            variable2 = direction2
        p2=ax.pcolormesh(variable2, cmap=cmap2, vmin=0, vmax=360)
        ax.set_ylabel('Depth (m)', fontsize=fontsize)
        ax.set_ylim(0, 31)
        ax.set_yticks(y)
        ax.set_yticklabels(labels=depth[::16])
        ax.set_xlabel('Time', fontsize=fontsize)
        ax.set_xlim(np.min(x),np.max(x))
        ax.set_xticks(x)
        ax.set_xticklabels(labels=time)
        cbar = plt.colorbar(p2, label=disp, ax=ax )#, ticks=60)#ax=ax
        cbar.ax.tick_params(labelsize=8)
        # linev1=ax.axvline(fixedir1, color='k')
        # linev2=ax.axvline(fixedir2, color='k')

elif dim=='3d':
    ax=fig.add_subplot(2, 1, 2, projection=dim)#, sharex='row')#,sharey = ax1,sharez = ax1)
    ax.set_zlabel('Depth (m)', fontsize=fontsize-3, labelpad=labelpad-4)
    ax.set_zlim(0, 35)
    ax.set_zticks(y)
    #ax.set_zticklabels(ticklabels=depth[::16], fontsize=fontsize-4)
    #ax.set_xlabel('Velocity East (m/s)', fontsize=fontsize-2, labelpad=labelpad)
    ax.set_xlabel('Profiles', fontsize=fontsize-2, labelpad=labelpad)
    ax.set_xlim(-limvel,limvel*5)
    ax.set_xticks(np.arange(-limvel,len(station)*limvel+0.1,0.3))
    #print('station2', station2)
    #ax.set_xticklabels(station2)
    #ax.set_ylabel('Velocity North (m/s)', fontsize=fontsize-2, labelpad=labelpad-6)
    #ax.set_ylabel('', fontsize=fontsize-2, labelpad=labelpad-6)
    ax.set_ylim(-limvel,limvel)
    ax.set_yticks(np.arange(-limvel,limvel+0.1,0.3))
    #ax.set_yticklabels([])
    ax.tick_params(axis='x', pad=-1)

    # 3D parameters
    azim = -90
    dist = 7
    elev = 25
    ax.azim = azim
    ax.dist = dist
    ax.elev = elev

    #plot échelle à station 0=-0.6m/s
    qNor=ax.quiver3D(-limvel, 0, 0, 0, val, 0, color='grey', arrow_length_ratio=0.3)# ,length=length)#, scale=9, scale_units='dots')
    qEas=ax.quiver(-limvel, 0, 0, val, 0, 0, color='grey',  arrow_length_ratio=0.3)# ,length=length)#, scale=9, scale_units='dots')
    ax.text(-0.5,0.2,1,s=str(val)+'\nm/s N', fontsize=fontsize-5)
    ax.text(-0.5,-0.25,0,s=str(val)+'m/s E', fontsize=fontsize-5)

if sta == 'S9-S24' or sta == 'random':
    for xplot in profilevertical:
        xplot2=np.ones(np.shape(yplot))
        xplot2=xplot2*xplot
        #print('xplot2', xplot2)
        #Define vx (east) and vy (north) in our case for each profile (x2)
        vx=(df.iloc[[xplot],3:34])# for mag 65:96])
        vx=vx.to_numpy()

        vxmoy=df.iloc[int(xplot-moy/2):int(xplot+moy/2), 3:34] #average velocity x over the moy in sec determined
        #print('vxmoy', vxmoy)
        vxmoy=vxmoy.mean() #TODO : check si ok avec nan ?
        vxmoy=vxmoy.to_numpy()
        #print('vxmoy', vxmoy)

        vy=(df.iloc[[xplot], 34:65])  # for dir 96:127])
        vy=vy.to_numpy()

        vymoy=df.iloc[int(xplot-moy/2):int(xplot+moy/2), 34:65]
        vymoy=vymoy.mean() #TODO : check si ok avec nan ?
        vymoy=vymoy.to_numpy()

        vd=df.iloc[[xplot], 96:127]
        vd=vd.to_numpy()

        vx=np.flip(vx) #to have the data of the bottom in first lines
        vxmoy=np.flip(vxmoy)
        vy=np.flip(vy)
        vymoy=np.flip(vymoy)
        vd=np.flip(vd)
        #print('vx, ' , vx)
        #print('vy, ' , vy)

        #average by layer, excluding the nan values
        vymoylayer1=np.nanmean(vymoy[0:lim1]) #bottom
        vymoylayer2=np.nanmean(vymoy[lim1:lim2]) #intermediate
        vymoylayer3=np.nanmean(vymoy[lim2:31]) #surface
        vxmoylayer1=np.nanmean(vxmoy[0:lim1])
        vxmoylayer2=np.nanmean(vxmoy[lim1:lim2])
        vxmoylayer3=np.nanmean(vxmoy[lim2:31])
        # print('vxmoylayer1', vxmoylayer1, vymoylayer1)
        # print('vxmoylayer2', vxmoylayer2, vymoylayer2)
        # print('vxmoylayer3', vxmoylayer3, vymoylayer3)
        # print('vxmoy', vxmoy, vymoy)

        vx=np.nan_to_num(vx, nan=0)
        vxmoy=np.nan_to_num(vxmoy, nan=0)
        vy=np.nan_to_num(vy, nan=0)
        vymoy=np.nan_to_num(vymoy, nan=0)
        vd=np.nan_to_num(vd, nan=0)


        if dim=='2d':
            ax.scatter(xplot2, yplot, marker='x', s=10, color='k')
            q = plt.quiver(xplot2, yplot, vxmoy, vymoy, width=0.003,  color='grey', scale=9 , scale_units='dots')
            qlayer1 = plt.quiver(xplot, 4, vxmoylayer1, vymoylayer1, width=0.005,  color='green', scale=9 , scale_units='dots')
            qlayer2 = plt.quiver(xplot, 12, vxmoylayer2, vymoylayer2, width=0.005,  color='red', scale=9 , scale_units='dots')
            qlayer3 = plt.quiver(xplot, 24, vxmoylayer3, vymoylayer3, width=0.005,  color='blue', scale=9 , scale_units='dots')
            avglen1 = np.around(np.sqrt(np.array(vxmoylayer1) ** 2 + np.array(vymoylayer1) ** 2).mean(),2)
            avglen2 = np.around(np.sqrt(np.array(vxmoylayer2) ** 2 + np.array(vymoylayer2) ** 2).mean(),2)
            avglen3 = np.around(np.sqrt(np.array(vxmoylayer3) ** 2 + np.array(vymoylayer3) ** 2).mean(),2)
            print('avglen', avglen1, avglen2, avglen3)
            #avglen = np.around(np.sqrt(np.array(vx) ** 2 + np.array(vy) ** 2).mean(),2)
            #displen = np.around(avglen, np.int(np.ceil(np.abs(np.log10(avglen)))))
            displen=250
            #displen=np.around(displen/1000,2)
            if incr==0 :
                plt.quiverkey(q, 1.1, 0.05, displen, '{} (m/s)'.format(np.around(displen/1000,2)))

            plt.quiverkey(qlayer1,posx[incr], posy[0], -1, '{}'.format(np.around(avglen1/1000, 2)), color='white', labelcolor='green')
            plt.quiverkey(qlayer2,posx[incr], posy[1], 0, '{}'.format(np.around(avglen2/1000,2)), color='white', labelcolor='red')
            plt.quiverkey(qlayer3,posx[incr], posy[2], 0, '{}'.format(np.around(avglen3/1000, 2)), color='white', labelcolor='blue')
            #plt.quiverkey(q, 0.12*n, 0.05, displen, '{} (m/s)'.format(np.around(displen/1000,2)), labelcolor=labelcolor[incr])
            #ax.quiverkey(q, 0.2, 0.05, 0.25, 'velocity: {} [m/s]'.format(0.25))
            #n=n+1
            #pos = [-1, -1]

        elif dim=='3d':
            zplot=np.zeros(np.shape(yplot))
            zmoy=np.zeros(np.shape(vxmoy))
            #TEST Double xaxis : principle axis=time, secondary = velocity east.
            ax.scatter(zplot+(n-1)*limvel, zplot, yplot, marker='x', s=10, color='k')
            q = ax.quiver(zplot+(n-1)*limvel, zplot, yplot, vxmoy, vymoy, zmoy, color='grey',  arrow_length_ratio=0.3,
                          length=length)#, scale=9, scale_units='dots')
            qlayer1 = ax.quiver(zplot+(n-1)*limvel, zplot, 4,  vxmoylayer1, vymoylayer1, 0, color='green',
                                arrow_length_ratio=0.3, length=length, zorder=1)#, width=0.005,  scale=9, scale_units='dots')
            qlayer2 = ax.quiver(zplot+(n-1)*limvel, zplot, 12, vxmoylayer2, vymoylayer2, 0, color='red',
                                arrow_length_ratio=0.3, length=length, zorder=1)#,width=0.005, , scale=9, scale_units='dots')
            qlayer3 = ax.quiver(zplot+(n-1)*limvel, zplot, 24, vxmoylayer3, vymoylayer3, 0, color='blue',
                                arrow_length_ratio=0.3, length=length, zorder=1, linestyle='solid')#, width=0.005, scale=9, scale_units='dots')
            # ax.scatter(xplot2, zplot, yplot, marker='x', s=10, color='k')
            # q = ax.quiver(xplot2, zplot, yplot, vxmoy, vymoy, zmoy, color='grey', length=0.1, arrow_length_ratio=0.1 )#, scale=9, scale_units='dots')
            # qlayer1 = ax.quiver(xplot, zplot, 4,  vxmoylayer1, vymoylayer1, 0, color='green', length=0.1)#, width=0.005,  scale=9, scale_units='dots')
            # qlayer2 = ax.quiver(xplot, zplot, 12, vxmoylayer2, vymoylayer2, 0, color='red',  length=0.1)#width=0.005, , scale=9, scale_units='dots')
            # qlayer3 = ax.quiver(xplot, zplot, 24, vxmoylayer3, vymoylayer3, 0, color='blue',  length=0.1)#, width=0.005, scale=9, scale_units='dots')
            avglen1 = np.around(np.sqrt(np.array(vxmoylayer1) ** 2 + np.array(vymoylayer1) ** 2).mean(), 2)
            avglen2 = np.around(np.sqrt(np.array(vxmoylayer2) ** 2 + np.array(vymoylayer2) ** 2).mean(), 2)
            avglen3 = np.around(np.sqrt(np.array(vxmoylayer3) ** 2 + np.array(vymoylayer3) ** 2).mean(), 2)
            #print('avglen', avglen1, avglen2, avglen3)
            # avglen = np.around(np.sqrt(np.array(vx) ** 2 + np.array(vy) ** 2).mean(),2)
            # displen = np.around(avglen, np.int(np.ceil(np.abs(np.log10(avglen)))))
            #displen = 250
            # displen=np.around(displen/1000,2)
            if sta=='random':
                if incr==2 or incr==3 :
                    ax.text(posx3dgreen[incr], -0.25, 1 , '{}'.format(np.around(avglen1 / 1000, 2)), color='green', fontsize=fontsize-4)
                    ax.text(posx3dred[incr], 0, 22, '{}'.format(np.around(avglen2 / 1000, 2)), color='red', fontsize=fontsize-4)
                    ax.text(posx3dblue[incr], 0.2, 31 , '{}'.format(np.around(avglen3 / 1000, 2)), color='blue', fontsize=fontsize-4 )
                else :
                    ax.text(posx3dgreen[incr], -0.25, 1 , '{}'.format(np.around(avglen1 / 1000, 2)), color='green', fontsize=fontsize-4)
                    ax.text(posx3dred[incr], 0, 20, '{}'.format(np.around(avglen2 / 1000, 2)), color='red', fontsize=fontsize-4)
                    ax.text(posx3dblue[incr], 0.2, 31 , '{}'.format(np.around(avglen3 / 1000, 2)), color='blue', fontsize=fontsize-4 )
            elif sta=='S9-S24':
                ax.text(-0.5+0.6*n, -0.6, 1, '{}'.format(np.around(avglen1 / 1000, 2)), color='green',
                        fontsize=fontsize - 4)
                ax.text(-0.5+0.6*n, -0.4, 1, '{}'.format(np.around(avglen2 / 1000, 2)), color='red',
                        fontsize=fontsize - 4)
                ax.text(-0.5+0.6*n, -0.2, 1, '{}'.format(np.around(avglen3 / 1000, 2)), color='blue',
                        fontsize=fontsize - 4)
            n=n+1

        else:
            print('ERROR in the dim')

        incr=incr+1

#plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.9, wspace=0.4,  hspace=0.4)
fig.set_size_inches(11, 5)
outfile='current_magnitude_'+sta+'_profiles_velocities_'+dim+'_bylayer_'+str(lim2)+'limbin_'+str(moy)+'sec_'+day+month+'_290923.png'
#fig.savefig(outfile, format='png')

plt.show()