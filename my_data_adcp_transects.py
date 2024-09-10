#27/07/2022 Process ADCP DATA
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
#from adcploader import *
from mpl_toolkits import mplot3d


#LIST OF variables : station, station2 same but with '' inbetween to plot on the 2d graph, profilevertical : the number of the rows of the table

month='06' #TO COMPLETE
year='2022'
test=True # ne pas enregistrer image

if month=='06':
    survey='June'
    transect = 'T2'  # T1 to T4 #TO COMPLETE
    if transect == 'T1' or transect == 'T2':
        day = '16'
        T = ''
    elif transect == 'T3' or transect == 'T4':
        day = '17'
        T = '_' + transect
    else:
        print('problem with the transect')
elif month=='08':
    survey='August'
elif month=='10':
    survey='October'
else :
    print("PROBLEM WITH THE MONTH")

file='/home/penicaud/Documents/Data/ADCP/Survey_'+survey+'/'+day+month+T+'_alldata_BT.csv'
print(file)

nb_bin=45 #TO COMPLETE
var='dir' #'dir or profile #TO COMPLETE
dim='3d' #'2d' or '3d' #TO COMPLETE
if var =='dir':
    dim='2d'

moy=16 #EVEN NUMBER #nb of seconds and of profiles we want to average #WARNING TODO check if it sec or nb of files
layer=False #Bool to add the averaged velocity of the layers, need to define the lim of the layers
lim1=7#nb of the limit bin between layer 1 and 2
lim2=17# nb of the limit bin between layer 2 and 3 #FOR 1806 : 17 contain every intense bins, 15 around 80 % (a vue de nez)

CTD_to_observe='LOG' #'LOG' or 'IMER'


#######################################################################################""
#load the data of the csv file of ADCP data
col_list = ["Ens", "HH", "MM", "SS",]
sta=[] #stations

for i in range(1,nb_bin+1):
    string="Mag, mm/s, "+str(i)
    col_list.append(string)
for i in range(1,nb_bin+1):
    string="Dir, deg, "+str(i)
    col_list.append(string)
for i in range(1,nb_bin+1):
    string="Eas, mm/s, "+str(i)
    col_list.append(string)
for i in range(1,nb_bin+1):
    string="Nor, mm/s, "+str(i)
    col_list.append(string)

data = pd.read_csv(file, skiprows=11, low_memory=False, sep=' ', usecols=col_list)
df = pd.DataFrame(data)

print(df)
hour = df["HH"]
mn = df["MM"]
sec = df["SS"]
ens = df["Ens"]


mag,dir=[],[]
for i in range(1,nb_bin):
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


######################################################################################
#DF of the stations of the transects
for i in range(1, 40):
    string = "S" + str(i)
    sta.append(string)

df_sta = pd.DataFrame({'stations': sta,
                    'hour' : ['13.09', '13.32','13.56','14.23','14.44','15.19','15.42','16.02','16.22','16.35',
                              '17.05','18.00', '18.28','18.52','19.20','19.35','19.48','12.22','12.32','12.40',
                              '13.00', '13.11','13.23','13.40','13.56','14.14','14.31', '14.48','20.31','20.47',
                              '21.05','21.20','21.34','21.42','21.52','22.00','22.22','22.35','22.58']
                       #bidouillage for the 1st station of the transect T1 : 13.08.00 doesn't exist, so 13.09
                       #same for T3 : 12.16 is the right hour of station, but only ADCP from 12.22
                    })

if transect=='T1':
    df2=df_sta[0:10].copy()
    label1='River' #label to localize in space
    label2='Ocean'
elif transect=='T2':
    df2=df_sta[10:17].copy()
    label2='River'
    label1='Ocean'
elif transect=='T3':
    df2=df_sta[17:28].copy()
    label2='River'
    label1='Ocean'
elif transect=='T4':
    df2=df_sta[28:39].copy()
    label2='River'
    label1='Ocean'

print(df2)
station=df2['stations']
hour_station=df2['hour']

#Add the numero of the ensemble of each hour
num=[]
for h in hour_station:
    hour_sta = int(h[0:2])
    mn_sta = int(h[3:5])
    sec_sta = 30
    condition = ((df.HH == hour_sta) & (df.MM == mn_sta) & (df.SS == sec_sta))
    l = df.index[condition]
    l = l[0] + 1  # +1 because it gives the index and indexing begins by 0
    print('hour', h, hour_sta, mn_sta, l)
    num.append(l)

df2.insert(2,'numero', num)
print(df2)


cmap=plt.cm.jet
cmap2=plt.cm.plasma
fontsize=12

# print((df.iloc[-1]))#==24186) #gives a serie
# print(df[df.columns[0]].count()) #Warning : if the file doesnt begin by 0 or 1, doesn't work
# print(ens.iloc[-1]) #THE GOOD SOLUTION


#X axis
#x=np.linspace(1,df2.at[len(df2['stations'])-1,'numero'],len(df2['stations']), dtype=int)
deb = df2['numero'].iloc[0]
end = df2['numero'].iloc[-1]
print('deb end', deb, end)
if deb != 1 :
    if deb-ens.iloc[0]>200 :
        deb = deb-200
    else:
        deb=ens.iloc[0]
if end != ens.iloc[-1]:
    if ens.iloc[-1]-end > 200 :
        end= end + 200
    else :
        end = ens.iloc[-1]-1
print('deb, end', deb, end)
x=np.linspace(deb,end,4, dtype=int)
#x2= #the x values
print('x', x)


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
y = np.linspace(0, nb_bin-1, 3)
print('y', y)
dmax=0.3*nb_bin+0.6 #0.6 is the blank first bin, 0.3 is the range of the bins
depth=np.around(np.arange(-dmax,-0.3,0.3), decimals=1)
print(depth)
depth2=[depth[0], -(depth[-1]-depth[0])/2, depth[-1]]
print('depth2', depth2) #better to use this one, the other is not right every time
print('depth', depth[::int(nb_bin/2)])
yplot=np.arange(1,nb_bin+1,1)
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
    profilevertical=df2['numero']
elif dim=='3d':
    fig=plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    station2 = station.copy()
    station2=station2.values.tolist() #from df or series to list
    station2.insert(0, '')
    i = 1
    while i < 2 * len(station): #need this trick only for 3D
        station2.insert(i, '')
        i += 2
    profilevertical=df2['numero']

#1st plot of the magnitude of current
# fig.suptitle('Current magnitude and velocity profiles '+transect)
fig.suptitle(day+'/'+month+', '+transect)
disp='Magnitude (mm/s)'
ax.set_ylabel('Depth (m)', fontsize=fontsize-2)
ax.set_yticks(y)
ax.set_yticklabels(labels=depth2, fontsize=fontsize-2)
ax.set_ylim(0, nb_bin)

# ax.set_yticklabels(labels=depth[::int(nb_bin/2)], fontsize=fontsize-2)
#ax.set_xlabel('Time', fontsize=fontsize)
ax.set_xlim(np.min(x),np.max(x))
ax.set_xticks(x)
ax.set_xticklabels(labels=time, fontsize=fontsize-4)
#ax.set_xlim(0, np.max(x) + 1)
#ax.set_xticks(ticks=x)  # , labels=stations)
#ax.set_xticklabels(stations)  # , minor=True)
if transect=='T4':
    vmax=1500
else :
    vmax=1000
p1=ax.pcolormesh(magnitude2, cmap=cmap, vmin=0, vmax=vmax)
cbar = plt.colorbar(p1, label=disp, ax=ax, extend='max')  # , ticks=1)#ax=ax
cbar.ax.tick_params(labelsize=8)
ax.text(x[0]-450,-10,label1, fontsize=fontsize-4, fontweight='bold')
ax.text(x[-1]-450,-10,label2, fontsize=fontsize-4, fontweight='bold')


#Plot the vertical profiles only
xver=0 #increment only for the list of "stations"
for xvertical in profilevertical:
    #s = station[xver]
    s = df2['stations'].iloc[xver]
    print('station', s)
    ax.axvline(xvertical, color='black')
    ax.text(xvertical-300, y=np.max(yplot)+1, s=s, color='k', fontsize=fontsize-2) #WARNING : the xvertical-300 depends on the xaxis size!!! 300 is ok for T4
    xver=xver+1


if var=='dir':

    #2d plot with the direction in degrees
    ax = axs[1]
    disp='Direction (degree)'
    p2=ax.pcolormesh(direction2, cmap=cmap2, vmin=0, vmax=360)
    ax.set_ylabel('Depth (m)', fontsize=fontsize)
    ax.set_ylim(0, nb_bin)
    ax.set_yticks(y)
    ax.set_yticklabels(labels=depth2)#depth[::int(nb_bin/2)])
    ax.set_xlabel('Time', fontsize=fontsize)
    ax.set_xlim(np.min(x),np.max(x))
    ax.set_xticks(x)
    ax.set_xticklabels(labels=time)
    cbar = plt.colorbar(p2, label=disp, ax=ax )#, ticks=60)#ax=ax
    cbar.ax.tick_params(labelsize=8)
    # Plot the vertical profiles only
    xver = 0  # increment only for the list of "stations"
    for xvertical in profilevertical:
        # s = station[xver]
        s = df2['stations'].iloc[xver]
        ax.axvline(xvertical, color='black')
        ax.text(xvertical - 300, y=np.max(yplot) + 1, s=s, color='k', fontsize=fontsize - 2)
        xver = xver + 1

    outfile='current_magnitude_direction_stations_'+transect+'_'+day+month+'.png'
    fig.savefig(outfile, format='png')
    plt.show()

    sys.exit(1)



elif var=='profile':
    ###################################################################################################################
    #Parameters for the 2d plot
    if transect=='T4':
        limvel=1.2 #0.6 #for values in m/s, every limvel, a new profile on the 3d plot
    else :
        limvel = 0.8
    length=0.001 #for values in m/s, length of the arrow
    val=limvel/2 #0.3 #Value of the unit
    labelpad = 3 #distance between ticks and the axis

    if dim=='2d':
        ax = axs[1]
        disp=' quiver plot '
        ax.set_ylabel('Depth (m)', fontsize=fontsize)
        ax.set_ylim(0, nb_bin+2) #+2 to see the arrows going up
        ax.set_yticks(y)
        ax.set_yticklabels(labels=depth2)#depth[::int(nb_bin/2)])
        ax.set_xlabel('Time', fontsize=fontsize)
        ax.set_xlim(np.min(x),np.max(x))
        ax.set_xticks(x)
        ax.set_xticklabels(labels=time, fontsize=fontsize-4)
        lineh2=ax.axhline(lim1,xmin=0, xmax=1, color='k')
        lineh3=ax.axhline(lim2, xmin=0, xmax=1, color='k')
        # cbar = plt.colorbar(p2, label=disp, ax=ax )#, ticks=60)#ax=ax
        # cbar.ax.tick_params(labelsize=8)

    elif dim=='3d':
        ax=fig.add_subplot(2, 1, 2, projection=dim)#, sharex='row')#,sharey = ax1,sharez = ax1)
        ax.set_zlabel('Depth (m)', fontsize=fontsize-3, labelpad=labelpad-4)
        ax.set_zlim(0, 35)
        ax.set_zticks(y)
        ax.set_zticklabels(ticklabels=depth2, fontsize=fontsize-4)#depth[::int(nb_bin/2)]
        #ax.set_xlabel('Velocity East (m/s)', fontsize=fontsize-2, labelpad=labelpad)
        ax.set_xlabel('Profiles', fontsize=fontsize-2, labelpad=labelpad)
        ax.set_xlim(-limvel,limvel*5) #TODO : check why *5
        ax.set_xticks(np.arange(-limvel,len(station)*limvel+0.1,val))
        #print('station2', station2)
        ax.set_xticklabels(station2)
        #ax.set_ylabel('Velocity North (m/s)', fontsize=fontsize-2, labelpad=labelpad-6)
        #ax.set_ylabel('', fontsize=fontsize-2, labelpad=labelpad-6)
        ax.set_ylim(-limvel,limvel)
        ax.set_yticks(np.arange(-limvel,limvel+0.1,val))
        ax.set_yticklabels([])
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
        qEas=ax.quiver3D(-limvel, 0, 0, val, 0, 0, color='grey',  arrow_length_ratio=0.3)# ,length=length)#, scale=9, scale_units='dots')
        ax.text(-3.2*val,0.5*val,1,s=str(val)+'\nm/s N', fontsize=fontsize-5)
        ax.text(-1.8*val,-0.9*val,0,s=str(val)+'m/s E', fontsize=fontsize-5)

    for xplot in profilevertical:
        xplot2=np.ones(np.shape(yplot))
        xplot2=xplot2*xplot
        #print('xplot2', xplot2)
        #Define vx (east) and vy (north) in our case for each profile (x2)
        vx=(df.iloc[[xplot],4:4+nb_bin])# for mag 65:96])
        vx=vx.to_numpy()

        if deb == 1:
            vxmoy=df.iloc[int(xplot):int(xplot+moy), 4:4+nb_bin] #do the average on the 16seconds after, because we cannot center on the 1st value
        else : #TODO : see if this condition is enough
            vxmoy=df.iloc[int(xplot-moy/2):int(xplot+moy/2), 4:4+nb_bin] #average velocity x over the moy in sec determined
        #print('vxmoy', vxmoy)
        vxmoy=vxmoy.mean() #TODO : check si ok avec nan ?
        vxmoy=vxmoy.to_numpy()
        #print('vxmoy', vxmoy)

        vy=(df.iloc[[xplot], 4+nb_bin:4+2*(nb_bin)])  # for dir 96:127])
        vy=vy.to_numpy()

        if deb ==1:
            vymoy=df.iloc[int(xplot):int(xplot+moy), 4+nb_bin:4+2*(nb_bin)] #do the average on the 16seconds after, because we cannot center on the 1st value
        else : #TODO : see if this condition is enough
            vymoy=df.iloc[int(xplot-moy/2):int(xplot+moy/2), 4+nb_bin:4+2*(nb_bin)] #average velocity x over the moy in sec determined

        #vymoy=df.iloc[int(xplot-moy/2):int(xplot+moy/2), 34:65]
        vymoy=vymoy.mean() #TODO : check si ok avec nan ?
        vymoy=vymoy.to_numpy()

        #vd=df.iloc[[xplot] ,4+5*nb_bin:4+6*nb_bin ] # 96:127] #DOESNT WORK : EMPTY
        #print('vd', vd)
        #vd=vd.to_numpy()

        vx=np.flip(vx) #to have the data of the bottom in first lines
        vxmoy=np.flip(vxmoy)
        vy=np.flip(vy)
        vymoy=np.flip(vymoy)
        #vd=np.flip(vd)
        #print('vx, ' , vx)
        #print('vy, ' , vy)

        if layer==True :
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
        #vd=np.nan_to_num(vd, nan=0)


        if dim=='2d':
            ax.scatter(xplot2, yplot, marker='x', s=10, color='k')
            q = plt.quiver(xplot2, yplot, vxmoy, vymoy, width=0.003,  color='grey', scale=9 , scale_units='dots')
            if layer == True :
                qlayer1 = plt.quiver(xplot, 4, vxmoylayer1, vymoylayer1, width=0.005,  color='green', scale=9 , scale_units='dots')
                qlayer2 = plt.quiver(xplot, 12, vxmoylayer2, vymoylayer2, width=0.005,  color='red', scale=9 , scale_units='dots')
                qlayer3 = plt.quiver(xplot, 24, vxmoylayer3, vymoylayer3, width=0.005,  color='blue', scale=9 , scale_units='dots')
                avglen1 = np.around(np.sqrt(np.array(vxmoylayer1) ** 2 + np.array(vymoylayer1) ** 2).mean(),2)
                avglen2 = np.around(np.sqrt(np.array(vxmoylayer2) ** 2 + np.array(vymoylayer2) ** 2).mean(),2)
                avglen3 = np.around(np.sqrt(np.array(vxmoylayer3) ** 2 + np.array(vymoylayer3) ** 2).mean(),2)
                print('avglen', avglen1, avglen2, avglen3)
                displen = 250
                plt.quiverkey(qlayer1, posx[incr], posy[0], -1, '{}'.format(np.around(avglen1 / 1000, 2)), color='white',
                              labelcolor='green')
                plt.quiverkey(qlayer2, posx[incr], posy[1], 0, '{}'.format(np.around(avglen2 / 1000, 2)), color='white',
                              labelcolor='red')
                plt.quiverkey(qlayer3, posx[incr], posy[2], 0, '{}'.format(np.around(avglen3 / 1000, 2)), color='white',
                              labelcolor='blue')
            #avglen = np.around(np.sqrt(np.array(vx) ** 2 + np.array(vy) ** 2).mean(),2)
            #displen = np.around(avglen, np.int(np.ceil(np.abs(np.log10(avglen)))))
            #displen=np.around(displen/1000,2)
            if incr==0 :
                plt.quiverkey(q, 1.1, 0.05, displen, '{} (m/s)'.format(np.around(displen/1000,2)))
            #plt.quiverkey(q, 0.12*n, 0.05, displen, '{} (m/s)'.format(np.around(displen/1000,2)), labelcolor=labelcolor[incr])
            #ax.quiverkey(q, 0.2, 0.05, 0.25, 'velocity: {} [m/s]'.format(0.25))
            #n=n+1
            #pos = [-1, -1]

        elif dim=='3d':
            zplot=np.zeros(np.shape(yplot))
            zmoy=np.zeros(np.shape(vxmoy))
            ax.scatter(zplot+(n-1)*limvel, zplot, yplot, marker='x', s=10, color='k')
            q = ax.quiver(zplot+(n-1)*limvel, zplot, yplot, vxmoy, vymoy, zmoy, color='grey',  arrow_length_ratio=0.3,
                          length=length)#, scale=9, scale_units='dots')
            if layer == True :
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
                ax.text(-0.5 + limvel * n, -0.6, 1, '{}'.format(np.around(avglen1 / 1000, 2)), color='green',
                        fontsize=fontsize - 4)
                ax.text(-0.5 + limvel * n, -0.4, 1, '{}'.format(np.around(avglen2 / 1000, 2)), color='red',
                        fontsize=fontsize - 4)
                ax.text(-0.5 + limvel * n, -0.2, 1, '{}'.format(np.around(avglen3 / 1000, 2)), color='blue',
                        fontsize=fontsize - 4)
            # if sta=='random':
            #     if incr==2 or incr==3 :
            #         ax.text(posx3dgreen[incr], -0.25, 1 , '{}'.format(np.around(avglen1 / 1000, 2)), color='green', fontsize=fontsize-4)
            #         ax.text(posx3dred[incr], 0, 22, '{}'.format(np.around(avglen2 / 1000, 2)), color='red', fontsize=fontsize-4)
            #         ax.text(posx3dblue[incr], 0.2, 31 , '{}'.format(np.around(avglen3 / 1000, 2)), color='blue', fontsize=fontsize-4 )
            #     else :
            #         ax.text(posx3dgreen[incr], -0.25, 1 , '{}'.format(np.around(avglen1 / 1000, 2)), color='green', fontsize=fontsize-4)
            #         ax.text(posx3dred[incr], 0, 20, '{}'.format(np.around(avglen2 / 1000, 2)), color='red', fontsize=fontsize-4)
            #         ax.text(posx3dblue[incr], 0.2, 31 , '{}'.format(np.around(avglen3 / 1000, 2)), color='blue', fontsize=fontsize-4 )

            n=n+1

        else:
            print('ERROR in the dim')



        incr=incr+1



    plt.subplots_adjust(left=0.12,
                        bottom=0.1,
                        right=0.95,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    if test :
        print('Pas d enregistrement')
        sys.exit(1)
    outfile='current_magnitude_profiles_velocities_'+dim+'_'+str(moy)+'sec_'+transect+'_'+day+month+'.png'
    fig.savefig(outfile, format='png')

    plt.show()