#27/07/2022 Treat ADCP DATA
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from adcploader import *

#Compare averages of the profiles (no avg, 5sec, 10 sec, 15sec)
day='18'
month='06'
year='2022'
file='/home/penicaud/Documents/Data/ADCP/'+day+month+'_alldata.csv'
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

data = pd.read_csv(file, skiprows=11, low_memory=False, sep=',', usecols=col_list)
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
cmap2=plt.cm.plasma
fontsize=12


var='salt'
#X axis
if var=='all data' :
    x=np.linspace(0,22948,6)
    fixe3 = 1850
    fixe1 = 6885
    fixe2 = 16064
    fixedir1 = 12362
    fixedir2 = 7742
elif var=='salt':
    x=np.linspace(16064, 22948, 6)
else:
    print('Problem')

print('shape', len(x))
for l in range(0,len(x)-1):
    x[l]=int(x[l])
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
y = np.linspace(0, 31, 3)
print('y', y)
depth=np.around(np.arange(-10.2,-0.3,0.3), decimals=1)
print(depth)
print('depth', depth[::16])

lim1=7#nb of the limit bin between layer 1 and 2
lim2=17# nb of the limit bin between layer 2 and 3 #17 contain every intense bins, 15 around 80 % (a vue de nez)

fig,axs = plt.subplots(nrows=2)
fig.suptitle('Current magnitude and direction')
ax =axs[0]
disp='Magnitude (mm/s)'
ax.set_ylabel('Depth (m)', fontsize=fontsize)
ax.set_ylim(0, 31)
ax.set_yticks(y)
ax.set_yticklabels(labels=depth[::16])
#ax.set_xlabel('Time', fontsize=fontsize)
ax.set_xlim(np.min(x),np.max(x))
ax.set_xticks(x)
ax.set_xticklabels(labels=time, fontsize=fontsize-4)
#ax.set_xlim(0, np.max(x) + 1)
#ax.set_xticks(ticks=x)  # , labels=stations)
#ax.set_xticklabels(stations)  # , minor=True)
p1=ax.pcolormesh(magnitude2, cmap=cmap, vmin=0, vmax=1000)
# linev3=ax.axvline(fixe3,color='k')
# #lineh3=ax.axhline(20, xmin=0, xmax=0.1, color='k')
# linev1=ax.axvline(fixe1, color='k')
# linev2=ax.axvline(fixe2, color='k')
lineh2=ax.axhline(lim1,xmin=0, xmax=1, color='k') #17 contain every intense bins, 15 around 80 % (a vue de nez)
lineh3=ax.axhline(lim2, xmin=0, xmax=1, color='k')
cbar = plt.colorbar(p1, label=disp, ax=ax)  # , ticks=1)#ax=ax
cbar.ax.tick_params(labelsize=8)


# ax = axs[1]
# disp='Direction (degree)'
# p2=ax.pcolormesh(direction2, cmap=cmap2, vmin=0, vmax=360)
# ax.set_ylabel('Depth (m)', fontsize=fontsize)
# ax.set_ylim(0, 31)
# ax.set_yticks(y)
# ax.set_yticklabels(labels=depth[::16])
# ax.set_xlabel('Time', fontsize=fontsize)
# ax.set_xlim(np.min(x),np.max(x))
# ax.set_xticks(x)
# ax.set_xticklabels(labels=time)
# cbar = plt.colorbar(p2, label=disp, ax=ax )#, ticks=60)#ax=ax
# cbar.ax.tick_params(labelsize=8)
# # linev1=ax.axvline(fixedir1, color='k')
# # linev2=ax.axvline(fixedir2, color='k')


ax = axs[1]
disp=' quiver plot '
ax.set_ylabel('Depth (m)', fontsize=fontsize)
ax.set_ylim(0, 31)
#ax.set_yticks(y)
#ax.set_yticklabels(labels=depth[::16])
ax.set_xlabel('Time', fontsize=fontsize)
ax.set_xlim(np.min(x),np.max(x))
ax.set_xticks(x)
ax.set_xticklabels(labels=time)
lineh2=ax.axhline(lim1,xmin=0, xmax=1, color='k')
lineh3=ax.axhline(lim2, xmin=0, xmax=1, color='k')
# cbar = plt.colorbar(p2, label=disp, ax=ax )#, ticks=60)#ax=ax
# cbar.ax.tick_params(labelsize=8)
x2=x
x2[0]=16382
x2[-1]=22382
print('x2', x2)
#yplot = depth[::4] #need the values of the bins
yplot = np.linspace(0,32, 9)
yplot=np.arange(1,32,1)
print('yplot', yplot)
#x2=x2[0:1]
n=1 #indicator for quiver plot legend
moy=16 #EVEN NUMBER #nb of seconds and of profiles we want to average
labelcolor='black'
incr=0
pos=[0.12,0.05]


for xplot in x2:
    xplot2=np.ones(np.shape(yplot))
    xplot2=xplot2*xplot
    print('xplot2', xplot2)
    #Define vx (east) and vy (north) in our case for each profile (x2)
    vx=(df.iloc[[xplot],3:34])# for mag 65:96])
    vx=vx.to_numpy()

    vxmoy=df.iloc[int(xplot-moy/2):int(xplot+moy/2), 3:34]
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

    #average by layer, excluding the nan values
    print(vymoy)
    vymoylayer1=np.nanmean(vymoy[0:lim1])
    print(vymoylayer1)
    vymoylayer2=np.nanmean(vymoy[lim1:lim2])
    print(vymoylayer2)
    vymoylayer3=np.nanmean(vymoy[lim2:31])
    print(vymoylayer3)
    print(vxmoy)
    vxmoylayer1=np.nanmean(vxmoy[0:lim1])
    print(vxmoylayer1)
    vxmoylayer2=np.nanmean(vxmoy[lim1:lim2])
    print(vxmoylayer2)
    vxmoylayer3=np.nanmean(vxmoy[lim2:31])
    print(vxmoylayer3)

    vx=np.nan_to_num(vx, nan=0)
    vxmoy=np.nan_to_num(vxmoy, nan=0)
    vy=np.nan_to_num(vy, nan=0)
    vymoy=np.nan_to_num(vymoy, nan=0)
    vd=np.nan_to_num(vd, nan=0)
    vx=np.flip(vx) #to have the data of the bottom in first lines
    vxmoy=np.flip(vxmoy)
    vy=np.flip(vy)
    vymoy=np.flip(vymoy)
    vd=np.flip(vd)
    #print('vx, ' , vx)
    #print('vy, ' , vy)

    ax.scatter(xplot2, yplot, marker='x', s=10, color='k')
    q = plt.quiver(xplot2, yplot, vxmoy, vymoy, width=0.003,  color='grey', scale=9 , scale_units='dots')
    q2 = plt.quiver(xplot2, yplot, vxmoy, vymoy, width=0.003,  color='grey', scale=9 , scale_units='dots')
    #avglen = np.sqrt(np.array(vxmoy) ** 2 + np.array(vymoy) ** 2).mean()
    #displen = np.around(avglen, np.int(np.ceil(np.abs(np.log10(avglen)))))
    displen=250
    #displen=np.around(displen/1000,2)
    plt.quiverkey(q, pos[0], pos[1], displen, '{} (m/s)'.format(np.around(displen/1000,2)), labelcolor=labelcolor)
    #plt.quiverkey(q, 0.12*n, 0.05, displen, '{} (m/s)'.format(np.around(displen/1000,2)), labelcolor=labelcolor[incr])
    #ax.quiverkey(q, 0.2, 0.05, 0.25, 'velocity: {} [m/s]'.format(0.25))
    n=n+1.3
    labelcolor='white'#to not see the text
    pos=[-1,-1]#to put the arrowws out of the graph


plt.subplots_adjust(left=0.11,
                    bottom=0.1,
                    right=0.95,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

outfile='current_magnitude_profiles_velocities_'+str(moy)+'sec_'+day+month+'.png'
#fig.savefig(outfile, format='png')

plt.show()


sys.exit(1)


def plot_profile_2d(p, cfg=[]):
    ''' plot the ground view
    cfg vars:
    vscale        float        scale the the displayed velocity vectors
    saveas        string        path to output file, optional
    title        string        title of plot, optional
    '''

    import matplotlib.pyplot as plt
    import numpy as np

    # position of ensembles
    x = [e.position.x for e in p.ensembles]
    y = [e.position.y for e in p.ensembles]

    # velocity components. expects uv_rot to be False!!
    vx = [e.velocity.x if not e.void else 0 for e in p.ensembles]
    vy = [e.velocity.y if not e.void else 0 for e in p.ensembles]

    # average length for the quiverkey
    avglen = np.sqrt(np.array(vx) ** 2 + np.array(vy) ** 2).mean()
    displen = np.around(avglen, np.int(np.ceil(np.abs(np.log10(avglen)))))

    # scale the vectors
    vscale = cfg['vscale'] if 'vscale' in cfg else 1
    # print(vscale)
    # FIXME: scale doesnt have any effect!

    plt.plot(x, y, marker='x', color='black')
    plt.scatter(x[0], y[0], marker='o')
    plt.axis('equal')
    plt.grid()
    q = plt.quiver(x, y, vx * vscale, vy * vscale, width=0.003, color='grey')
    plt.quiverkey(q, 0.2, 0.05, displen * vscale, 'velocity: {} [m/s]'.format(displen))

    plt.title(cfg['title'] if 'title' in cfg else 'plan view')
    plt.ylabel('y coordinate [m]')
    plt.xlabel('x coordinate [m]')

    if not 'saveas' in cfg or cfg['saveas'] == False:
        plt.show()
    else:
        plt.savefig(cfg['saveas'])

    pass

    # clear figure and axis.
    plt.clf()
    plt.cla()
