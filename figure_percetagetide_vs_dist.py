# 11/07 : Create a figure of percentage of tide max at which we detected the frontier between sea water and freshwater
# vs distance

import pandas as pd
import numpy as np
import sys, os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib as mpl
from openpyxl import load_workbook


list_june = {'percentage' : [78,68.6,-72.6,88.8],
            'distance': [1180,3660,-4276, 0],
            'day': [datetime(2022,6,16), datetime(2022,6,17), datetime(2022,6,17), datetime(2022,6,18)],
            'station':['J4', 'J26', 'J33', 'SFJ24']}
# WARNING : the station J17 is not the last value with salinity !=0 but an indication that the salt wedge is upper,
# J17 : 78.5%, -2923, datetime(2022,6,16),
list_august = {'percentage': [85.6,-94.3, 80.8],
            'distance': [-6856, -13073, -800],
            'day': [datetime(2022,8,10), datetime(2022,8,11), datetime(2022,8,12)],
            'station': ['A11', 'A25', 'AF11']}

list_octobre = {'percentage': [-25.64 , -79.87, 85.8, 82.4, 73.7],
                'distance': [-834, -13789, -6311, -3305, -1600],
                'day': [datetime(2022,10,2), datetime(2022,10,3), datetime(2022,10,4), datetime(2022,10,4), datetime(2022,10,4)],
                'station': ['O2', 'O21', 'O33', 'O31', 'OF48']}
# Remove the data of O22 = : -52.6, -7433, datetime(2022,10,3)

df_june = pd.DataFrame(list_june)
df_august = pd.DataFrame(list_august)
df_octobre = pd.DataFrame(list_octobre)

c=0
list_color=['orange', 'green', 'brown']
list_month = ['June', 'August', 'Octobre']
fontsize = 10

fig, ax = plt.subplots()
for df in [df_june,df_august, df_octobre] :
    ax.scatter(df['distance'], df['percentage'], color=list_color[c], label=list_month[c])
    c=c+1
ax.legend()
ax.grid(True)
ax.set_xlabel('Distance (m)', fontsize=fontsize)
ax.set_ylabel('Percentage of tide (%)', fontsize=fontsize)
ax.set_ylim(-100, 100)
outfile = 'percentagetide_vs_distance.png'
fig.savefig(outfile)