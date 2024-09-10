# 02/03/2023 : moyenne sur les 10 ans de données des rivières à débit horaire
import os
import sys
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from sklearn.metrics import r2_score
import scipy.signal as signal

rep='/home/penicaud/Documents/Data/Décharge_waterlevel/Décharges_rivières/'
create=0 #create the tab 1 or read already saved one 0
save=1

file=os.listdir(rep)
#file=rep+'balat_canal3.txt'
for f in file:
    print("file : ", f)
    date=datetime.datetime(2008,1,1,0,0,0) #Reinitialisation des dates de début des décharges
    year=date.year
    month=date.month
    day=date.day
    hour=date.hour
    print('date', year, month, day, hour)#, date.year())
    # date2 = date + datetime.timedelta(days=59)
    # print('date +60', date2 )
    # date2 = date + datetime.timedelta(days=62)
    # print('date +60', date2 )
    discharge=pd.read_csv(rep+f) #IL FAUDRA LE REP + file
    print('len discharge', len(discharge))

    tab_date=[]
    #create a tab of all the datetime
    for ind in range(len(discharge)):
        tab_date.append(date)
        date=date+datetime.timedelta(days=1)

    df_date=pd.DataFrame(tab_date)
    with pd.option_context('display.max_rows', None, ):
        print(df_date)
    sys.exit(1)

