import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
import csv
import sys
import xarray as xr

f = 'MES-turbidity_june.xlsx'

col_list = ["Station", "Turbidity", "GFF", "Nuclepore"]
data = pd.read_excel(f, usecols=col_list)
#print(data["Station"])
#df=pd.DataFrame(data, columns=['Station', 'Nuclepore', 'Turbidity'])
df=pd.DataFrame(data, columns=['Turbidity'])
#print(df.sort_values(by='Turbidity')[0:20])

#Selectionner que les couples ou il y a turbidity + concnetration (GFF ou Nucle)
#Comparer l'écart entre GFF et Nucle
#Faire une étude avec les concentrations confondues GFF et Nucle
#Référence = turbidimètre ?


#######################################################################################
#CREATION DES QUANTILES
nq=4 #nb of quantile-1,
a=10
b=0
tab=np.zeros(nq)
df2=df.to_numpy()
#print(df2)
for i in range(nq) :
    #b=int(a/10)-1 #ok pour 10 quantiles
    #print('b', b)
    q=np.nanpercentile(df2, a)
    #print('q', q)
    tab[b]=q
    #print('tab', tab)
    a = a + 10
    b=b+1
#print('quantile' , np.quantile( 0.1))
print(tab)

tab2=np.where(data['Turbidity']<tab[0])
print('numero of cases', data['Turbidity'].to_numpy()[tab2])
print('data', data["GFF"].to_numpy()[tab2])

colors=['blue', 'grey', 'yellow', 'green', 'black', 'purple', 'red', 'turquoise', 'lightgreen', 'pink']
#Figure avec les différentes classes
fig, ax = plt.subplots()#nrows=2)
#ax=axs[0]
ax.set_xlabel('Turbidity', fontsize=15)
ax.set_ylabel('Concentration (mg/L)', fontsize=15)
ax.set_title('Quantile')
for i in range(4):
    print('i', i)
    if i==0 :
        tab2 = np.where(data['Turbidity'] < tab[i])
    else:
        tab2 = np.where(tab[i-1] < data['Turbidity'] < tab[i])

    #print('numero of cases', data['Turbidity'].to_numpy()[tab2])
    #print('data', data["GFF"].to_numpy()[tab2])
    p1=ax.scatter(data['Turbidity'].to_numpy()[tab2], data['GFF'].to_numpy()[tab2], alpha=0.8, marker='x', color=colors[i], label=('GFF q', i))
    ax.legend(loc='lower right')

plt.show()

#######################################################################################"

