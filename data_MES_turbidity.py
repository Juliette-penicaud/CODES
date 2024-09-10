# Author: Florian Wilhelm -- <florian.wilhelm@gmail.com>
# License: BSD 3 clause

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
import csv
import sys
import xarray as xr

f = 'concentration-turbidity.xlsx'

col_list = ["Station Nucle", "Concentration Nucle", "Turbidity Nucle", "Station GFF", "Concentration GFF", "Turbidity GFF"]
data = pd.read_excel(f, usecols=col_list)
#print(data["Station"])
df=pd.DataFrame(data, columns=['Concentration GFF', 'Turbidity GFF'])

estimators = [
    ("OLS", LinearRegression()),
    ("Theil-Sen", TheilSenRegressor(random_state=42))]
#    ("RANSAC", RANSACRegressor(random_state=42)),
colors = {"OLS": "black", "Theil-Sen": "orange"}#, "RANSAC": "lightgreen"}
lw = 2

xGFF=data["Turbidity GFF"]
XGFF = xGFF[:, np.newaxis]
yGFF=data["Concentration GFF"]
#yGFF2=yGFF[:,np.newaxis]

xNucle=data["Turbidity Nucle"]
xNucle=xNucle[np.logical_not(np.isnan(xNucle))]
XNucle = xNucle[:, np.newaxis]
yNucle=data["Concentration Nucle"]
yNucle=yNucle[np.logical_not(np.isnan(yNucle))]
#print('xnucle', XNucle)

reg = LinearRegression().fit(XGFF, yGFF)
print(reg.score(XGFF, yGFF))
print(reg.coef_)
print(reg.intercept_)
#print(reg.get_params)
#reg.predict(np.array([[3, 5]]))
print(df.sort_values(by='Turbidity GFF'))
print(df.quantile(q=0.25))
print(df.quantile(q=0.5))
print(df.quantile(q=0.75))


#Graphe avec des groupes de 5 mesures



sys.exit(1)
ax: object
fig, axs = plt.subplots(nrows=2, ncols=2)
#1. Que GFF
ax=axs[0, 0]
p1=ax.scatter(data["Turbidity GFF"], data["Concentration GFF"], alpha=0.8, marker='x', color='blue', label='GFF')
#ax.set_xlabel('Turbidity', fontsize=15)
ax.set_ylabel('Concentration (mg/L)', fontsize=15)
ax.set_title('GFF')
#ax.legend(loc='lower right')

line_x=np.array([np.min(data["Turbidity GFF"]) , np.max(data["Turbidity GFF"])]) #commun à tous, axe de turbidité max
for name, estimator in estimators:
    #t0 = time.time()
    estimator.fit(XGFF, yGFF)
    #elapsed_time = time.time() - t0
    y_pred = estimator.predict(line_x.reshape(2, 1))
    ax.plot(
        line_x,
        y_pred,
        color=colors[name],
        linewidth=lw,
        label="%s, R² = %2f" % (name, estimator.score(XGFF,yGFF)) # (fit time: %.2fs)" % (name, elapsed_time),
    )
ax.legend(loc="upper left")

#plt.show()

#fig,ax =plt.subplots()
#2.Que Nucle
ax=axs[0, 1]
#ax.set_xlabel('Turbidity', fontsize=15)
ax.set_ylabel('Concentration (mg/L)', fontsize=15)
ax.set_title('Nuclepore')
#colors  = {"OLS": "gold", "Theil-Sen": "lightgreen"}#, "RANSAC": "lightgreen"}
p2=ax.scatter(data["Turbidity Nucle"], data["Concentration Nucle"], alpha=0.8, marker='.', color='red', label='Nuclepore')

lin=LinearRegression().fit(XNucle, yNucle)
y_pred = lin.predict(line_x.reshape(2, 1))
OLSNucle=ax.plot(line_x,y_pred, color=colors["OLS"],linewidth=lw,
                  label="%s, R² = %2f" % ('OLS', lin.score(XNucle,yNucle)))


Theil=TheilSenRegressor().fit(XNucle, yNucle)
y_pred = Theil.predict(line_x.reshape(2, 1))
TSNucle=ax.plot(line_x,y_pred, color=colors["Theil-Sen"],linewidth=lw,
                  label="%s, R² = %2f" % ('Theil-Sen', Theil.score(XNucle,yNucle)))
#for name, estimator in estimators:
#    #t0 = time.time()
#    estimator.fit(XNucle, yNucle)
#    #elapsed_time = time.time() - t0
#    y_pred = estimator.predict(line_x.reshape(2, 1))
#    name =plt.plot(
#        line_x,
#        y_pred,
#        color=colors[name],
#        linewidth=lw,
#        label="%s, R² = %2f" % (name, estimator.score(XNucle,yNucle)) # (fit time: %.2fs)" % (name, elapsed_time),
#    )

#l2=plt.legend([p2, OLSNucle, TSNucle], loc='lower right')
#ax.add_artist(l1) # add l1 as a separate artist to the axes
#ax.grid(True)
ax.legend()
#fig.tight_layout()

#3. Que OLS
ax=axs[1, 0]
p1=ax.scatter(data["Turbidity GFF"], data["Concentration GFF"], alpha=0.8, marker='x', color='blue', label='GFF')
p2=ax.scatter(data["Turbidity Nucle"], data["Concentration Nucle"], alpha=0.8, marker='.', color='red', label='Nuclepore')

ax.set_xlabel('Turbidity', fontsize=15)
ax.set_ylabel('Concentration (mg/L)', fontsize=15)
ax.set_title('OLS')
#ax.legend(loc='lower right')


lin1=LinearRegression().fit(XGFF,yGFF)
y_pred = lin1.predict(line_x.reshape(2, 1))
OLSGFF=ax.plot(line_x,y_pred, color='blue',linewidth=lw,
                  label="%s, R² = %2f" % ('OLS', lin1.score(XGFF,yGFF)))

lin2=LinearRegression().fit(XNucle, yNucle)
y_pred = lin2.predict(line_x.reshape(2, 1))
OLSNucle=ax.plot(line_x,y_pred, color='red',linewidth=lw,
                  label="%s, R² = %2f" % ('OLS', lin2.score(XNucle,yNucle)))
ax.legend()

#3. Que TS
ax=axs[1, 1]
p1=ax.scatter(data["Turbidity GFF"], data["Concentration GFF"], alpha=0.8, marker='x', color='blue', label='GFF')
p2=ax.scatter(data["Turbidity Nucle"], data["Concentration Nucle"], alpha=0.8, marker='.', color='red', label='Nuclepore')

ax.set_xlabel('Turbidity', fontsize=15)
ax.set_ylabel('Concentration (mg/L)', fontsize=15)
ax.set_title('Theil-Sen')
#ax.legend(loc='lower right')


TS1=TheilSenRegressor().fit(XGFF,yGFF)
y_pred = TS1.predict(line_x.reshape(2, 1))
TSGFF=ax.plot(line_x,y_pred, color='blue',linewidth=lw,
                  label="%s, R² = %2f" % ('Theil-Sen', TS1.score(XGFF,yGFF)))

TS2=TheilSenRegressor().fit(XNucle, yNucle)
y_pred = TS2.predict(line_x.reshape(2, 1))
TSNucle=ax.plot(line_x,y_pred, color='red',linewidth=lw,
                  label="%s, R² = %2f" % ('Theil-Sen', TS2.score(XNucle,yNucle)))
ax.legend(loc="upper left")


#plt.savefig('2-5june_MES_turb.png', format='png')
plt.show()

sys.exit(1)
##################################################################

#Graphe avec quantiles (combien ?) avec X points dedans, 1 point par quantile choisi randomly pour faire une courbe.
# Expé répétée X fois et on prend la médiane des pentes (représenter la valeur des pentes sur un graphe, en calculant percentiles.







############################################################################################

estimators = [
    ("OLS", LinearRegression()),
    ("Theil-Sen", TheilSenRegressor(random_state=42)),
    ("RANSAC", RANSACRegressor(random_state=42)),
]
colors = {"OLS": "turquoise", "Theil-Sen": "gold", "RANSAC": "lightgreen"}
lw = 2

plt.scatter(x, y, color="indigo", marker="x", s=40)
line_x = np.array([-3, 3])
for name, estimator in estimators:
    t0 = time.time()
    estimator.fit(X, y)
    elapsed_time = time.time() - t0
    y_pred = estimator.predict(line_x.reshape(2, 1))
    plt.plot(
        line_x,
        y_pred,
        color=colors[name],
        linewidth=lw,
        label="%s (fit time: %.2fs)" % (name, elapsed_time),
    )

plt.axis("tight")
plt.legend(loc="upper left")
_ = plt.title("Corrupt y")



# for row in data:
#   print(row['TURBIDITY'], row['TURBIDITY'])
#with open(f) as csvfile:
# data = csv.DictReader(csvfile)