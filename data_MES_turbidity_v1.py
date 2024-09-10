import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
import csv
import sys
import xarray as xr

#Analyse avec les 2 estimators Theil Sen et OLS des regressions pour la courbe de targae ==> pour le profil de turbidité et quantifier les flux.


month='08' #TO CHANGE
oneplot=False #Only GFF and Nucle with OLS trend and values

if month=='06':
    f = 'concentration-turbidity_16-18june.xlsx'
    survey='Survey June'
elif month=='08':
    f='/home/penicaud/Documents/Data/survey_august/Turbidity-Concentration_August.xlsx'#Stations_10-13_august.xlsx'#Turbidity_concentration_august.xlsx'
    survey='Survey August'
    #nrows=32 #for both column
    nrows=64 #only for the gff column

print('file', f)

#col_list = ["Stations", "Turbidity", "NUCLEPORE Concentration (mg/L)", "GFF Concentration (mg/L)"]#, "Station GFF", "Turbidity GFF", "Concentration GFF"]
col_list=["Station GFF", "Turbidity GFF", "Concentration GFF", "Turbidity Nuclepore", "Concentration Nuclepore", "GFF 2", "Station Nuclepore"]
data = pd.read_excel(f, usecols=col_list, nrows=nrows)
#print(data["Station"])
df=pd.DataFrame(data, columns=col_list)
print(df.sort_values(by='Turbidity GFF', ascending=True))#[60:80])

print(df['Concentration GFF'].min())#, df.iloc[df['Concentration GFF'].min()])
print(df['Concentration GFF'].max())


estimators = [
    ("OLS", LinearRegression()),
    ("Theil-Sen", TheilSenRegressor(random_state=42))]
#    ("RANSAC", RANSACRegressor(random_state=42)),
#colors = {"OLS": "turquoise", "Theil-Sen": "lightgreen"}#, "RANSAC": "lightgreen"}
colors = {"OLS": "black", "Theil-Sen": "orange"}#, "RANSAC": "lightgreen"}
lw = 2

fontsize=10

xGFF=data["Turbidity GFF"]
XGFF = xGFF[:, np.newaxis]
yGFF=data["Concentration GFF"]
#yGFF2=yGFF[:,np.newaxis]

xNucle=data["Turbidity Nuclepore"]
xNucle=xNucle[np.logical_not(np.isnan(xNucle))]
XNucle = xNucle[:, np.newaxis]
yNucle=data["Concentration Nuclepore"]
yNucle=yNucle[np.logical_not(np.isnan(yNucle))]
#print('xnucle', XNucle)

if oneplot :
    print('1 plot OLS ')
    fig, ax =plt.subplots()
    fig.suptitle(survey, fontsize=fontsize)
    line_x = np.array(
        [np.min(data["Turbidity GFF"]), np.max(data["Turbidity GFF"])])  # commun à tous, axe de turbidité max
    p1 = ax.scatter(data["Turbidity GFF"], data["Concentration GFF"], alpha=0.8, marker='x', color='blue', label='GFF')
    p2 = ax.scatter(data["Turbidity Nuclepore"], data["Concentration Nuclepore"], alpha=0.8, marker='.', color='red',
                    label='Nuclepore')

    ax.set_xlabel('Turbidity', fontsize=fontsize)
    ax.set_ylabel('Concentration (mg/L)', fontsize=fontsize)
    ax.set_title('OLS', fontsize=fontsize)
    # ax.legend(loc='lower right')

    lin1 = LinearRegression().fit(XGFF, yGFF)
    y_pred = lin1.predict(line_x.reshape(2, 1))
    OLSGFF = ax.plot(line_x, y_pred, color='blue', linewidth=lw,
                     label="%s, R² = %2f, y=%.2f x + %.2f" % ('OLS', lin1.score(XGFF, yGFF), lin1.coef_, lin1.intercept_))

    lin2 = LinearRegression().fit(XNucle, yNucle)
    y_pred = lin2.predict(line_x.reshape(2, 1))
    OLSNucle = ax.plot(line_x, y_pred, color='red', linewidth=lw,
                       label="%s, R² = %2f, y=%.2f x + %.2f" % ('OLS', lin2.score(XNucle, yNucle), lin2.coef_, lin2.intercept_))
    ax.legend(ncol=1, fontsize='xx-small', loc='upper left')
    print(lin1.coef_, 'x +', lin1.intercept_)
    print(lin2.coef_, 'x+ ', lin2.intercept_)
    outfile='1plot_OLS_GFF_Nucle_'+month+'2022.png'
    fig.savefig(outfile, format='png')
    plt.show()


fig, axs = plt.subplots(nrows=2, ncols=2)
fig.suptitle(survey, fontsize=fontsize)
line_x=np.array([np.min(data["Turbidity GFF"]) , np.max(data["Turbidity GFF"])]) #commun à tous, axe de turbidité max
# for name, estimator in estimators:
#     #t0 = time.time()
#     estimator.fit(XGFF, yGFF)
#     #elapsed_time = time.time() - t0
#     y_pred = estimator.predict(line_x.reshape(2, 1))
#     plt.plot(
#         line_x,
#         y_pred,
#         color=colors[name],
#         linewidth=lw,
#         label="%s, R² = %2f" % (name, estimator.score(XGFF,yGFF)) # (fit time: %.2fs)" % (name, elapsed_time),
#     )
#plt.legend(loc="upper left")

ax=axs[0, 0]
ax.set_xlabel('Turbidity', fontsize=fontsize)
ax.set_ylabel('Concentration (mg/L)', fontsize=fontsize)
ax.set_title('GFF', fontsize=fontsize)
#colors  = {"OLS": "gold", "Theil-Sen": "lightgreen"}#, "RANSAC": "lightgreen"}
p2=ax.scatter(data["Turbidity GFF"], data["Concentration GFF"], alpha=0.8, marker='+', color='blue', label='GFF')

lin=LinearRegression().fit(XGFF, yGFF)
y_pred = lin.predict(line_x.reshape(2, 1))
OLSGFF=ax.plot(line_x,y_pred, color=colors["OLS"],linewidth=lw,
                  label="%s, R² = %2f" % ('OLS', lin.score(XGFF,yGFF)))


Theil=TheilSenRegressor().fit(XGFF, yGFF)
y_pred = Theil.predict(line_x.reshape(2, 1))
TSGFF=ax.plot(line_x,y_pred, color=colors["Theil-Sen"],linewidth=lw,
                  label="%s, R² = %2f" % ('Theil-Sen', Theil.score(XGFF,yGFF)))
ax.legend(ncol=1, fontsize='xx-small', loc='upper left')





#fig,ax =plt.subplots()
#2.Que Nucle
ax=axs[0, 1]
#ax.set_xlabel('Turbidity', fontsize=fontsize)
ax.set_ylabel('Concentration (mg/L)', fontsize=fontsize)
ax.set_title('Nuclepore', fontsize=fontsize)
#colors  = {"OLS": "gold", "Theil-Sen": "lightgreen"}#, "RANSAC": "lightgreen"}
p2=ax.scatter(data["Turbidity Nuclepore"], data["Concentration Nuclepore"], alpha=0.8, marker='.', color='red', label='Nuclepore')

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
ax.legend(ncol=1, fontsize='xx-small', loc='upper left')
#fig.tight_layout()

#3. Que OLS
ax=axs[1, 0]
p1=ax.scatter(data["Turbidity GFF"], data["Concentration GFF"], alpha=0.8, marker='x', color='blue', label='GFF')
p2=ax.scatter(data["Turbidity Nuclepore"], data["Concentration Nuclepore"], alpha=0.8, marker='.', color='red', label='Nuclepore')

ax.set_xlabel('Turbidity', fontsize=fontsize)
ax.set_ylabel('Concentration (mg/L)', fontsize=fontsize)
ax.set_title('OLS', fontsize=fontsize)
#ax.legend(loc='lower right')


lin1=LinearRegression().fit(XGFF,yGFF)
y_pred = lin1.predict(line_x.reshape(2, 1))
OLSGFF=ax.plot(line_x,y_pred, color='blue',linewidth=lw,
                  label="%s, R² = %2f" % ('OLS', lin1.score(XGFF,yGFF)))

lin2=LinearRegression().fit(XNucle, yNucle)
y_pred = lin2.predict(line_x.reshape(2, 1))
OLSNucle=ax.plot(line_x,y_pred, color='red',linewidth=lw,
                  label="%s, R² = %2f" % ('OLS', lin2.score(XNucle,yNucle)))
ax.legend(ncol=1, fontsize='xx-small', loc='upper left')

#3. Que TS
ax=axs[1, 1]
p1=ax.scatter(data["Turbidity GFF"], data["Concentration GFF"], alpha=0.8, marker='x', color='blue', label='GFF')
p2=ax.scatter(data["Turbidity Nuclepore"], data["Concentration Nuclepore"], alpha=0.8, marker='.', color='red', label='Nuclepore')

ax.set_xlabel('Turbidity', fontsize=fontsize)
ax.set_ylabel('Concentration (mg/L)', fontsize=fontsize)
ax.set_title('Theil-Sen', fontsize=fontsize)
#ax.legend(loc='lower right')


TS1=TheilSenRegressor().fit(XGFF,yGFF)
y_pred = TS1.predict(line_x.reshape(2, 1))
TSGFF=ax.plot(line_x,y_pred, color='blue',linewidth=lw,
                 label="%s, R² = %2f" % ('Theil-Sen', TS1.score(XGFF,yGFF)))

TS2=TheilSenRegressor().fit(XNucle, yNucle)
y_pred = TS2.predict(line_x.reshape(2, 1))
TSNucle=ax.plot(line_x,y_pred, color='red',linewidth=lw,
                 label="%s, R² = %2f" % ('Theil-Sen', TS2.score(XNucle,yNucle)))
ax.legend(ncol=1, fontsize='xx-small', loc='upper left')


#print(data)

#xGFF=data["Turbidity GFF"]
#XGFF = xGFF[:, np.newaxis]
#yGFF=data["Concentration GFF"]
#yGFF2=yGFF[:,np.newaxis]

#xNucle=data["Turbidity Nuclepore"]
#xNucle=xNucle[np.logical_not(np.isnan(xNucle))]
#XNucle = xNucle[:, np.newaxis]
#yNucle=data["Concentration Nuclepore"]
#yNucle=yNucle[np.logical_not(np.isnan(yNucle))]
#print('xnucle', XNucle)


#reg = LinearRegression().fit(XGFF, yGFF)
#print(reg.score(XGFF, yGFF))
#print(reg.coef_)
#print(reg.intercept_)
#print(reg.get_params)
#reg.predict(np.array([[3, 5]]))


#fig, ax = plt.subplots()
#p1=ax.scatter(data["Turbidity GFF"], data["Concentration GFF"], alpha=0.8, marker='x', color='green', label='GFF')
#ax.set_xlabel('Turbidity', fontsize=15)
#ax.set_ylabel('Concentration (mg/L)', fontsize=15)
#ax.set_title('Turbidity-concentration 2-5 June')
#ax.legend(loc='lower right')

#line_x=np.array([np.min(data["Turbidity GFF"]) , np.max(data["Turbidity GFF"])])
#for name, estimator in estimators:
    #t0 = time.time()
 #   estimator.fit(XGFF, yGFF)
    #elapsed_time = time.time() - t0
 #   y_pred = estimator.predict(line_x.reshape(2, 1))
 #   plt.plot(
  #      line_x,
  #      y_pred,
   #     color=colors[name],
   #     linewidth=lw,
   #     label="%s, R² = %2f" % (name, estimator.score(XGFF,yGFF)) # (fit time: %.2fs)" % (name, elapsed_time),
   # )
#l1= ax.legend(loc="upper left")
plt.subplots_adjust(left=0.11,
                    bottom=0.1,
                    right=0.95,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

plt.show()
outfile='4subplot_TS_OLS_GFF_Nucle_relation_survey'+month+'2022.png'
fig.savefig(outfile, format='png')
sys.exit(1)

############################################################################################"

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