# 01/09/23 : Plot of Radium etc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys,os
import seaborn as sns
import math
from scipy import stats

rep = '/home/penicaud/Documents/Data/Géochimie/'
file = rep + 'resume_1st_2d_counting.xlsx'
data = pd.read_excel(file, skiprows= 26, nrows=8, usecols= list(range(30)))

lambda_223 = 0.0608
lambda_224 = 0.189

# Figure parameter
fontsize = 5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


dict_ra = {'Ra 224 ex' : {'max': 120},
           'Ra 226': {'max': 50},
           'Ra 223' : {'max': 10},
           'Ra 228' : {'max': 150},
           'Ra224ex/Ra223' : {'max': 30} ,
           'Ra224ex/Ra226' : {'max': 4} ,
           'Ra224ex/Ra228' : {'max': 1.6}}
# Plot parameter
transect='T1'
selected = data[(data['Transect'] == transect)]
list_ra = ['Ra 224 ex', 'Ra 223', 'Ra 226', 'Ra 228']
i = 0
fig, axs = plt.subplots(figsize = (5,5), nrows=len(list_ra), ncols = 2)
#fig.suptitle('Discharge at Trung Trang ' + a + str(month) + '/' + str(year))
for r in list_ra:
    ax = axs[i, 0]
    ax.grid(True, alpha=0.5, linewidth=0.5)
    ax.set_xlim(0,30)
    ax.tick_params(axis='both', which='both', labelsize=fontsize,  width=0.5)
    rounded_value = math.ceil(np.max(selected[r]) / 10) * 10
    ax.set_ylim(0, dict_ra[r]['max'])
    if i == len(list_ra) :
        ax.set_xlabel('Salinity (PSU)', fontsize=fontsize)
    else :
        ax.set_xticklabels('')
    ax.set_ylabel(r+'\n(dpm 100$L^{-1}$)', fontsize=fontsize)
    y_err = selected[r] * (selected['Err '+r])
    ax.errorbar(selected['Salinité'], selected[r], yerr=y_err, fmt='.', markersize=2, elinewidth=0.5,
                color=sns.color_palette("colorblind")[i], linewidth=0.5)
    #ax.scatter(selected['Salinité'], selected[r], color = sns.color_palette("colorblind")[i], marker = 'x',
    #           s=10, linewidths=0.5)
    i = i+1

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)  # Adjust the linewidth as needed

i=0
for r in list_ra:
    ax = axs[i, 1]
    ax.grid(True, alpha=0.5, linewidth=0.5)
    ax.set_xlim(-0.5, 8)
    ax.tick_params(axis='both', labelsize=fontsize)
    rounded_value = math.ceil(np.max(selected[r]) / 10) * 10
    ax.set_ylim(0, dict_ra[r]['max'])
    ax.tick_params(axis='both', which='both', labelsize=fontsize,  width=0.5)
    if i == len(list_ra):
        ax.set_xlabel('Distance (km)', fontsize=fontsize)
    else :
        ax.set_xticklabels('')
    #ax.set_ylabel(r+' (dpm 100$L^{-1}$)', fontsize=fontsize)
    # ax.scatter(selected['Distance pt à pt'], selected[r], color = sns.color_palette("colorblind")[i], marker = 'o',
    #            s=10, linewidths=0.5)
    y_err = selected[r] * (selected['Err '+r])
    ax.errorbar(selected['Distance pt à pt'], selected[r], yerr=y_err, fmt='.', markersize=2, elinewidth=0.5,
                color=sns.color_palette("colorblind")[i], linewidth=0.5)
    i = i+1

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)  # Adjust the linewidth as needed
plt.tight_layout()
plt.show()
fig.savefig('Geochimie_Ra_vs_salinity')


# PLOT rapp  ####################################################################################################
list_ra = ['Ra224ex/Ra223', 'Ra224ex/Ra226', 'Ra224ex/Ra228']
i = 0
fig, axs = plt.subplots(figsize = (5,5), nrows=len(list_ra), ncols = 2)
#fig.suptitle('Discharge at Trung Trang ' + a + str(month) + '/' + str(year))
for r in list_ra:
    ax = axs[i, 0]
    ax.grid(True, alpha=0.5, linewidth=0.5)
    ax.set_xlim(0,30)
    ax.tick_params(axis='both', which='both', labelsize=fontsize,  width=0.5)
    rounded_value = math.ceil(np.max(selected[r]) / 10) * 10
    ax.set_ylim(0, dict_ra[r]['max'])
    if i+1 == len(list_ra) :
        ax.set_xlabel('Salinity (PSU)', fontsize=fontsize)
    else :
        ax.set_xticklabels('')
    ax.set_ylabel(r+'\n(dpm 100$L^{-1}$)', fontsize=fontsize)
    #y_err = selected[r] * (selected['Err '+r])
    #ax.errorbar(selected['Salinité'], selected[r], yerr=y_err, fmt='.', markersize=2, elinewidth=0.5,
    #            color=sns.color_palette("colorblind")[i], linewidth=0.5)
    ax.scatter(selected['Salinité'], selected[r], color = sns.color_palette("colorblind")[i], marker = 'x',
               s=10, linewidths=0.5)
    i = i+1

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)  # Adjust the linewidth as needed

i=0
for r in list_ra:
    ax = axs[i, 1]
    ax.grid(True, alpha=0.5, linewidth=0.5)
    ax.set_xlim(-0.5, 8)
    ax.tick_params(axis='both', labelsize=fontsize)
    rounded_value = math.ceil(np.max(selected[r]) / 10) * 10
    ax.set_ylim(0, dict_ra[r]['max'])
    ax.tick_params(axis='both', which='both', labelsize=fontsize,  width=0.5)
    if i+1 == len(list_ra):
        ax.set_xlabel('Distance (km)', fontsize=fontsize)
    else :
        ax.set_xticklabels('')
    #ax.set_ylabel(r+' (dpm 100$L^{-1}$)', fontsize=fontsize)
    ax.scatter(selected['Distance pt à pt'], selected[r], color = sns.color_palette("colorblind")[i], marker = 'x',
                s=10, linewidths=0.5)
    #y_err = selected[r] * (selected['Err '+r])
    #ax.errorbar(selected['Distance pt à pt'], selected[r], yerr=y_err, fmt='.', markersize=2, elinewidth=0.5,
    #            color=sns.color_palette("colorblind")[i], linewidth=0.5)
    i = i+1

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)  # Adjust the linewidth as needed
plt.tight_layout()
plt.show()
fig.savefig('Geochimie_rapport_Ra_vs_salinity')



# Just apparent age #####################################################################""
fig, axs = plt.subplots(figsize=(5, 5), ncols=2)
ax = axs[0]
ax.grid(True, alpha=0.5, linewidth=0.5)
ax.tick_params(axis='both', which='both', labelsize=fontsize, width=0.5)
ax.set_xlabel('Salinity (PSU)', fontsize=fontsize)
ax.set_ylabel('Apparent age (days)', fontsize=fontsize)
ax.scatter(selected['Salinité'], selected['t'], color=sns.color_palette("colorblind")[5], marker='x',
           s=10, linewidths=0.5)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)  # Adjust the linewidth as needed

ax = axs[1]
ax.grid(True, alpha=0.5, linewidth=0.5)
ax.tick_params(axis='both', which='both', labelsize=fontsize, width=0.5)
ax.set_xlabel('Distance (km)', fontsize=fontsize)
ax.set_ylabel('')
#ax.set_ylabel('Apparent age (days)', fontsize=fontsize)
ax.scatter(selected['Distance pt à pt'], selected['t'], color=sns.color_palette("colorblind")[5], marker='x',
           s=10, linewidths=0.5)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)  # Adjust the linewidth as needed

plt.tight_layout()
plt.show()
fig.savefig('Geochimie_T1_apparentage')


# Just apparent age #####################################################################""
fig, ax = plt.subplots(figsize=(5, 5))
ax.grid(True, alpha=0.5, linewidth=0.5)
ax.tick_params(axis='both', which='both', labelsize=fontsize, width=0.5)
ax.set_xlabel('Apparent age (days)', fontsize=fontsize)
ax.set_ylabel('Distance to the mouth (km)', fontsize=fontsize)

x= selected['t']
y = selected['Distance pt à pt']
ax.scatter(x, y, color=sns.color_palette("colorblind")[5], marker='x',
           s=15, linewidths=1)
#slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
slope = np.dot(x, y) / np.dot(x, x)
regression_line = slope * x #+ intercept
r_value, _ = stats.pearsonr(regression_line, y)
ax.plot(x, regression_line, color='grey',  label='Linear Regression Fit', linestyle = '--', linewidth = 0.8)
r_text = f'R-value: {r_value:.2f}'
ax.annotate(r_text, xy=(0.02, 0.95), xycoords='axes fraction', fontsize=fontsize, color='grey')
for spine in ax.spines.values():
    spine.set_linewidth(0.5)  # Adjust the linewidth as needed
plt.show()
fig.savefig('Geochimie_T1_apparentage_velocity')


print('echo')