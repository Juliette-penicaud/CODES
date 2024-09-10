from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

print('Beginning')
#########
path = '/home/penicaud/Documents/Article/Article_hydro_survey/Figures/'
Image1 = Image.open(path + 'illustration_transects_v2.jpeg')


###########
path = '/home/penicaud/Documents/Article/Article_hydro_survey/Figures/'
Image1 = Image.open(path + 'illustration_transects_v2.jpeg')
Image2 = Image.open(path + 'ADCP_06_T2_sens_vel_filtered_pcolor.png')
Image3 = Image.open(path + 'ADCP_06_T4_sens_vel_filtered.png')

F = 10
f = 8
x_text = 100 # x = -100 pour min_max_data et amplitude
y_text = 200
x_text = 50
y_text = 100
x_text2 = 3000

val = 1. # 0.42 pour val tempo
val2 = 0.45 # Attenuation vs discharge : 0.4

fig = plt.figure(figsize=(4,6)) # 6,4 pour attenuation, amplitude : (3.5, 3)
ax1 = fig.add_axes([0, 2*val2, val, val])  # [left, bottom, width, height]
ax2 = fig.add_axes([0, val2, val, val])
ax3 = fig.add_axes([0, 0, val, val])

ax1.imshow(Image1, zorder=0.5)
ax1.axis('off')
ax1.text(x_text, y_text, "(a)", fontsize=f, color='k')
#ax1.text(x_text2, y_text, "FSJ, 18 june 2022 (1954 m³/s)", fontsize=F, zorder=1, color = 'k' ,
#         horizontalalignment='center')

ax2.imshow(Image2)
ax2.axis('off')
ax2.text(x_text, y_text, "(b)", fontsize=f, color='k')
#ax2.text(x_text2, y_text, "FSA, 12 august 2022 (1577 m³/s)", fontsize=F, zorder=1, color = 'k',
#         horizontalalignment='center')

ax3.imshow(Image3)
ax3.axis('off')
ax3.text(x_text, y_text, "(c)", fontsize=f, color='k')
#ax3.text(x_text2, y_text, "FSO, 5 october 2022 (691 m³/s)", fontsize=F, zorder=1, color = 'k',
#         horizontalalignment='center')

plt.subplots_adjust(wspace=0, hspace=0)#, left=0.01, bottom=0.04, right=0.99, top=0.96)

path_save = path
#plt.savefig(path_save + 'Combine_temporal_evolution.png', dpi=300, bbox_inches='tight', pad_inches=.1)
#plt.savefig(path_save + 'Combine_evolution_simpson_fixe.png', dpi=300, bbox_inches='tight', pad_inches=.1)
plt.savefig(path_save + 'Combine_T2_T4_velocity.png', dpi=300, bbox_inches='tight', pad_inches=.1)



########"
path = '/home/penicaud/Documents/Article/Article_hydro_survey/Figures/'
Image1 = Image.open(path + 'TJ1_S_plot.png')
Image2 = Image.open(path + 'TA2_S_plot.png')

F = 10
f = 8
x_text = 0 # x = -100 pour min_max_data et amplitude
y_text = 0

val = 1 # 0.42 pour val tempo
val2 = 0.23

fig = plt.figure(figsize=(4,6)) # 6,4 pour attenuation, amplitude : (3.5, 3)
ax1 = fig.add_axes([0, val2, val, val])  # [left, bottom, width, height]
ax2 = fig.add_axes([0, 0, val, val])

ax1.imshow(Image1, zorder=0.5)
ax1.axis('off')
ax1.text(-20, -20, "(a)", fontsize=f, color='k')
ax1.text(300, 280, "Distance (m)", fontsize=f, color='k', zorder=1)
ax1.text(-20, 150, "Depth (m)", fontsize=f, color='k', rotation='vertical')
ax1.text(40, 30, "TJ1", fontsize=f, color='white')

ax2.imshow(Image2)
ax2.axis('off')
ax2.text(-20, 40, "(b)", fontsize=f, color='k')
ax2.text(300, 350, "Distance (m)", fontsize=f, color='k')
ax2.text(-20, 220, "Depth (m)", fontsize=f, color='k', rotation='vertical')
ax2.text(40, 100, "TA2", fontsize=f, color='white')

plt.subplots_adjust(wspace=0, hspace=0)#, left=0.01, bottom=0.04, right=0.99, top=0.96)

path_save = path
plt.savefig(path_save + 'Combine_transect_salinity.png', dpi=300, bbox_inches='tight', pad_inches=.1)

########"
path = '/home/penicaud/Documents/Article/Article_hydro_survey/Figures/'
Image1 = Image.open(path + 'Salinity_vertical_evolution_SFJ_18062022.png')
Image2 = Image.open(path + 'Salinity_vertical_evolution_SF_24_12-13082022.png')
Image3 = Image.open(path + 'Salinity_vertical_evolution_SF_24_4-5102022.png')

F = 10
f = 6
x_text = 0 # x = -100 pour min_max_data et amplitude
y_text = 0

val = 0.35 # 0.42 pour val tempo
val2 = 0.35

fig = plt.figure(figsize=(4,6)) # 6,4 pour attenuation, amplitude : (3.5, 3)
ax1 = fig.add_axes([0, 0, val, val])  # [left, bottom, width, height]
ax2 = fig.add_axes([val2, 0, val, val])
ax3 = fig.add_axes([2*val2, 0, val, val])

ax1.imshow(Image1, zorder=0.5)
ax1.axis('off')
ax1.text(x_text, y_text, "(a)", fontsize=f, color='k')

ax2.imshow(Image2)
ax2.axis('off')
ax2.text(x_text, y_text, "(b)", fontsize=f, color='k')

ax3.imshow(Image3)
ax3.axis('off')
ax3.text(x_text, y_text, "(c)", fontsize=f, color='k')

plt.subplots_adjust(wspace=0, hspace=0)#, left=0.01, bottom=0.04, right=0.99, top=0.96)

path_save = path
plt.savefig(path_save + 'Combine_temporal_evolution_salinity.png', dpi=300, bbox_inches='tight', pad_inches=.1)

#########"
path = '/home/penicaud/Documents/Article/Article_hydro_survey/Figures/'
Image1 = Image.open(path + 'Temporal_salinity_velocity_evolution_fixeJune_layer_2m.png')
Image2 = Image.open(path + 'Temporal_salinity_velocity_evolution_fixeAugust_layer_2m.png')
Image3 = Image.open(path + 'Temporal_salinity_velocity_evolution_fixeOctobre_layer_2m.png')

Image1 = Image.open(path + 'Simpson_fixe_station_June_0.6depth.png')
Image2 = Image.open(path + 'Simpson_fixe_station_August_0.6depth.png')
Image3 = Image.open(path + 'Simpson_fixe_station_Octobre_0.6depth.png')

Image1 = Image.open(path + 'Simpson_log_filtered_June_0.7depth.png')
Image2 = Image.open(path + 'Simpson_log_filtered_August_0.7depth.png')
Image3 = Image.open(path + 'Simpson_log_filtered_Octobre_0.7depth.png')

F = 10
f = 8
x_text = 100 # x = -100 pour min_max_data et amplitude
y_text = 200
x_text = 50
y_text = 100
x_text2 = 3000

val = 1. # 0.42 pour val tempo
val2 = 0.45 # Attenuation vs discharge : 0.4

fig = plt.figure(figsize=(4,6)) # 6,4 pour attenuation, amplitude : (3.5, 3)
ax1 = fig.add_axes([0, 2*val2, val, val])  # [left, bottom, width, height]
ax2 = fig.add_axes([0, val2, val, val])
ax3 = fig.add_axes([0, 0, val, val])

ax1.imshow(Image1, zorder=0.5)
ax1.axis('off')
ax1.text(x_text, y_text, "(a)", fontsize=f, color='k')
#ax1.text(x_text2, y_text, "FSJ, 18 june 2022 (1954 m³/s)", fontsize=F, zorder=1, color = 'k' ,
#         horizontalalignment='center')

ax2.imshow(Image2)
ax2.axis('off')
ax2.text(x_text, y_text, "(b)", fontsize=f, color='k')
#ax2.text(x_text2, y_text, "FSA, 12 august 2022 (1577 m³/s)", fontsize=F, zorder=1, color = 'k',
#         horizontalalignment='center')

ax3.imshow(Image3)
ax3.axis('off')
ax3.text(x_text, y_text, "(c)", fontsize=f, color='k')
#ax3.text(x_text2, y_text, "FSO, 5 october 2022 (691 m³/s)", fontsize=F, zorder=1, color = 'k',
#         horizontalalignment='center')

plt.subplots_adjust(wspace=0, hspace=0)#, left=0.01, bottom=0.04, right=0.99, top=0.96)

path_save = path
#plt.savefig(path_save + 'Combine_temporal_evolution.png', dpi=300, bbox_inches='tight', pad_inches=.1)
#plt.savefig(path_save + 'Combine_evolution_simpson_fixe.png', dpi=300, bbox_inches='tight', pad_inches=.1)
plt.savefig(path_save + 'Combine_evolution_simpson_transect.png', dpi=300, bbox_inches='tight', pad_inches=.1)

########
path = '/home/penicaud/Documents/These/Figures/Chapitre méthode modèle/'
Image1 = Image.open(path + 'Bathy_stations_bathy_corrigee_Juliette_v3(1).png')
Image2 = Image.open(path + 'Bathy_stations_bathy_corrigee_Juliette_bathyAlexeisurGEBCO_16052023(1).png')

F = 6
f = 6
x_text = 0 # x = -100 pour min_max_data et amplitude
y_text = 200

val = 0.42 # 0.42 pour val tempo
val2 = 0.4 # Attenuation vs discharge : 0.4

fig = plt.figure(figsize=(6,4)) # 6,4 pour attenuation, amplitude : (3.5, 3)
ax1 = fig.add_axes([0, val2, val, val])  # [left, bottom, width, height]
ax2 = fig.add_axes([val2, val2,val, val])

ax1.imshow(Image1, zorder=0.5)
ax1.axis('off')
ax1.text(x_text, y_text, "(a)", fontsize=f, zorder=1, )

ax2.imshow(Image2)
ax2.axis('off')
ax2.text(x_text, y_text, "(b)", fontsize=f, color='k')

plt.subplots_adjust(wspace=0.01, hspace=0.01)#, left=0.01, bottom=0.04, right=0.99, top=0.96)

path_save = path
plt.savefig(path_save + 'Combine_ex_bathy.png', dpi=300, bbox_inches='tight', pad_inches=.1)

##########
path = '/home/penicaud/Documents/These/Figures/Chapitre méthode modèle/'
Image1 = Image.open(path + 'Diff_Bathy_Grande_carte.png')
Image2 = Image.open(path + 'Diff_Bathy_petite_carte.png')
Image3 = Image.open(path + 'Bathy_1an_h3m5_debit-4j_Grande_carte.png')
Image4 = Image.open(path + 'Bathy_1an_h3m5_debit-4j_petite_carte.png')

F = 6
f = 6
x_text = 0 # x = -100 pour min_max_data et amplitude
y_text = 200

val = 0.42 # 0.42 pour val tempo
val2 = 0.4 # Attenuation vs discharge : 0.4

fig = plt.figure(figsize=(6,4)) # 6,4 pour attenuation, amplitude : (3.5, 3)
ax1 = fig.add_axes([0, 2*val2, val, val])  # [left, bottom, width, height]
ax2 = fig.add_axes([val2, 2*val2,val, val])
ax3 = fig.add_axes([0, val2, val, val])
ax4 = fig.add_axes([val2, val2,val, val])  # [left, bottom, width, height]

ax1.imshow(Image1, zorder=0.5)
ax1.axis('off')
ax1.text(x_text, y_text, "(a)", fontsize=f, zorder=1, )

ax2.imshow(Image2)
ax2.axis('off')
ax2.text(x_text, y_text, "(b)", fontsize=f, color='k')

ax3.imshow(Image3)
ax3.axis('off')
ax3.text(x_text, y_text, "(c)", fontsize=f)

ax4.imshow(Image4)
ax4.axis('off')
ax4.text(x_text, y_text, "(d)", fontsize=f, color='k')

plt.subplots_adjust(wspace=0.01, hspace=0.01)#, left=0.01, bottom=0.04, right=0.99, top=0.96)

path_save = path
plt.savefig(path_save + 'Combine_diff_gebco_bathy_finale_4sub.png', dpi=300, bbox_inches='tight', pad_inches=.1)


###########
path = '/home/penicaud/Documents/Modèles/Validation_config/1an_h3m5_debit-4j/new/'
var = "S1"
if var == "serie_tempo":
    Image1 = Image.open(path + "AUG_serie_tempo_sal_vel.png")
    Image2 = Image.open(path + "SEPT_serie_tempo_sal_vel.png")
    Image3 = Image.open(path + "DECS_serie_tempo_sal_vel.png")
    Image4 = Image.open(path + "DECN_serie_tempo_sal_vel.png")
elif var== "S1":
    Image1 = Image.open(path + 'S1_AUG_comp_mod_obs.png')
    Image2 = Image.open(path + 'S1_SEPT_comp_mod_obs.png')
    Image3 = Image.open(path + 'S1_DECS_comp_mod_obs.png')
    Image4 = Image.open(path + 'S1_DECN_comp_mod_obs.png')
elif var =='S2':
    Image1 = Image.open(path + 'S2_AUG_comp_mod_obs.png')
    Image2 = Image.open(path + 'S2_SEPT_comp_mod_obs.png')
    Image3 = Image.open(path + 'S2_DECS_comp_mod_obs.png')
    Image4 = Image.open(path + 'S2_DECN_comp_mod_obs.png')
elif var =='S3':
    Image1 = Image.open(path + 'S3_AUG_comp_mod_obs.png')
    Image2 = Image.open(path + 'S3_SEPT_comp_mod_obs.png')
    Image3 = Image.open(path + 'S3_DECS_comp_mod_obs.png')
    Image4 = Image.open(path + 'S3_DECN_comp_mod_obs.png')

f=6
x_text = -30
y_text = 300
val = 0.40 # 0.42 pour val tempo
val2 = 0.4 # Attenuation vs discharge : 0.4

fig = plt.figure(figsize=(6,4)) # 6,4 pour attenuation, amplitude : (3.5, 3)
ax1 = fig.add_axes([0, val2, val, val])  # [left, bottom, width, height]
ax2 = fig.add_axes([val2, val2,val, val])
ax3 = fig.add_axes([0, 0, val, val])
ax4 = fig.add_axes([val2, 0, val, val])

F = 6
f = 6
x_text = 0 # x = -100 pour min_max_data et amplitude
y_text = 100

ax1.imshow(Image1)
ax1.axis('off')
ax1.text(x_text, y_text, "(a)", fontsize=f)

ax2.imshow(Image2)
ax2.axis('off')
ax2.text(x_text, y_text, "(b)", fontsize=f)

ax3.imshow(Image3)
ax3.axis('off')
ax3.text(x_text, y_text, "(c)", fontsize=f)

ax4.imshow(Image4)
ax4.axis('off')
ax4.text(x_text, y_text, "(d)", fontsize=f)

plt.subplots_adjust(wspace=0, hspace=0)#, left=0.01, bottom=0.04, right=0.99, top=0.96)

path_save = '/home/penicaud/Documents/These/Figures/Chapitre méthode modèle/'
out = path_save + '4subplots_val_' + var
plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=.1)

###########"
path = '/home/penicaud/Documents/These/Figures/Chapitre méthode modèle/'
Image1 = Image.open(path + 'deformation_grille_initiale_dx.png')
Image2 = Image.open(path + 'deformation_grille_initiale_dy.png')
Image1 = Image.open(path + 'deformation_grille_dx.png')
Image2 = Image.open(path + 'deformation_grille_dy.png')
Image1 = Image.open(path + 'deformation_grille_initiale_dy.png')
Image2 = Image.open(path + 'deformation_grille_dy.png')

Image1 = Image.open(path + 'deformation_grille_finale_ij_dx.png')
Image2 = Image.open(path + 'deformation_grille_finale_ij_dy.png')

F = 6
f = 6
x_text = 0 # x = -100 pour min_max_data et amplitude
y_text = 100

fig = plt.figure(figsize=(6, 3))  # 6,4 pour attenuation, amplitude : (3.5, 3)
ax1 = fig.add_axes([0.0, 0, 0.55, 0.55])  # [left, bottom, width, height]
ax2 = fig.add_axes([0.38, 0, 0.55, 0.55])

#fig, ax = plt.subplots(1, 2, figsize=(6,3)) # figsize=(4,6) pour amplitudeTTvsHD
ax1.imshow(Image1)
ax1.axis('off')
ax1.text(x_text, y_text, "(a)", fontsize=f)

ax2.imshow(Image2)
ax2.axis('off')
ax2.text(x_text, y_text, "(b)", fontsize=f)

plt.subplots_adjust(wspace=0, hspace=0)#, left=0.01, bottom=0.04, right=0.99, top=0.96)

path_save = path
plt.savefig(path_save + 'Combine_fig_initiale_grid_vanuc.png', dpi=300, bbox_inches='tight', pad_inches=.1)
plt.savefig(path_save + 'Combine_fig_grid_vanuc.png', dpi=300, bbox_inches='tight', pad_inches=.1)
plt.savefig(path_save + 'Combine_fig_grid_dy_vanuc_deformation.png', dpi=300, bbox_inches='tight', pad_inches=.1)
plt.savefig(path_save + 'Combine_fig_grid_vanuc_deformation_finale.png', dpi=300, bbox_inches='tight', pad_inches=.1)


############
# Combine data water level and harmonic tide prediction spring tides
Image1 = Image.open('Spring_tide_per_year_HD_0.9_and_0.95_0.99.png')
Image2 = Image.open('Spring_tide_per_year_HD_from_harmonic_0.9_and_0.95_0.99.png')

F = 6
f = 6
x_text = 0 # x = -100 pour min_max_data et amplitude
y_text = 0

fig, ax = plt.subplots(1, 2, figsize=(6,3)) # figsize=(4,6) pour amplitudeTTvsHD
ax[0].imshow(Image1)
ax[0].axis('off')
ax[0].text(x_text, y_text, "(a)", fontsize=f)

ax[1].imshow(Image2)
ax[1].axis('off')
ax[1].text(x_text, y_text, "(b)", fontsize=f)

plt.subplots_adjust( wspace=0.01, hspace=0.01)#, left=0.01, bottom=0.04, right=0.99, top=0.96)

path_save = ''
plt.savefig(path_save + 'Combine_fig_springtides_2008-2022.png', dpi=300, bbox_inches='tight', pad_inches=.1)

#####################
# 3*3 subplots :
Image1 = Image.open('Attenuation_vs_discharge_to_combine_v2_2015.png')
Image2 = Image.open('Attenuation_vs_discharge_to_combine_v2_2016.png')
Image3 = Image.open('Attenuation_vs_discharge_to_combine_v2_2017.png')
Image4 = Image.open('Attenuation_vs_discharge_to_combine_v2_2018.png')
Image5 = Image.open('Attenuation_vs_discharge_to_combine_v2_2019.png')
Image6 = Image.open('Attenuation_vs_discharge_to_combine_v2_2020.png')
Image7 = Image.open('Attenuation_vs_discharge_to_combine_v2_2021.png')
Image8 = Image.open('Attenuation_vs_discharge_to_combine_v2_2022.png')
Image9 = Image.open('Attenuation_vs_discharge_all_years.png')

Image1 = Image.open('Attenuation_vs_TR_2015.png')
Image2 = Image.open('Attenuation_vs_TR_2016.png')
Image3 = Image.open('Attenuation_vs_TR_2017.png')
Image4 = Image.open('Attenuation_vs_TR_2018.png')
Image5 = Image.open('Attenuation_vs_TR_2019.png')
Image6 = Image.open('Attenuation_vs_TR_2020.png')
Image7 = Image.open('Attenuation_vs_TR_2021.png')
Image8 = Image.open('Attenuation_vs_TR_2022.png')
Image9 = Image.open('Attenuation_vs_TR_all_years.png')


Image1 = Image.open('Attenuation_vs_TR_filtered_2015.png')
Image2 = Image.open('Attenuation_vs_TR_filtered_2016.png')
Image3 = Image.open('Attenuation_vs_TR_filtered_2017.png')
Image4 = Image.open('Attenuation_vs_TR_filtered_2018.png')
Image5 = Image.open('Attenuation_vs_TR_filtered_2019.png')
Image6 = Image.open('Attenuation_vs_TR_filtered_2020.png')
Image7 = Image.open('Attenuation_vs_TR_filtered_2021.png')
Image8 = Image.open('Attenuation_vs_TR_filtered_2022.png')
Image9 = Image.open('Attenuation_vs_TR_filterd_all_years.png')

Image1 = Image.open('Amplitude_TT_vs_HD_Amplitude_2015_polyfitto_combinev2.png')
Image2 = Image.open('Amplitude_TT_vs_HD_Amplitude_2016_polyfitto_combinev2.png')
Image3 = Image.open('Amplitude_TT_vs_HD_Amplitude_2017_polyfitto_combinev2.png')
Image4 = Image.open('Amplitude_TT_vs_HD_Amplitude_2018_polyfitto_combinev2.png')
Image5 = Image.open('Amplitude_TT_vs_HD_Amplitude_2019_polyfitto_combinev2.png')
Image6 = Image.open('Amplitude_TT_vs_HD_Amplitude_2020_polyfitto_combinev2.png')
Image7 = Image.open('Amplitude_TT_vs_HD_Amplitude_2021_polyfitto_combinev2.png')
Image8 = Image.open('Amplitude_TT_vs_HD_Amplitude_2022_polyfitto_combinev2.png')
Image9= Image.open('Amplitude_TT_vs_HD_Amplitude_polyfit_all_year.png')

# Pour Attenuation vs discharge
f = 9
x_text = 300
y_text = 500
val = 0.42 # Attenuation vs discharge : 0.42
val2 = 0.4 # Attenuation vs discharge : 0.4

# Attenuation vs TR :
x_text = 0
y_text = 100
val = 0.432
val2 = 0.4
opt1 = True

# Amplitude TT vs HD
f=6
x_text = -30
y_text = 300
val = 0.48 # Attenuation vs discharge : 0.42
val2 = 0.4 # Attenuation vs discharge : 0.4

# Attenuation vs TR
f=8
x_text = 120
y_text = 100
val = 0.45 # Attenuation vs discharge : 0.42
val2 = 0.42
# Define a 3x3 grid (3 rows, 3 columns) with no space between subplots
#gs = GridSpec(3, 3, figure=fig, wspace=0, hspace=0)
# Add subplots to the grid
# ax1 = fig.add_subplot(gs[0, 0])
# ax4 = fig.add_subplot(gs[0, 1])
# ax7 = fig.add_subplot(gs[0, 2])
#
# ax2 = fig.add_subplot(gs[1, 0])
# ax5 = fig.add_subplot(gs[1, 1])
# ax8 = fig.add_subplot(gs[1, 2])
#
# ax3 = fig.add_subplot(gs[2, 0])
# ax6 = fig.add_subplot(gs[2, 1])
# ax9 = fig.add_subplot(gs[2, 2])

# OU :
# Create a figure with tight margins
if opt1 :
    fig = plt.figure(figsize=(6,4)) # 6,4 pour attenuation, amplitude : (3.5, 3)
    ax1 = fig.add_axes([0.0, 2*val2, val, val])  # [left, bottom, width, height]
    ax4 = fig.add_axes([val2, 2*val2,val, val])
    ax7 = fig.add_axes([2*val2, 2*val2, val, val])

    ax2 = fig.add_axes([0, val2, val, val])
    ax5 = fig.add_axes([val2,val2, val, val])
    ax8 = fig.add_axes([2*val2,val2, val, val])

    ax3 = fig.add_axes([0.0, 0,val, val])
    ax6 = fig.add_axes([val2, 0, val, val])  # Spanning more columns
    ax9 = fig.add_axes([2*val2, 0, val, val])
else :
    fig = plt.figure(figsize=(6, 4))
    gs = GridSpec(3, 3, figure=fig, wspace=0, hspace=0)
    # Add subplots to the grid
    ax1 = fig.add_subplot(gs[0, 0])
    ax4 = fig.add_subplot(gs[0, 1])
    ax7 = fig.add_subplot(gs[0, 2])

    ax2 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax8 = fig.add_subplot(gs[1, 2])

    ax3 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])

ax1.imshow(Image1)
ax1.axis('off')
ax1.text(x_text, y_text, "(a)", fontsize=f)

ax2.imshow(Image2)
ax2.axis('off')
ax2.text(x_text, y_text, "(b)", fontsize=f)

ax3.imshow(Image3)
ax3.axis('off')
ax3.text(x_text, y_text, "(c)", fontsize=f)

ax4.imshow(Image4)
ax4.axis('off')
ax4.text(x_text, y_text, "(d)", fontsize=f)

ax5.imshow(Image5)
ax5.axis('off')
ax5.text(x_text, y_text, "(e)", fontsize=f)

ax6.imshow(Image6)
ax6.axis('off')
ax6.text(x_text, y_text, "(f)", fontsize=f)

ax7.imshow(Image7)
ax7.axis('off')
ax7.text(x_text, y_text, "(g)", fontsize=f)

ax8.imshow(Image8)
ax8.axis('off')
ax8.text(x_text, y_text, "(h)", fontsize=f)

ax9.imshow(Image9)
ax9.axis('off')
ax9.text(x_text, y_text, "(i)", fontsize=f)

#plt.subplots_adjust( wspace=0.01, hspace=0.01)#, left=0.01, bottom=0.04, right=0.99, top=0.96)

path_save = ''
#plt.savefig(path_save + 'Combine_fig_attenuation_3x3_2015-2022.png', dpi=300, bbox_inches='tight', pad_inches=.1)
#plt.savefig(path_save + 'Combine_fig_attenuation_TR_3x3_2015-2022.png', dpi=300, bbox_inches='tight', pad_inches=.1)
plt.savefig(path_save + 'Combine_fig_amplitudeTTvsHD_3x3_2015-2022.png', dpi=300, bbox_inches='tight', pad_inches=.1)
plt.savefig(path_save + 'Combine_fig_attenuation_TR_3x3_2015-2022_filtered.png', dpi=300, bbox_inches='tight', pad_inches=.1)

############################
# 4 * 2
# COmbine figure amplitude TT vs HD from data
rep = ''

Image1 = Image.open('daily_discharge_tocombine_2015.png')
Image2 = Image.open('daily_discharge_tocombine_2016.png')
Image3 = Image.open('daily_discharge_tocombine_2017.png')
Image4 = Image.open('daily_discharge_tocombine_2018.png')
Image5 = Image.open('daily_discharge_tocombine_2019.png')
Image6 = Image.open('daily_discharge_tocombine_2020.png')
Image7 = Image.open('daily_discharge_tocombine_2021.png')
Image8 = Image.open('daily_discharge_tocombine_2022.png')

Image1 = Image.open('Attenuation_vs_discharge_to_combine_v2_2015.png')
Image2 = Image.open('Attenuation_vs_discharge_to_combine_v2_2016.png')
Image3 = Image.open('Attenuation_vs_discharge_to_combine_v2_2017.png')
Image4 = Image.open('Attenuation_vs_discharge_to_combine_v2_2018.png')
Image5 = Image.open('Attenuation_vs_discharge_to_combine_v2_2019.png')
Image6 = Image.open('Attenuation_vs_discharge_to_combine_v2_2020.png')
Image7 = Image.open('Attenuation_vs_discharge_to_combine_v2_2021.png')
Image8 = Image.open('Attenuation_vs_discharge_to_combine_v2_2022.png')
Image9 = Image.open('Attenuation_vs_discharge_all_years.png')

Image1 = Image.open('Amplitude_TT_vs_HD_Amplitude_2015to_combine.png')
Image2 = Image.open('Amplitude_TT_vs_HD_Amplitude_2016to_combine.png')
Image3 = Image.open('Amplitude_TT_vs_HD_Amplitude_2017to_combine.png')
Image4 = Image.open('Amplitude_TT_vs_HD_Amplitude_2018to_combine.png')
Image5 = Image.open('Amplitude_TT_vs_HD_Amplitude_2019to_combine.png')
Image6 = Image.open('Amplitude_TT_vs_HD_Amplitude_2020to_combine.png')
Image7 = Image.open('Amplitude_TT_vs_HD_Amplitude_2021to_combine.png')
Image8 = Image.open('Amplitude_TT_vs_HD_Amplitude_2022to_combine.png')

Image1 = Image.open('Amplitude_TT_vs_HD_Amplitude_2015_polyfitto_combine.png')
Image2 = Image.open('Amplitude_TT_vs_HD_Amplitude_2016_polyfitto_combine.png')
Image3 = Image.open('Amplitude_TT_vs_HD_Amplitude_2017_polyfitto_combine.png')
Image4 = Image.open('Amplitude_TT_vs_HD_Amplitude_2018_polyfitto_combine.png')
Image5 = Image.open('Amplitude_TT_vs_HD_Amplitude_2019_polyfitto_combine.png')
Image6 = Image.open('Amplitude_TT_vs_HD_Amplitude_2020_polyfitto_combine.png')
Image7 = Image.open('Amplitude_TT_vs_HD_Amplitude_2021_polyfitto_combine.png')
Image8 = Image.open('Amplitude_TT_vs_HD_Amplitude_2022_polyfitto_combine.png')

Image1 = Image.open('min_max_data_tocombine_2015.png')
Image2 = Image.open('min_max_data_tocombine_2016.png')
Image3 = Image.open('min_max_data_tocombine_2017.png')
Image4 = Image.open('min_max_data_tocombine_2018.png')
Image5 = Image.open('min_max_data_tocombine_2019.png')
Image6 = Image.open('min_max_data_tocombine_2020.png')
Image7 = Image.open('min_max_data_tocombine_2021.png')
Image8 = Image.open('min_max_data_tocombine_2022.png')

F = 6
f = 6
x_text = -100 # x = -100 pour min_max_data et amplitude
y_text = 70

fig, ax = plt.subplots(4, 2, figsize=(7,4)) # figsize=(4,7) pour amplitudeTTvsHD # (7,4) pour daily discharge
ax[0, 0].imshow(Image1)
ax[0, 0].axis('off')
ax[0, 0].text(x_text, y_text, "(a)", fontsize=f)

ax[1, 0].imshow(Image2)
ax[1, 0].axis('off')
ax[1, 0].text(x_text, y_text, "(b)", fontsize=f)

ax[2, 0].imshow(Image3)
ax[2, 0].axis('off')
ax[2, 0].text(x_text, y_text, "(c)", fontsize=f)

ax[3, 0].imshow(Image4)
ax[3, 0].axis('off')
ax[3, 0].text(x_text, y_text, "(d)", fontsize=f)

ax[0, 1].imshow(Image5)
ax[0, 1].axis('off')
ax[0, 1].text(x_text, y_text, "(e)", fontsize=f)

ax[1, 1].imshow(Image6)
ax[1, 1].axis('off')
ax[1, 1].text(x_text, y_text, "(f)", fontsize=f)

ax[2, 1].imshow(Image7)
ax[2, 1].axis('off')
ax[2, 1].text(x_text, y_text, "(g)", fontsize=f)

ax[3, 1].imshow(Image8)
ax[3, 1].axis('off')
ax[3, 1].text(x_text, y_text, "(h)", fontsize=f)

plt.subplots_adjust( wspace=0.01, hspace=0.01)#, left=0.01, bottom=0.04, right=0.99, top=0.96)

path_save = ''
plt.savefig(path_save + 'Combine_fig_attenuation_2015-2022.png', dpi=300, bbox_inches='tight', pad_inches=.1)
plt.savefig(path_save + 'Combine_fig_amplitudeTTvsHD_polyfit_2015-2022.png', dpi=300, bbox_inches='tight', pad_inches=.1)
plt.savefig(path_save + 'Combine_fig_daily_discharge_2015-2022.png', dpi=300, bbox_inches='tight', pad_inches=.1)

#################
fig, ax = plt.subplots(2, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [1, 2]})
Image1 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'Amplitude_TT_vs_HD_Amplitude_withdischargecategories_polyfit_allvalues.png')
Image2 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'amplification_vs_discharge_and_tidal_range.png')
F = 6
f = 6
ax[0].imshow(Image1)
ax[0].axis('off')
ax[0].text(0, 50, "(a)", fontsize=f)
ax[1].imshow(Image2)
ax[1].axis('off')
ax[1].text(-500, 0, "(b)", fontsize=f)
ax[1].text(-500, 3500, "(c)", fontsize=f)
plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, bottom=0.01, right=0.99, top=0.99)
#plt.subplots_adjust( wspace=0.01, hspace=0.01)#, left=0.01, bottom=0.04, right=0.99, top=0.96)
path_save = '/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
plt.savefig(path_save + 'Combined_amplification.png', dpi=300, bbox_inches='tight', pad_inches=.1)
fig, ax = plt.subplots(2, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [1, 2]})
Image1 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'Amplitude_TT_vs_HD_Amplitude_withdischargecategories_polyfit_allvalues.png')
Image2 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'amplification_vs_discharge_and_tidal_range.png')
F = 6
f = 6
ax[0].imshow(Image1)
ax[0].axis('off')
ax[0].text(0, 50, "(a)", fontsize=f)
ax[1].imshow(Image2)
ax[1].axis('off')
ax[1].text(-500, 0, "(b)", fontsize=f)
ax[1].text(-500, 3500, "(c)", fontsize=f)

############
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
Image1 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'Comp_Min_and_max_Qvalues_vs_water_levels_zoom.png')
F = 6
f = 6
ax.imshow(Image1)
ax.axis('off')
ax.text(0, 700, "(a)", fontsize=f)
ax.text(0, 2500, "(b)", fontsize=f)
ax.text(0, 4000, "(c)", fontsize=f)
plt.subplots_adjust( wspace=0.01, hspace=0.01)#, left=0.01, bottom=0.04, right=0.99, top=0.96)
path_save = '/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
plt.savefig(path_save + 'Combined_comp_min_max.png', dpi=300, bbox_inches='tight', pad_inches=.1)

##############
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
Image1 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'Tidal_elevations_Q_lag_1562022.png')
F = 6
f = 6
y_text = 2000
ax.imshow(Image1)
ax.axis('off')
ax.text(500, 500, "(a)", fontsize=f)
ax.text(500, 2500, "(b)", fontsize=f)
plt.subplots_adjust( wspace=0.01, hspace=0.01)#, left=0.01, bottom=0.04, right=0.99, top=0.96)
path_save = '/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
plt.savefig(path_save + 'Combined_tidal_elevation.png', dpi=300, bbox_inches='tight', pad_inches=.1)

############################
fig, ax = plt.subplots(2, 1, figsize=(6, 4))
Image1 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'Zoom_RRD_v3.jpeg')
Image2 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'Vanuc.png')
F = 6
f = 6
y_text = 2000
ax[0].imshow(Image1)
ax[0].axis('off')
ax[0].text(200, 700, "(a)", fontsize=f)
ax[0].text(2000, 700, "(b)", fontsize=f)
ax[1].imshow(Image2)
ax[1].axis('off')
ax[1].text(1, 1, "(c)", fontsize=f)
plt.subplots_adjust( wspace=0.01, hspace=0.01)#, left=0.01, bottom=0.04, right=0.99, top=0.96)
path_save = '/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
plt.savefig(path_save + 'Combined_region.png', dpi=300, bbox_inches='tight', pad_inches=.1)

##########################
fig, ax = plt.subplots(2, 1, figsize=(6, 4))

Image1 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'test_Ebb_Flood_duration_TT_vs_tidal_range.png')
Image2 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'Ebb_Flood_duration_TT_vs_discharge_alldata.png')
F = 6
f = 6
y_text = 2000
ax[0].imshow(Image1)
ax[0].axis('off')
ax[0].text(200, 70, "(a)", fontsize=f)

ax[1].imshow(Image2)
ax[1].axis('off')
ax[1].text(200, 70, "(b)", fontsize=f)

plt.subplots_adjust( wspace=0.01, hspace=0.01)#, left=0.01, bottom=0.04, right=0.99, top=0.96)

path_save = '/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
plt.savefig(path_save + 'Combined_ebb_and_flood_duration.png', dpi=300, bbox_inches='tight', pad_inches=.1)



#############################################################################################
#########################################""
fig, ax = plt.subplots(3, 2, figsize=(6, 4))

Image1 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'low_Q_low_TR_exemple_tidal_elevations_Q_lag_812021.png')
Image2 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'low_Q_median_TR_exemple_tidal_elevations_Q_lag_1232021.png')
Image3 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'low_Q_high_TR_exemple_tidal_elevations_Q_lag_28102022.png')
Image4 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'high_Q_low_TR_exemple_tidal_elevations_Q_lag_2552022.png')
Image5 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'high_Q_median_TR_exemple_tidal_elevations_Q_lag_2682022.png')
Image6 = Image.open('/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
                    'high_Q_high_TR_exemple_tidal_elevations_Q_lag_1862022.png')

F = 6
f = 6
y_text = 2000
ax[0, 0].imshow(Image1)
ax[0, 0].axis('off')
ax[0, 0].set_title("Low discharge", fontsize=F)
ax[0, 0].text(-100, y_text, "Low tidal range", fontsize=F, rotation='vertical', horizontalalignment='center', verticalalignment='center')
ax[0, 0].text(200, 70, "(a)", fontsize=f)

ax[1, 0].imshow(Image2)
ax[1, 0].axis('off')
ax[1, 0].text(-100, y_text, "Medium tidal range", fontsize=F, rotation='vertical', horizontalalignment='center', verticalalignment='center')
ax[1, 0].text(200, 70, "(c)", fontsize=f)

ax[2, 0].imshow(Image3)
ax[2, 0].axis('off')
ax[2, 0].text(-100, y_text, "High tidal range", fontsize=F, rotation='vertical', horizontalalignment='center', verticalalignment='center')
ax[2, 0].text(200, 70, "(e)", fontsize=f)

ax[0, 1].imshow(Image4)
ax[0, 1].axis('off')
ax[0, 1].set_title("High discharge", fontsize=F)
#ax[0, 1].set_title(-100, 500, "(c) Bias DJF", fontsize=F, rotation='vertical', horizontalalignment='center', verticalalignment='center')
#ax[2, 0].text(215, 210, "R=0.93", fontsize=f, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))
#ax[2, 0].text(1050, 210, "B=-0.17", fontsize=f,bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))
ax[0, 1].text(200, 70, "(b)", fontsize=f)

ax[1, 1].imshow(Image5)
ax[1, 1].axis('off')
#ax[3, 0].text(-100, 500, "(d) SYM JJA", fontsize=F, rotation='vertical', horizontalalignment='center', verticalalignment='center')
ax[1, 1].text(200, 70, "(d)", fontsize=f)

ax[2, 1].imshow(Image6)
ax[2, 1].axis('off')
ax[2, 1].text(200, 70, "(f)", fontsize=f)

plt.subplots_adjust( wspace=0.01, hspace=0.01)#, left=0.01, bottom=0.04, right=0.99, top=0.96)

path_save = '/home/penicaud/Documents/Article/Article_tidal_gauge/Figures/'
plt.savefig(path_save + 'Combine_fig_lag_elevations.png', dpi=300, bbox_inches='tight', pad_inches=.1)

######################################################""
fig, ax = plt.subplots(2, 2, figsize=(6, 4))

Image1 = Image.open('/home/penicaud/Documents/Modèles/Marée_FES/amplitude_O1_no_Hg.png')
Image2 = Image.open('/home/penicaud/Documents/Modèles/Marée_FES/amplitude_K1_no_Hg.png')
Image3 = Image.open('/home/penicaud/Documents/Modèles/Marée_FES/amplitude_M2_no_Hg.png')
Image4 = Image.open('/home/penicaud/Documents/Modèles/Marée_FES/amplitude_S2_no_Hg.png')

F = 12
f = 9
ax[0, 0].imshow(Image1)
ax[0, 0].axis('off')
ax[0, 0].set_title("(1) SST [°C]", fontsize=F)
#ax[0, 0].text(-100, 500, "(a) SYM DJF", fontsize=F, rotation='vertical', horizontalalignment='center', verticalalignment='center')
# ax[0, 0].text(200, 70, "(a)", fontsize=f)

ax[0, 1].imshow(Image2)
ax[0, 1].axis('off')
#ax[1, 0].text(-100, 500, "(b) Obs DJF", fontsize=F, rotation='vertical', horizontalalignment='center', verticalalignment='center')
# ax[1, 0].text(200, 70, "(d)", fontsize=f)

ax[1, 0].imshow(Image3)
ax[1, 0].axis('off')
#ax[2, 0].text(-100, 500, "(c) Bias DJF", fontsize=F, rotation='vertical', horizontalalignment='center', verticalalignment='center')
#ax[2, 0].text(215, 210, "R=0.93", fontsize=f, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))
#ax[2, 0].text(1050, 210, "B=-0.17", fontsize=f,bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))

ax[1, 1].imshow(Image4)
ax[1, 1].axis('off')
#ax[3, 0].text(-100, 500, "(d) SYM JJA", fontsize=F, rotation='vertical', horizontalalignment='center', verticalalignment='center')
# ax[3, 0].text(200, 70, "(j))", fontsize=f)


plt.subplots_adjust( wspace=0.01, hspace=0.01)#, left=0.01, bottom=0.04, right=0.99, top=0.96)

path_save = '/home/penicaud/Documents/These/Figures/Chapter_Region/'
plt.savefig(path_save + 'amplitude_O1K1M2S2.png', dpi=300, bbox_inches='tight', pad_inches=.1)
# plt.show()
