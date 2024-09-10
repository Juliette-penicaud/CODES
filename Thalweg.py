# 12/07/2024 : adaptation de MODELE_bathy_detecter_ijlonlat pour trouver les coordonnées i j depuis un fichier kml
# programme qui lit un fichier lonlatdepth issu d'un fichier kml (la bathy que l'on veut incorporer au modèle
# et le compare au fichier ijlonlatdepth du modèle.
# OBJECTIF : trouver la maille correspondante i j
# fichier de sortie : lon_kml lat_kml lon_mod lat_mod i_mod j_mod depth_kml depth_kml diff_depth
import sys
import numpy as np
import pandas as pd

rep = '/home/penicaud/Documents/Data/Map/'
file = rep + 'coordinates_points_tal'
data_kml=pd.read_csv(file, sep=' ', header=None)
data_kml.columns = ['lon', 'lat', 'suppr']

file_modele = '/home/penicaud/Documents/Modèles/Création_bathy/fichier_all_mod'
data_modele = pd.read_csv(file_modele , sep='   ', header=None)
data_modele = data_modele.rename(columns={0:'i', 1:'j', 2: 'lon', 3:'lat'})
lon_kml = data_kml['lon']
lat_kml = data_kml['lat']
# print('lon kml', lon_kml)lr
i_mod = data_modele['i']
j_mod = data_modele['j']
lon_mod = data_modele['lon']
lat_mod = data_modele['lat']

data_sortie = np.zeros((len(lon_kml),6))
# 6 pour i j  lon_m lat_m lon_k lat_k
# 11 pour i j  lon_m lat_m depth_m lon_k lat_k depth_k depth_m-depth_k depth_k_moy depth_kmoy-depth_mod depth_moy et diff_dmoy
list_ind_trouve = []
for ind_kml in range(len(lon_kml)):  # boucle sur les lon_kml
    print('ind_kml', ind_kml, lon_kml[ind_kml], lat_kml[ind_kml])
    diff_min = 10000
    for ind_mod in range(len(lon_mod)):
        diff = abs(lon_kml[ind_kml] - lon_mod[ind_mod]) + abs(lat_kml[ind_kml] - lat_mod[ind_mod])
        # print('diff', diff)
        if diff < diff_min:
            # print('nouveau diff_min', diff_min, 'devient ', diff)
            diff_min = diff
            ind_trouve = ind_mod  # indice de la ligne ou il faudra chercher les valeurs de i et j
            # print('ind_trouve', ind_trouve)
    # print('diff min', diff_min, ind_trouve, ind_kml)
    list_ind_trouve.append(ind_trouve)
    data_sortie[ind_kml, 0] = i_mod[ind_trouve]
    data_sortie[ind_kml, 1] = j_mod[ind_trouve]
    data_sortie[ind_kml, 2] = lon_mod[ind_trouve]
    data_sortie[ind_kml, 3] = lat_mod[ind_trouve]
    data_sortie[ind_kml, 4] = lon_kml[ind_kml]
    data_sortie[ind_kml, 5] = lat_kml[ind_kml]

print('ok')

####### VOIR LE NOMBRE DE REDONDANCE DANS LES CASES (FORTRAN = 359 VALEURS UNIQUES SEUELEMENT SUR LES 1334)
count = pd.Series(list_ind_trouve).value_counts()
nsame = len(count[(count > 1)])
print(count[(count > 1)])  # affiche seulement la ou il y a plusieurs valeurs sur une seule case
print('nsame', nsame)
ndiff = len(lon_kml) - nsame  # valeur de i et j uniques
print('ndiff', ndiff)

#### FAIRE MOYENNE LA OU LES VALEURS SONT DANS LES MEMES CASES
list_unique = set(list_ind_trouve)  # liste de valeurs uniques
df_sortie = pd.DataFrame(data_sortie)
#df_sortie.columns = ['i', 'j', 'lon_mod', 'lat_mod', 'depth_mod', 'lon_kml', 'lat_kml', 'depth_kml', 'diff_depth',
#                     'depth_moy', 'diff_depth_moy']
df_sortie.columns = ['i', 'j', 'lon_mod', 'lat_mod', 'lon_kml', 'lat_kml']
df_sortie_unique = df_sortie.drop_duplicates(subset=['i', 'j'], keep='first')

savefile = True
if savefile:
    outfile = rep + 'data_sortie_ijlonlatdepth_thalweg'
    df_sortie_unique.to_csv(outfile, sep=' ', index=False, float_format='%.4f')
    # np.savetxt(outfile+'.txt', data_sortie)
    # data_sortie.tofile(outfile+'2.csv')

sys.exit(1)