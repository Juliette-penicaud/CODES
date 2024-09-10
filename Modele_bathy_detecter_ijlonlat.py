# 10/02/2023 : programme qui lit un fichier lonlatdepth issu d'un fichier kml (la bathy que l'on veut incorporer au modèle
# 10/02/2023 : programme qui lit un fichier lonlatdepth issu d'un fichier kml (la bathy que l'on veut incorporer au modèle
# et le compare au fichier ijlonlatdepth du modèle.
# OBJECTIF : trouver la maille correspondante i j
# fichier de sortie : lon_kml lat_kml lon_mod lat_mod i_mod j_mod depth_kml depth_kml diff_depth
import sys
import numpy as np
import pandas as pd


bidouille1=0
bidouille=False
if bidouille1 : #aller jusqua la derniere valeur de la valeur que l'on choisi
    rep='/home/penicaud/Documents/Modèles/Création_bathy/'
    file=rep+'post_smoothed_bathy_finale_v5.ijDeltaH'
    outfile=rep+'post_smoothed_bathy_finale_v5_unique.ijDeltaH'
    data=pd.read_csv(file, sep=' ', header=None)
    print(data)
    #df_sortie = pd.DataFrame(data)
    data.columns = ['i', 'j', 'depth', 'h']
    df_sortie_unique = data.drop_duplicates(subset=['i', 'j'], keep='last')
    df_sortie_unique.to_csv(outfile, sep=' ', index=False, float_format='%.4f')
    print(outfile)
    print(df_sortie_unique)
    sys.exit(1)
elif bidouille : #détecter des valeurs isolées de la bathy 3ple
    import matplotlib.pyplot as plt
    rep='/home/penicaud/Documents/Modèles/Création_bathy/'
    file=rep+'partie_bathy3ple_+2m_v2.ijDeltaH'
    #outfile=rep+'partie_bathy3ple_+2m_modified.ijDeltaH'
    #file = rep + 'partie_-2_5_sur_bathyAlexei_plus2'
    outfile = rep + 'partie_procheestuaire_onGEBCO_16052023_tronquée'
    data=pd.read_csv(file, sep=' ', header=None)
    print(data)
    data.columns = ['x', 'y', 'depth', 'h']
    condition = (data['y'] < 1.43 * data['x'] + 291.3) | (data['y']>495) | (data['y']>-5.67*data['x']+1095) | (data['y'] > -0.78*data['x']+546.7)
    #condition1 = (data['y'] > -0.78*data['x']+546.7)
    val_test=(data[condition]) #sur le plateau,
    # j'ai calculé la fonction affine j=1.43 * i+291.3 si les valeurs sont inférieures, elles sont isolée (déduction visuelle àp de visualisation ferret
    data2 = data.drop(np.where(condition==True)[0])
    val_test = val_test.drop(['depth', 'h'], axis=1)
    fig, ax = plt.subplots()
    ax.scatter(data2['x'], data2['y'], color='red', marker='x', alpha=0.5)
    ax.scatter(val_test['x'], val_test['y'], color='blue')
    fig.savefig(rep + 'test_points_bathy_16052023')

    data2.to_csv(outfile, sep=' ', index=False, float_format='%.4f')

    sys.exit(1)





create_tab = True  # 1 is to create the data_sorti table, 0 is to read it
savefile = True
file = 'bathy_chenal_16052023'  # main or suppl or chenaux
rep = '/home/penicaud/Documents/Modèles/Création_bathy/'

dict_mod = {'val_to_change_kml': 'val_to_change_mod',
            'bathy-2-0_procheestuaire_10052023': 'bathy_modele_inclu_Alexei_ijlonlatdepth' ,
            'bathy_-2_5_procheestuaire_16052023': 'bathy_modele_inclu_Alexei_ijlonlatdepth'
            }

if file in dict_mod.keys() :
    file_modele = rep + dict_mod[file]
else :
    file_modele = rep + 'fichier_ijlonlatdepth_modele'

dict_kml = {'main': {'name' : 'fichier_lonlatdepth_kml' , 'out' : '_main' } ,
        'suppl':  {'name' : 'fichier_points_kml_suppl', 'out' : '_unique_suppl'} ,
        'chenaux': {'name' :'fichier_points_kml_chenaux' , 'out' : '_unique_chenaux' },
        'all' : {'name' : 'fichier_all_bathy', 'out' : '_unique_all_bathy' } ,
        '0-5' : {'name' : 'fichier_0-5_all', 'out' : '_unique_0-5' } ,
        '-2-0' : {'name' : 'fichier_-2-0_all' , 'out' : '_unique_-2-0' } ,
        'val_to_change_kml' : {'name' :'val_to_change_kml' , 'out' : 'val_to_change_kml' } ,
        'all_no_vanuc' : {'name' :'fichier_bathy_no_vanuc_no10-25' , 'out' : '_unique_bathy_no_vanuc_no10-25' } ,
        'bathy_Alexei' : {'name' :  'bathy_Alexei.lonlat' , 'out' :   '_bathy_Alexei'} ,
        'bathy-2-0_procheestuaire_10052023': {'name' : 'bathy_-2-0_procheestuaire_10052023' , 'out' :
            'bathy-2-0_procheestuaire_10052023' },
        'bathy_-2_5_procheestuaire_16052023': {'name' : 'bathy_-2_5_procheestuaire_16052023' , 'out' :
            '_procheestuaire_1605' },

        'bathy_chenal_16052023' : {'name' : 'bathy_chenal_16052023' , 'out' :
            'bathy_chenal_16052023' }
       }
file_kml = rep + dict_kml[file]['name'] #always the last one
bidouille=False
if bidouille : #cette bidouille sert à trouver les lon lat du fichier de la bathy de Alexei (i j depth) en se servant
    # du fichier initial avec i j lon lat depth
    file_modele2 = rep + 'fichier_ijlonlatdepth_modele'
    data_modele2 = pd.read_csv(file_modele2, sep="   ", header=None, names=['i', 'j', 'lon', 'lat', 'depth'],
                               engine='python')
    file_modele = rep + 'bathy_modele_inclu_Alexei' # i j depth
    data_alexei = pd.read_csv(file_modele, sep=" ", header=None, names=['i', 'j', 'depth'],
                              engine='python')

    data_alexei = data_alexei.reset_index()
    data_alexei = data_alexei.drop('index', axis=1)
    i_alexei = data_alexei['i']
    j_alexei = data_alexei['j']
    depth_alexei = data_alexei['depth']
    # print('lon kml', lon_alexei
    i_mod = data_modele2['i']
    j_mod = data_modele2['j']
    lon_mod = data_modele2['lon']
    lat_mod = data_modele2['lat']
    depth_mod = data_modele2['depth']

    lonlat_sortie = np.zeros((len(i_alexei), 2))
    for i in range(len(i_alexei)) :
        condition = ((i_mod == int(i_alexei[i])) & (j_mod == int(j_alexei[i]))) == True
        ind_trouve = np.where(condition)[0]
        print(ind_trouve)
        if len(ind_trouve) == 0 :
            lonlat_sortie[i, 0] = np.nan
            lonlat_sortie[i, 1] = np.nan
        else :
            lonlat_sortie[i,0] = data_modele2['lon'][ind_trouve].values
            lonlat_sortie[i,1] = data_modele2['lat'][ind_trouve].values

    data_alexei['lon'] = lonlat_sortie[:, 0]
    data_alexei['lat'] = lonlat_sortie[:, 1]
    data_alexei_unique = data_alexei.dropna(how='any')
    #data_alexei_unique = data_alexei_unique.drop_duplicates(subset=['i', 'j'], keep='first')
    outfile = rep + 'bathy_modele_inclu_alexei_ijlonlatdepth_16052023'
    data_alexei_unique.to_csv(outfile, sep=' ', index=False)

if create_tab:

    data_kml = pd.read_csv(file_kml, sep=' ', names=['lon', 'lat', 'depth'], header=None, engine='python')
    data_modele = pd.read_csv(file_modele, sep="   ", header=None, names=['i', 'j', 'lon', 'lat', 'depth'],
                              engine='python') #sep = "   " for normal file, and " " for others

    # print('data_modele')
    # print(data_modele)
    # print(data_kml)

    lon_kml = data_kml['lon']
    lat_kml = data_kml['lat']
    depth_kml = data_kml['depth']
    # print('lon kml', lon_kml)lr
    i_mod = data_modele['i']
    j_mod = data_modele['j']
    lon_mod = data_modele['lon']
    lat_mod = data_modele['lat']
    depth_mod = data_modele['depth']

    data_sortie = np.zeros((len(lon_kml),
                            11))  # 11 pour i j  lon_m lat_m depth_m lon_k lat_k depth_k depth_m-depth_k depth_k_moy depth_kmoy-depth_mod depth_moy et diff_dmoy
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
        data_sortie[ind_kml, 4] = depth_mod[ind_trouve]
        data_sortie[ind_kml, 5] = lon_kml[ind_kml]
        data_sortie[ind_kml, 6] = lat_kml[ind_kml]
        data_sortie[ind_kml, 7] = depth_kml[ind_kml]
        data_sortie[ind_kml, 8] = float(depth_kml[ind_kml]) - float(
            depth_mod[ind_trouve])  # diff depth à donner pour le post_smooth

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
    df_sortie.columns = ['i', 'j', 'lon_mod', 'lat_mod', 'depth_mod', 'lon_kml', 'lat_kml', 'depth_kml', 'diff_depth',
                         'depth_moy', 'diff_depth_moy']
    # print(df_sortie.head())
    table_moy_depth = []
    df_ind_trouve = pd.DataFrame(list_ind_trouve)
    print(list_ind_trouve)
    for l in list_unique:
        # print('l',l)
        index = np.where(df_ind_trouve[:] == l)[0]
        cumul_depth = 0
        print('len index', len(index), index)
        for ind in index:
            cumul_depth = cumul_depth + data_sortie[ind, 7]
            # print('cumul depth', cumul_depth)
        moy_depth = cumul_depth / len(index)
        table_moy_depth.append(cumul_depth / len(index))
        data_sortie[index, 9] = moy_depth
        data_sortie[index, 10] = moy_depth - data_sortie[index, 4]  # difference of depth mod and depth moy,

        # transform into df
        df_sortie = pd.DataFrame(data_sortie)
        df_sortie.columns = ['i', 'j', 'lon_mod', 'lat_mod', 'depth_mod', 'lon_kml', 'lat_kml', 'depth_kml',
                             'diff_depth', 'depth_moy', 'diff_depth_moy']
        # uniques , supress duplicates
        df_sortie_unique = df_sortie.drop_duplicates(subset=['i', 'j'], keep='first')
        if file == 'last':
            df_sortie_unique = df_sortie.copy()

    if savefile:
        outfile = rep + 'data_sortie_ijlonlatdepth'
        outfile = outfile + dict_kml[file]['out']

        df_sortie_unique.to_csv(outfile, sep=' ', index=False, float_format='%.4f')
        # np.savetxt(outfile+'.txt', data_sortie)
        # data_sortie.tofile(outfile+'2.csv')
    sys.exit(1)

else:
    data_sortie = pd.read_csv(rep + 'data_sortie_ijlonlatdepth.txt', sep=' ',
                              header=None)  # , names=['i', 'j', 'lon_mod', 'lat_mod', 'depth_mod',
    # 'lon_kml', 'lat_kml', 'depth_kml',
    #        'diff_depth', 'depth_moy', 'diff_depth_moy' ])
    df_sortie = pd.DataFrame(data_sortie)
    df_sortie.columns = ['i', 'j', 'lon_mod', 'lat_mod', 'depth_mod', 'lon_kml', 'lat_kml', 'depth_kml',
                         'diff_depth', 'depth_moy', 'diff_depth_moy']
    # print(df_sortie)
    # print(np.shape(np.where(df_sortie.duplicated(subset=['i','j'])==True)))
    # print(np.where(df_sortie.duplicated(subset=['i','j'])==True))
    # print(df_sortie.loc[np.where(df_sortie.duplicated(subset=['i','j'])==True)[0],:])
    df_sortie_unique = df_sortie.drop_duplicates(subset=['i', 'j'], keep='first')
    # print(df_sortie_unique, np.shape(df_sortie_unique))

    if savefile:
        outfile = rep + 'data_unique_ijlonlat'
        outfile = outfile + dict_kml[file]['out']
        df_sortie_unique.to_csv(outfile, sep=' ', index=False, float_format='%.4f')
        # df_sortie.to_csv(rep+'data_sortie_ijlonlat_v2', sep=' ', index=False, float_format='%.4f')

    df_unique_sorted = df_sortie_unique.sort_values(by=['i', 'j'])
    print(df_unique_sorted)
    df_unique_sorted = df_unique_sorted.set_index(['i', 'j'])
    # df_unique_sorted.index= df_unique_sorted.index.astype(int)
    # print(df_unique_sorted)
    # Je veux compléter la liste des i et j pour ensuite interpoler la profondeur kml, et ensuite déterminer la diff de bathy à appliquer.

    # print(int(np.min(df_unique_sorted['i'])), int(np.max(df_unique_sorted['i'])))
    new_array = df_unique_sorted.reindex((70, 114), (420, 460))
    # new_array=df_unique_sorted.set_index(['i','j']).reindex(
    #    range(int((np.min(df_unique_sorted['i'])), int(np.max(df_unique_sorted['i'])))),
    #    range(int(np.min(df_unique_sorted['j'])), int(np.max(df_unique_sorted['j'])))).ffill().reset_index
    # new_array=df_unique_sorted.set_index(['j']).reindex(range(int(np.min(df_unique_sorted['j'])), int(np.max(df_unique_sorted['j'])))).ffill().reset_index
    # new_array=df_unique_sorted.set_index(pd.interval_range(70,110))
    print('shape new array', np.shape(new_array), new_array.head())

    from scipy.interpolate import interpn
    # x = np.linspace(0, 50, 50)
    # y = np.linspace(0, 50, 50)
    # z = np.linspace(0, 50, 50)
    # points = (x, y, z)
    # X, Y, Z = np.meshgrid(x, y, z)
    # values = func_3d(X, Y, Z)
    # point = np.array([2.5, 3.5, 1.5])
    # print(interpn(points, values, point))

    # Vi = interpn(new_array, df_unique_sorted['i','j', 'depth_kml'],  )
