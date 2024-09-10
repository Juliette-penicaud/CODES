# 15/03/2023 : programme qui lit un fichier lonlatdepth issu d'un fichier kml (la bathy que l'on veut incorporer au modèle
# et le compare au fichier ijlonlatdepth du modèle.
# JE VEUX : une bathy régulière, sans impact d'endroits ou il n'y a pas de points de post smoothed. PB : obtenir tous les points de bathy
# DEROULE : obtenir le bathy  GEBCO du point i, les bathy les plus proches des kml, faire une interpolation pour avoir la bathy voulu, faire la diff avec la bathy GEBCO
# AUTRE IDEE : si "frontière detectée avec un point proche qui est sous 0, mettre le d à 1. 
import sys
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

create_tab1= True #1 is to create the data_sorti table, 0 is to read it
check_polygon = True
interpolate=True
savefile=True


rep='/home/penicaud/Documents/Modèles/Création_bathy/'
rep2 = '/home/penicaud/Documents/Data/Map/'
file_modele=rep+'fichier_ijlonlatdepth_modele'
#file_sortie=rep + 'data_sortie_ijlonlatdepth_unique_0-5_sorted'
file_sortie=rep+ 'data_sortie_ijlonlatdepth_unique_bathy_no_vanuc_no10-25'


data_modele=pd.read_csv(file_modele, sep="   " , header=None, names=['i', 'j', 'lon', 'lat', 'depth'], engine='python')
data_sortie = pd.read_csv(file_sortie, sep=' ', header=None)
df_sortie = pd.DataFrame(data_sortie)
df_sortie.columns = ['i', 'j', 'lon_mod', 'lat_mod', 'depth_mod', 'lon_kml', 'lat_kml', 'depth_kml',
                     'diff_depth', 'depth_moy', 'diff_depth_moy']
i_kml=df_sortie['i']
j_kml=df_sortie['j']
lon_mod= df_sortie['lon_mod']
lat_mod=df_sortie['lat_mod']
depth_mod=df_sortie['depth_mod']
lon_kml = df_sortie['lon_kml']
lat_kml = df_sortie['lat_kml']
depth_kml = df_sortie['depth_kml']
diff_depth=df_sortie['diff_depth']
depth_moy=df_sortie['depth_moy']
diff_depth_moy=df_sortie['diff_depth_moy']

print('df_sortie \n', df_sortie.head(), '\n', data_modele.head())

#data_to_complete = np.zeros((len(lon_kml),11))  # 11 pour i j  lon_m lat_m depth_m lon_k lat_k depth_k depth_m-depth_k depth_k_moy depth_kmoy-depth_mod depth_moy et diff_dmoy
#data_to_complete=np.zeros((len(lon_modele),11))
#list_lon_kml list_lat_kml, list_depth_kml, list_depth_moy, list_diff_depth_moy = [], [], [], [], []
data_to_complete=data_modele.copy()
data_to_complete["lon_kml"], data_to_complete['lat_kml'], data_to_complete['depth_kml'], data_to_complete['diff_depth'],\
    data_to_complete["depth_moy"], data_to_complete["diff_depth_moy"] \
    = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
data_to_complete=data_to_complete.rename(columns={'lon': 'lon_mod', 'lat': "lat_mod", 'depth': 'depth_mod'})

print('shape ', np.shape(df_sortie), np.shape(data_to_complete))
print(np.shape(data_to_complete), type(data_to_complete) ,'\n', data_to_complete.tail(5))

print(list(df_sortie), '\n', list(data_to_complete))
taille=len(data_to_complete)-len(df_sortie)
a=np.zeros((taille,11))
a=pd.DataFrame(a)
a.columns = ['i', 'j', 'lon_mod', 'lat_mod', 'depth_mod', 'lon_kml', 'lat_kml', 'depth_kml',
                     'diff_depth', 'depth_moy', 'diff_depth_moy']
#print(np.shape(a),a)
#df_sortie2=pd.concat([df_sortie,a], ignore_index=True, axis=0)
#print(np.where(df_sortie2==data_to_complete))
#print(df_sortie2[np.where(df_sortie2==data_to_complete)])
#df_sortie2=np.zeros((len(data_to_complete["lon_kml"]),11))
#df_sortie2[0:len(data_to_complete["lon_kml"]),:]=df_sortie[:,:]
#df=data_to_complete.compare(df_sortie2, keep_equal=True, keep_shape=True)

#var = data_to_complete[np.where(data_to_complete['lon'] == df_sortie['lon_mod'])[0]]#and data_to_complete['lat'] == df_sortie['lat_mod']]
#print(var)

#1. interpoler les bathy voulues à partir de la bathy kml
#TOUT FAIRE dans le domaine i j ? ou lon lat ? lon lat serait mieux plus précis et tiendrait compte de la distance entre les variables.

# COmparer les fichiers data_sortie_chenaux et ijlonlat.. modele. Si pas val de i et j dans le fichier sortie, ajouter les valeurs du fichier de bathy, et mettre -1

#1.copier fichier_ijlonlatdepth_modele.
# Boucler dessus: ajouter si ca existe les valeurs de data_sortie.
# Si non, tester si dans valeur d'un des polygone. Si oui : affecter une constante ? si non : quoi faire ?

if create_tab1 :
    print('Create v1 bathy')
    for ind_kml in range(len(lon_kml)):
        print("val", ind_kml)
        val_i =i_kml[ind_kml]
        val_j =j_kml[ind_kml]
        print('val_i and j ',  val_i, val_j)
        condition=((data_to_complete["i"]==val_i) & (data_to_complete["j"]==val_j))
        indice=np.where(condition)[0]
        indice=int(indice[0])
        print('indice ', indice)
        for k in range(5,11):
            #print('k=', k)
            val=df_sortie.iloc[ind_kml][k]
            #print('val', val, type(val))
            data_to_complete.iat[indice,k]=val
            #print(data_to_complete.iat[indice,k])

    outfile=rep+"all_bathy_v1"
    data_to_complete.to_csv(outfile, sep=' ', index=False, float_format='%.4f')
    #print(data_to_complete.iat[7770,10])

if check_polygon :
    print('CHECK POLYGON')
    rep2='/home/penicaud/Documents/Data/Map/'
    file_polygon=['file_polygon'+str(i) for i in range(1,6)]
    var_polygon=['polygon'+str(i) for i in range(1,6)]
    for i in range(1,6):
        print('i',i)
        file_polygon[i-1]='patch_banc_sable'+str(i)+'.kml'
        var_polygon[i-1] = Polygon(pd.read_csv(rep2+file_polygon[i-1], sep=' '))
        print(var_polygon[i-1])
    print(file_polygon)
    data=pd.read_csv(rep+'all_bathy_v1', sep=' ')#, header=None)#, names=['i', 'j', 'lon_mod', 'lat_mod', 'depth_mod',
    data_to_complete=pd.DataFrame(data)
    #print(data_to_complete.iloc[47][10])
    print(data_to_complete.isnull().sum())
    soustab=data_to_complete[data_to_complete['diff_depth'].isna()==True]
    #print(soustab, np.shape(soustab))

    c_no, c_yes = 0 , 0
    cst=-0.5
    print('sous tab index', len(list(soustab.index)))
    #data_to_complete.loc[150, 9] = -0.5
    #print('data',     data_to_complete.loc[150, 9])
    #sys.exit(1)
    for v in list(soustab.index):
        #print('v', v)
        lon = data_to_complete.iloc[v]['lon_mod']
        lat = data_to_complete.iloc[v]['lat_mod']
        point=Point(lon,lat)
        for i in range(len(var_polygon)):
            if point.within(var_polygon[i]):
                c_yes=c_yes+1
                data_to_complete.at[v,'depth_kml']=cst
                data_to_complete.at[v,'depth_moy']=cst
                data_to_complete.at[v,'lon_kml']=lon
                data_to_complete.at[v,'lat_kml']=lat
                gebco_depth=data_to_complete.iloc[v]['depth_mod']
                val_depth=-0.5-float(gebco_depth)
                #print('val depth', val_depth)
                data_to_complete.at[v,'diff_depth']=val_depth
                data_to_complete.at[v,'diff_depth_moy']=val_depth
            else :
                c_no=c_no+1
                #print('NO')

    print('c yes and no', c_yes, c_no)
    print(data_to_complete['depth_moy'].isnull().sum())
    print(data_to_complete)
    outfile=rep+'bathy_all_v2'
    data_to_complete.to_csv(outfile,sep=' ', index=False, float_format='%.4f')

    data_values=data_to_complete.dropna()
    print('data_values', data_values, type(data_values))
    outfile2=rep+'test_v2'
    data_values.to_csv(outfile2,sep=' ', index=False, float_format='%.4f')

if interpolate :
    counter=0
    data=pd.read_csv(rep+'bathy_all_v2', sep=' ')#, header=None)#, names=['i', 'j', 'lon_mod', 'lat_mod', 'depth_mod',
    data_to_complete=pd.DataFrame(data)
    soustab=data_to_complete[data_to_complete['diff_depth'].isna()==True]
    print(soustab)
    print(len(list(soustab.index)))

    file_polygon=rep2+'bathy_5-10m.kml'
    print('file poly',file_polygon)
    polygon=Polygon(pd.read_csv(file_polygon, sep=' '))

    i1=70 #80
    i2=110 #90
    j1=420 #420
    j2=480 #471
    to_check = np.where((soustab['i'] >= i1) & (soustab['i']<=i2) & (soustab['j']>=j1) & (soustab['j']<= j2))[0]
    print('to check ', len(to_check), '\n', to_check)
    #print(data_to_complete[to_check])

    for v in to_check:
        print('i', v)
        #print(soustab.iloc[i])
        val_i=soustab.iloc[v]['i']
        val_j=soustab.iloc[v]['j']
        lon=soustab.iloc[v]['lon_mod']
        lat=soustab.iloc[v]['lat_mod']
        point=Point(lon,lat)
        if not point.within(polygon):
            index=np.where((data_to_complete['i']==val_i) & (data_to_complete["j"]==val_j))[0][0] # on retrouve depuis le soutableau, l'indice dand le tab à compléter
            #print('index', index, data_to_complete.iloc[index])
            #var_index = ['ind' + str(i) for i in range(1, 5)]
            couple_index=[[val_i, val_j+1],[val_i+1, val_j],[val_i, val_j-1], [val_i-1, val_j]]
            #print('couple', couple_index,'\n', couple_index[0], couple_index[0][0], len(couple_index))
            list_val, indice=[], []
            #div=len(couple_index)
            sum=0
            for i in range(len(couple_index)):
                print(i)
                print('val', couple_index[i][0],couple_index[i][1] )
                if couple_index[i][0]<10 or couple_index[i][0] > 99 or couple_index[i][1] <420 or couple_index[i][1]> 499 :
                    print('out of bond')
                    val=np.nan
                else :
                    var = np.where((data_to_complete['i'] == couple_index[i][0]) & (data_to_complete["j"] == couple_index[i][1]))[0][0]
                    val = data_to_complete.iloc[var]['depth_moy']
                    print(val)
                list_val.append(val)
                indice.append(var)
            print('indice', list_val, (data_to_complete.iloc[ind]['depth_moy'] for ind in list_val))
            #if div==0 :
            #    print('PROBLEEEEEEEEM')
            #    moy=-9999
            #moy=sum/div
            moy=np.nanmean(list_val)
            print('moy', moy)

            if moy != np.nan:
                counter=counter+1
                data_to_complete.at[index, 'depth_moy'] = moy
                data_to_complete.at[index, 'depth_kml'] = moy
                data_to_complete.at[index, 'lon_kml'] = lon #on copie toutes les case pour ne plus avoir de nan, puis supprimer toutes les lignes avec nan
                data_to_complete.at[index, 'lat_kml'] = lat

                gebco_depth = data_to_complete.iloc[index]['depth_mod']
                val_depth = moy - float(gebco_depth)
                data_to_complete.at[index, 'diff_depth'] = val_depth
                data_to_complete.at[index, 'diff_depth_moy'] = val_depth

                print('nouvel indice', data_to_complete.iloc[index])

    print('COUNTER : ', counter)
    outfile = rep + 'bathy_all_v5'
    data_to_complete.to_csv(outfile, sep=' ', index=False, float_format='%.4f')

    data_values = data_to_complete.dropna()
    print('data_values', data_values, type(data_values))
    outfile2 = rep + 'test_v5'
    data_values.to_csv(outfile2, sep=' ', index=False, float_format='%.4f')


sys.exit(1)