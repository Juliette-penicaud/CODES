# 16/02 : intersection de 2 couches sur pandas pour couper les rivières qui dépassent sur l'océan

import geopandas as gpd
import numpy as np


rep = '/home/penicaud/Documents/QGIS/'
file_river = rep + 'shp/AS/as_sword_nodes_hb43_v15.shp'
file_vietnam = rep + 'vietnam.shp'

vietnam = gpd.read_file(file_vietnam)
bbox_vietnam = tuple(vietnam.total_bounds)
river = gpd.read_file(file_river, bbox=bbox_vietnam)
river_intersectes = gpd.sjoin(river, vietnam, how='left', predicate='intersects')

river_intersectes = river_intersectes[river_intersectes['index_right'].notnull()]
river_intersectes.to_file(rep+'river_interesected.shp')

print('end')