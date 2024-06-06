# load packages
import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.mask
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np


# load modules
from census_100_file_loader import file_loader,census_loader
from census_100_geocoder import Glocer

from census_100_demands import Demands

clc_raster = {
    1: 'Sealed',
    2: 'Woody needle leaved trees',
    3: 'Woody Broadleaved deciduous trees',
    4: 'Woody Broadleaved evergreen trees',
    5: 'Low-growing woody plants (bushes, shrubs)',
    6: 'Permanent herbaceous',
    7: 'Periodically herbaceous',
    8: 'Lichens and mosses',
    9: 'Non- and sparsely-vegetated',
    10: 'Water',
    11: 'Snow and ice',
    254: 'outside area',
    255: 'No data'
}

clc_vec = {'111': 'Continuous urban fabric',
'112': 'Discontinuous urban fabric',
'121': 'Industrial or commercial units',
'122': 'Road and rail networks and associated land',
'123': 'Port areas',
'124': 'Airports',
'131': 'Mineral extraction sites',
'132': 'Dump sites',
'133': 'Construction sites',
'141': 'Green urban areas',
'142': 'Sport and leisure facilities',
'211': 'Non-irrigated arable land',
'212': 'Permanently irrigated land',
'213': 'Rice fields',
'221': 'Vineyards',
'222': 'Fruit trees and berry plantations',
'223': 'Olive groves',
'231': 'Pastures',
'241': 'Annual crops associated with permanent crops',
'242': 'Complex cultivation patterns',
'243': 'Land principally occupied by agriculture, with significant areas of natural vegetation',
'244': 'Agro-forestry areas',
'311': 'Broad-leaved forest',
'312': 'Coniferous forest',
'313': 'Mixed forest',
'321': 'Natural grasslands',
'322': 'Moors and heathland',
'323': 'Sclerophyllous vegetation',
'324': 'Transitional woodland-shrub',
'331': 'Beaches, dunes, sands',
'332': 'Bare rocks',
'333': 'Sparsely vegetated areas',
'334': 'Burnt areas',
'335': 'Glaciers and perpetual snow',
'411': 'Inland marshes',
'412': 'Peat bogs',
'421': 'Salt marshes',
'422': 'Salines',
'423': 'Intertidal flats',
'511': 'Water courses',
'512': 'Water bodies',
'521': 'Coastal lagoons',
'522': 'Estuaries',
'523': 'Sea and ocean',
}

class tile_loader:
    def __init__(self, file_loader, demands) -> None:
        
        self.file_loader = file_loader
        self.demands = demands
          
        
    def make_grid_100_quality(self, geocode_address="fuzzdress_short"):
        ''' # load relevant data
        grid_100m = self.file_loader.zensus100.copy()
        grid_100m = grid_100m[["id","geometry", "projekt"]]


        gas_cons = self.demands.copy()
        gas_cons = gas_cons.drop_duplicates(geocode_address)
        grid_100m_gas_cons = grid_100m.sjoin(gas_cons)
        grid_100m_gas_cons = grid_100m_gas_cons.groupby("id").size().rename("points_dem")

        grid_100m = self.file_loader.zensus100.copy()
        grid_100m = grid_100m[["id","geometry", "projekt"]]
        '''
    	# load grid        
        grid_100m = self.file_loader.zensus100.copy()
        grid_100m = grid_100m[["id","geometry", "projekt"]]

        # count addresses in each tile of the grid
        ahkpoints = self.file_loader.hauskoordinaten.copy()
        grid_100m_ahkpoints = grid_100m.sjoin(ahkpoints)
        grid_100m_ahkpoints = grid_100m_ahkpoints.groupby("id").size().rename("points_ahk")


        grid_100m_quality = grid_100m.merge(self.demands, left_on="id", how='outer', right_index=True)
        # merge different grids
        grid_100m_quality = grid_100m_quality.merge(grid_100m_ahkpoints, left_on="id", how='outer', right_index=True)
        grid_100m_quality['points_dem'] = grid_100m_quality['points_dem'].fillna(0)

        # calculate measures of quality for each tile
        grid_100m_quality["abs_ratio"] = grid_100m_quality["points_ahk"] - grid_100m_quality["points_dem"]
        grid_100m_quality["rel_ratio"] = grid_100m_quality["points_dem"] / grid_100m_quality["points_ahk"]
        
        grid_100m_quality = grid_100m_quality.set_index('id')
        '''
        demands = self.demands[["2020_heat"]].sjoin(grid_100m)
        demands = demands.groupby("id").sum([ "2020_heat"])
        demands = demands.drop(["index_right"], axis=1)
        
        # merge demands to grid
        grid_100m_quality = grid_100m_quality.merge(demands, left_index=True, right_index=True, how="left")
        '''
        return grid_100m_quality
    
    
    def make_grid_100_building(self, path_bclass_information="building_class_information.xlsx", feature_col="btypology"):
        buildings = self.file_loader.building_alkis.copy()
        grid = self.file_loader.zensus100.copy()
        ahk_b = self.file_loader.hauskoordinaten.copy()

        bclass_info = pd.read_excel(path_bclass_information)
        bclass_info_view = bclass_info[bclass_info["code"].notna()]
        bclass_info_view["code"] = bclass_info_view["code"].astype(int).copy()
        bclass_info_view = bclass_info_view[["code", "bclass", "heated", "residential", "btypology", "btypology_II"]]

        buildings["bclass"] = buildings["bclass"].map(bclass_info_view.set_index("code")["bclass"]).fillna(buildings["bclass"])

        only_main = False
        if only_main:
            buildings = buildings.sjoin(ahk_b[["ahk_id", "geometry"]])

        buildings = pd.merge(buildings, bclass_info_view, left_on="bclass", right_on="bclass")

        only_heated = False
        if only_heated:
            buildings = buildings[buildings["heated"] == 1]

            

        buildings = buildings.to_crs('EPSG:25832')
        buildings["area"] = buildings.area
        buildings["geometry"] = buildings.centroid
        grid  = grid.to_crs('EPSG:25832')
        buildings = buildings[["geometry", feature_col, "area"]]
        buildings = buildings.sjoin(grid[["id", "geometry"]])

        buildings = buildings.pivot_table(values='area', index='id', columns=feature_col, aggfunc='sum', fill_value=0, margins=False, dropna=True)

        relative_buildings = False
        if relative_buildings:
            row_totals = buildings.sum(axis=1)
            # Divide each value by its row total to get relative values
            buildings = buildings.div(row_totals, axis=0)
        return buildings

    def make_grid_100_clc(self, path_clc='daten\\raster\\CLCplus_2018_010m.tif'):
        clc = self.make_grid_100_from_raster(path=path_clc)
        clc['count_me'] = 1
        clc = pd.pivot_table(clc, index='index_right', columns='value', values='count_me', fill_value=0, aggfunc='count')
        clc = clc.rename(columns=clc_raster) 
        return clc
    
    def make_grid_100_sbu(self, path_sbu='daten\\raster\\SBU_2018_100m_03035_V1_0.tif'):
        points_with_ids = self.make_grid_100_from_raster(path=path_sbu)
        points_with_ids = points_with_ids.set_index('index_right')
        return points_with_ids['value'].rename("sbu")
    
    
    def make_grid_100_imd(self, path_imd='daten\\raster\\IMD_2018_100m_03035_V2_0.tif'):
        points_with_ids = self.make_grid_100_from_raster(path=path_imd)
        points_with_ids = points_with_ids.set_index('index_right')
        return points_with_ids['value'].rename("imd")

    
    
    def make_grid_100_from_raster(self, path='',):
        gdf = self.file_loader.zensus100.copy()
        gdf = gdf.set_index('id')
        out_image = None
        # Open the raster file
        with rasterio.open(path) as src:
            # Read the raster's metadata
            out_meta = src.meta.copy()
            raster_crs = src.crs

            # Reproject the GeoDataFrame to match the raster CRS
            gdf = gdf.to_crs(raster_crs)

            # Mask the raster with the polygons
            out_image, out_transform = rasterio.mask.mask(src, gdf.geometry, crop=True, filled=False)

            # Update metadata for the output file
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
        
        # Get the affine transformation
        affine_transform = out_transform

        # Create a list to store the points and values
        points = []

        # Iterate through the masked raster
        for j in range(out_image.shape[1]):  # y
            for i in range(out_image.shape[2]):  # x
                # If the pixel is not masked
                if out_image[0, j, i] is not np.ma.masked:
                    # Get the center of the pixel
                    x, y = affine_transform * (i + 0.5, j + 0.5)
                    # Append the point, raster value to the list
                    points.append((x, y, out_image[0, j, i]))
                    
        
        # Convert the list of points to a GeoDataFrame
        points_gdf = pd.DataFrame(points, columns=['x', 'y', 'value'])
        geometry = [Point(xy) for xy in zip(points_gdf.x, points_gdf.y)]
        points_gdf = gpd.GeoDataFrame(points_gdf, geometry=geometry)

        # Set the CRS to match the raster
        points_gdf.set_crs(raster_crs, inplace=True)

        # Perform a spatial join to assign polygon IDs to the points
        return gpd.sjoin(points_gdf, gdf, how='left', op='within')
        
