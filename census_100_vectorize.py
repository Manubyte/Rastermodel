import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from rasterstats import zonal_stats

# Function to generate polygons from raster
def raster_to_vector_points(raster_path='raster\SBU_2018_100m_03035_V1_0.tif', crs='EPSG:3035', export='IMD_2018_100m_03035_V2_0.gpkg'):
    # Load the raster data
    
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)  # Read the first band
        raster_meta = src.meta  # Get metadata
        
    transform = src.transform
    polygons = []
    values = []
    
    for row in range(raster_data.shape[0]):
        for col in range(raster_data.shape[1]):
            value = raster_data[row, col]
            if value != 0:  # Assuming 0 is nodata; adjust as needed
                x, y = transform * (col, row)
                polygons.append(box(x, y, x + transform[0], y + transform[4]))
                values.append(value)
    
    raster_as_vector = gpd.GeoDataFrame({'geometry': polygons, 'value': values})
    raster_as_vector.crs = crs
    
    if export:
        raster_as_vector.to_file(export, driver='GPKG')
        
    return raster_as_vector


def raster_to_vector_research_area(file_loader, raster_path='raster\SBU_2018_100m_03035_V1_0.tif'):
    raster = rasterio.open(raster_path)
    
    # Load the vector grid (GeoDataFrame)
    vector_grid_path = 'research_area.gpkg'
    vector_grid = gpd.read_file(vector_grid_path)
    vector_grid = file_loader.zensus100.copy()

    if vector_grid.crs != raster.crs:
        vector_grid = vector_grid.to_crs(raster.crs)
        
    with raster as src:
        nodata = src.nodata
        print(f'No data value: {nodata}')
        
    # Perform zonal statistics
    stats = zonal_stats(vector_grid, raster_path, stats=['mean', 'median', 'sum', 'std', 'min', 'max'])

    # Convert the stats to a DataFrame
    
    stats_df = pd.DataFrame(stats)

    # Add the stats back to the vector grid GeoDataFrame
    vector_grid = vector_grid.join(stats_df)
    # Save the result to a new shapefile
    vector_grid.to_file('research_area_IMD_2018_100m_03035_V2_0.gpkg', driver='GPKG')
    

def pivot