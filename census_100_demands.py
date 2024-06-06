#import glocer
# load packages
import geopandas as gpd
import pandas as pd
import seaborn as sns

# load modules
from census_100_file_loader import file_loader
from census_100_geocoder import Glocer

# have a demand and a loader

class Demands:
    
    def __init__(self, loader, demand_path='') -> None:
        self.loader = loader
        self.efficency = {
            'Gas': 0.85,
            'Wärmenetz': 0.95,
            'Wärmepumpe': 3.5,
            'Nachtspeicher': 0.95
        }
        
        self.geolocations = None
        self.demands = None
        self.demands_heat = None
        
        self.demands_interim = None
        self.demands_interim_group = None
        self.raster_demand = None
    
        if demand_path == '':
            self.geolocations = Glocer(self.loader)
            # create new geocodng, based on ahk only
            self.geolocations.fuzz_street(threshold=80, export_excel='', manual_correct=True)
            self.geolocations.fuzz_hnr()
            self.geolocations.geolocate_demlocs(ahk=False, googleV3=False, ahk_short=True)
            self.geolocations.geolocate_demands()
            self.demands = self.geolocations.demands
            self.demands.columns = self.demands.columns.astype(str)
        else:
            self.demands = gpd.read_file(demand_path)
            self.demands.columns = self.demands.columns.astype(str)
            
            
    def culc_heat_demand(self):
        
        
        # Define a function to apply the factor based on string value
        def apply_factor(row):
            string_value = row['technology']
            factor = self.efficency.get(string_value, 1.0)  # Default to 1.0 if no mapping found
            return row['2020'] * factor
    
        dems = self.demands.copy()
        dems.columns = dems.columns.astype(str)

        # calculate the heat demand
        dems['2020_heat'] = dems.apply(lambda row: apply_factor(row), axis=1)
        dems['factor'] = dems['factor'].fillna(1)
        dems['2020_heat'] = dems['2020_heat'] * dems['factor']
        self.demands_heat = dems.copy()
        
        grid_100m = self.loader.zensus100.copy()
        grid_100m = grid_100m[["id","geometry"]]
        self.raster_demand = grid_100m.sjoin(dems[['geometry', '2020_heat']])
        self.raster_demand.columns = self.raster_demand.columns.astype(str)
        self.raster_demand = self.raster_demand.drop('geometry', axis = 1).groupby("id").sum()
        self.raster_demand = self.raster_demand.drop(["index_right"], axis=1) 
        
        # add amount of heat connections to demands
        heat_cons = dems
        heat_cons = heat_cons.drop_duplicates("fuzzdress_short")
        grid_100m_gas_cons = grid_100m.sjoin(heat_cons)
        grid_100m_gas_cons = grid_100m_gas_cons.groupby("id").size().rename("points_dem")
        
        self.raster_demand = pd.concat([grid_100m_gas_cons, self.raster_demand], axis=1)

               
            
'''
    def culc_heat_demand_by_tech(self):
        
        
        # Define a function to apply the factor based on string value
        def apply_factor(row):
            string_value = row['technology']
            factor = self.efficency.get(string_value, 1.0)  # Default to 1.0 if no mapping found
            return row['2020'] * factor
    
        dems = self.demands.copy()
        dems.columns = dems.columns.astype(str)

        # calculate the heat demand
        dems['2020_heat'] = dems.apply(lambda row: apply_factor(row), axis=1)
        dems['factor'] = dems['factor'].fillna(1)
        dems['2020_heat'] = dems['2020_heat'] * dems['factor']
        self.demands_interim = dems.copy()
        
        # HIER GEHT WAS SCHIEF
        # aggegate demands by address and technology
        dems = dems[['2020_heat', 'address', 'technology']].groupby(['address', 'technology']).sum()
        self.demands_interim_group = dems.copy()
        dems = dems.pivot_table(index='address', columns=['technology'], values='2020_heat')
        dems.columns = [''.join(str(col)).strip() for col in dems.columns.values]
        dems = dems.fillna(0)
        dems['2020_heat'] = dems.loc[:, dems.columns != 'address'].sum(axis=1)
        dems = dems.groupby('address').sum()
        dems = pd.merge(self.demands[['address', 'geometry']], dems, left_on='address', right_index=True, how='right')

        grid_100m = self.loader.zensus100.copy()
        grid_100m = grid_100m[["id","geometry", "projekt"]]
    
        self.raster_demand = dems.sjoin(grid_100m)
        self.raster_demand.columns = self.raster_demand.columns.astype(str)
        self.raster_demand = self.raster_demand.drop(['geometry', 'address'], axis = 1).groupby("id").sum()
        self.raster_demand = self.raster_demand.drop(["index_right"], axis=1)'''