# class to load files

import geopandas as gpd
import pandas as pd
import fiona
from os import listdir, getcwd
from os.path import isfile, join


class file_loader:
    # set to collect all names of projects
    projekte = set()
    
    def __init__(self, path=None, projekte="all", dataframes='all', database="/bigdata", crs='EPSG:4326') -> None:
    # Use current directory as default path if none provided
        if not path:
            path = getcwd()
    
        self.path_demand_gas = path + database + "/demands/gas/"
        self.path_demand_el = path + database + "/demands/el/"
        self.path_demand_heat_net = path + database + "/demands/heat_net/"
        self.path_building_alkis = path + database + "/buildings_alkis/"
        self.path_parcels = path + database + "/parcels/"
        self.path_geoloc = path + database + "/g_loc/"
        self.path_servicearea = path + database + "/service_area/"
        self.path_hauskoordinaten = path + database + "/HK/projekte/"
        self.path_zensus100 = path + database + "/zensus100/"
        
        self.crs = crs
        
        # Set default dataframes if none provided
        if dataframes is None:
            dataframes = ['gas', 'el', 'heat_net', 'building_alkis', 'parcels', 'geoloc', 'servicearea', 'hauskoordinaten', 'zensus']
    
        # Load dataframes conditionally based on user input
        if 'all' in dataframes or 'gas' in dataframes:
            self.demand_gas = self.load_df(self.path_demand_gas, projekte, id_prefix="gas")
        if 'all' in dataframes or 'el' in dataframes:
            self.demand_el = self.load_df(self.path_demand_el, projekte, id_prefix="el")
        if 'all' in dataframes or 'heat_net' in dataframes:
            self.demand_heat_net = self.load_df(self.path_demand_heat_net, projekte, id_prefix="heat_net")
        if 'all' in dataframes or 'building_alkis' in dataframes:
            self.building_alkis = self.load_gdf(self.path_building_alkis, projekte, "b")
        if 'all' in dataframes or 'parcels' in dataframes:
            self.parcels = self.load_gdf(self.path_parcels, projekte, id_prefix="p")
        if 'all' in dataframes or 'geoloc' in dataframes:
            self.geoloc = self.load_gdf(self.path_geoloc, projekte, id_prefix="googleV3")
        if 'all' in dataframes or 'servicearea' in dataframes:
            self.servicearea = self.load_gdf(self.path_servicearea, projekte, id_prefix="service_area")
        if 'all' in dataframes or 'hauskoordinaten' in dataframes:
            # Filter paths by specific projects if provided
            self.hauskoordinaten = self.load_gdf(self.path_hauskoordinaten, projekte, id_prefix="ahk")
        if 'all' in dataframes or 'zensus' in dataframes:
            self.zensus100 = self.load_gdf(self.path_zensus100, projekte, id_prefix="zensus")
            

    def load_df(self, path, projecs, id_prefix):
        
        
        print("start to load files from", path)
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        dfs = []
        for element in onlyfiles:
            projekt = element.split('_')[-1:][0].split('.')[:1][0]
            if projekt in projecs or "all" in projecs:
                print("load file", element)
                df = pd.read_excel(join(path, element))
                df["projekt"] = projekt
                df.columns = df.columns.astype(str)
                file_loader.projekte.add(projekt)
                dfs.append(df)
            
        df = pd.DataFrame(pd.concat(dfs, ignore_index=True))
        if id_prefix:
            df = df.reset_index(names=str(id_prefix + "_id"))
            
        return df
    
    def load_gdf(self, path, project, id_prefix):
        
        
        print("start to load files from", path)
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        gpkgs = []
        for element in onlyfiles:
            projekt = element.split('_')[-1:][0].split('.')[:1][0]
            if "all" in project or projekt in project:
                layers = fiona.listlayers(join(path, element))
                print(element, "has layers:", layers, "loading layer:", layers[0])
                gdf = gpd.read_file(join(path, element), layer=layers[0])
                gdf = gdf.to_crs(crs=self.crs)
                gdf["projekt"] = projekt
                gdf.columns = gdf.columns.astype(str)
                file_loader.projekte.add(projekt)
                gpkgs.append(gdf)
            
        gdf = gpd.GeoDataFrame(pd.concat(gpkgs, ignore_index=True), crs=self.crs)
        
        if id_prefix:
            gdf = gdf.reset_index(names=str(id_prefix + "_id"))
            
        return gdf
    
    
class census_loader:
    def __init__(self, select_tables=False, select_features=False):
        '''
        print('start loading bev')
        bev = pd.read_csv('C:\\Users\\fischerm\\BBH Consulting AG\\Intern_BBHC GF Integrierte Netzplanung - Dokumente\\Abschlussarbeiten\Manuel Fischer\\01_Daten\\zensus\\Zensus_Bevoelkerung_100m-Gitter.csv', encoding_errors="ignore", delimiter=";")
        print('start loading whg')
        whg = pd.read_csv('C:\\Users\\fischerm\\BBH Consulting AG\\Intern_BBHC GF Integrierte Netzplanung - Dokumente\\Abschlussarbeiten\Manuel Fischer\\01_Daten\\zensus\\Wohnungen100m.csv', encoding_errors="ignore")
        print('start loading geb')
        geb = pd.read_csv('C:\\Users\\fischerm\\BBH Consulting AG\\Intern_BBHC GF Integrierte Netzplanung - Dokumente\\Abschlussarbeiten\Manuel Fischer\\01_Daten\\zensus\\Geb100m.csv', encoding_errors="ignore")
        print('start loading hoh')
        hoh = pd.read_csv('C:\\Users\\fischerm\\BBH Consulting AG\\Intern_BBHC GF Integrierte Netzplanung - Dokumente\\Abschlussarbeiten\Manuel Fischer\\01_Daten\\zensus\\Haushalte100m.csv', encoding_errors="ignore")
        print('start loading dmo')
        dmo = pd.read_csv('C:\\Users\\fischerm\\BBH Consulting AG\\Intern_BBHC GF Integrierte Netzplanung - Dokumente\\Abschlussarbeiten\Manuel Fischer\\01_Daten\\zensus\\Bevoelkerung100M.csv', sep=';', encoding_errors="ignore")
        print('start loading fam')
        fam = pd.read_csv('C:\\Users\\fischerm\\BBH Consulting AG\\Intern_BBHC GF Integrierte Netzplanung - Dokumente\\Abschlussarbeiten\Manuel Fischer\\01_Daten\\zensus\\Familie100m.csv', encoding_errors="ignore")
        ''' 
        
        bev = pd.read_csv('daten\\zensus\\bev_100m_short.csv')
        whg = pd.read_csv('daten\\zensus\\whg_100m_short.csv')
        geb = pd.read_csv('daten\\zensus\\geb_100m_short.csv')
        hoh = pd.read_csv('daten\\zensus\\hoh_100m_short.csv')
        dmo = pd.read_csv('daten\\zensus\\dmo_100m_short.csv')
        fam = pd.read_csv('daten\\zensus\\fam_100m_short.csv')
        
        
        self.files = {
            'Bevölkerung' : bev,
            'Wohnung' : whg,
            'Gebäude' : geb,
            'Haushalt' : hoh,
            'Demographie' : dmo,
            'Familie' : fam
            }
    
        
        self.census_pop_abs = None
        self.census_whg_abs = None
        self.census_bui_abs = None
        self.census_hoh_abs = None
        self.census_dmo_abs = None
        self.census_fam_abs = None
        
        self.census_pop_rel = None
        self.census_whg_rel = None
        self.census_bui_rel = None
        self.census_hoh_rel = None
        self.census_dmo_rel = None
        self.census_fam_rel = None
        
        self.pointer_abs = {
            'Bevölkerung' : self.census_pop_abs,
            'Wohnung' :self.census_whg_abs,
            'Gebäude' : self.census_bui_abs,
            'Haushalt' : self.census_hoh_abs,
            'Demographie' :self.census_dmo_abs,
            'Familie' : self.census_fam_abs
            }
        
        self.pointer_rel = {
            'Bevölkerung' : self.census_pop_rel,
            'Wohnung' :self.census_whg_rel,
            'Gebäude' : self.census_bui_rel,
            'Haushalt' : self.census_hoh_rel,
            'Demographie' :self.census_dmo_rel,
            'Familie' : self.census_fam_rel
            }
        
        self.census_abs = None
        self.census_rel = None
        
        self.census_ger_predict = None
        self.complete_tiles = None
        # self.load_census(select_tables, select_features)


    def load_census_table(self, prefix='', select_tables=False, select_features=False):
        # Filter based on grid index        
        if select_tables:
            load_tabe = input(f"load table: {prefix}? (y/n)")
            if load_tabe.lower() != 'y':
                return
         
        if prefix == 'Bevölkerung':
            pop = self.files[prefix]
            pop = pop.set_index('Gitter_ID_100m')
            pop = pop['Einwohner']
            self.pointer_abs[prefix] = pop
            self.pointer_rel[prefix] = pop
            return
         
        df = self.files[prefix]
        if select_features:
            for element in df["Merkmal"].unique():
                answer = input( f"keep {element} | (y, n):")
                if answer.lower() != 'y':
                    df = df[df['Merkmal'] != element]
        
        # Pivot the DataFrame with absolut values
        df_abs = df.pivot(index="Gitter_ID_100m", columns=["Merkmal", "Auspraegung_Text"], values='Anzahl')
        df_abs.columns = [prefix + ''.join(str(col)).strip() for col in df_abs.columns.values]
        df_abs = df_abs.fillna(0)
        self.pointer_abs[prefix] = df_abs
        
        
        # Pivot the DataFrame with relative values
        df['Anzahl_rel'] = df['Anzahl'] / df.groupby(['Gitter_ID_100m', 'Merkmal'], sort=False)['Anzahl'].transform('sum') * 100
        df_rel = df.pivot(index="Gitter_ID_100m", columns=["Merkmal", "Auspraegung_Text"], values='Anzahl_rel')
        df_rel.columns = [prefix + ''.join(str(col)).strip() for col in df_rel.columns.values]
        df_rel = df_rel.fillna(0)
        self.pointer_rel[prefix] = df_rel
        

    def load_census(self, select_tables=False, select_features=False):
        dfs_abs = []
        dfs_rel = []
        for key in self.files:
            self.load_census_table(key, select_tables, select_features)
            df_abs = self.pointer_abs[key] 
            if df_abs is not None:
                dfs_abs.append(df_abs)
            df_rel = self.pointer_rel[key] 
            if df_rel is not None:
                dfs_rel.append(df_rel)
        
        self.census_abs = pd.concat(dfs_abs, axis=1)
        self.census_rel = pd.concat(dfs_rel, axis=1)
        
    def avaliable_tiles(self):
        
        s = set(self.pointer_abs['Bevölkerung'].index)
        for key in self.pointer_abs:
            df_abs = self.pointer_abs[key] 
            if df_abs is not None:
                s = s.intersection(df_abs.index)
            
        self.complete_tiles = s
        return s
        
    # kaputt?
    def trim_german_and_trainingset(self):
        # Assuming grid_100m_all is defined elsewhere
        common_columns = self.census_ger.columns.intersection(self.zensus100.columns)

        if '2020' not in common_columns:
            common_columns = pd.Index(['2020']).append(common_columns)

        # Now filter both dataframes to only keep these columns
        self.census_ger = self.census_ger[common_columns]
        self.census_ger_training = self.zensus100[common_columns].copy()
