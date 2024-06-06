import pandas as pd
import geopandas as gpd

import re
from shapely import wkt
from thefuzz import fuzz, process


class Glocer:
    def __init__(self, loader) -> None:
        # check if loader has loaded all dataframes 
        self.demands = loader.demand_gas.copy()
        self.ahks = loader.hauskoordinaten.copy()
        try:
            self.googleV3 = loader.geoloc.copy()
            self.prep_googleV3()
        except:
            print('no googleV3 loaded. Cant golocate with google')
        self.crs = loader.crs 
        
        self.prep_ahks()
        self.prep_dems()
        
        
    def save_geo_dataframe(self, gdf, filename):
        full_path = f"{filename}.gpkg"
        gdf.to_file(full_path, driver="GPKG")
    
    def load_geo_dataframe(self, filename):
        full_path = f"{filename}.gpkg"
        return gpd.read_file(full_path)

    def save_geo_excel(self, gdf, filename):
        # Convert GeoDataFrame to DataFrame by converting geometry to WKT
        df = pd.DataFrame(gdf.copy())
        df['geometry'] = df['geometry'].apply(lambda x: x.wkt if x is not None else None)

        # Save to Excel with the 'openpyxl' engine
        df.to_excel(f"{filename}.xlsx", engine='openpyxl', index=False)
        print(f"Saved GeoDataFrame to {filename}.xlsx successfully!")

    def load_geo_excel(self, filename, crs='EPSG:4326'):
        # Load data from Excel
        df = pd.read_excel(f"{filename}.xlsx", engine='openpyxl')

        # Convert WKT string back to geometry
        df['geometry'] = df['geometry'].apply(wkt.loads if pd.notna else None)

        # Convert DataFrame to GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs)
        print(f"Loaded {filename}.xlsx into GeoDataFrame successfully!")
        return gdf
    
    
    def prep_ahks(self):
        relevant_columns = ["oid", "str", "hnr", "adz", "plz", "gmd", "projekt", "ostwert", "nordwert", "geometry"]
        self.ahks = self.ahks[relevant_columns]
        self.ahks["hnr+"] = self.ahks["hnr"].astype(str) + self.ahks["adz"].fillna("").str.lower()
        self.ahks["address"] = self.ahks["str"].astype(str) + " " + self.ahks["hnr+"] + ", " + self.ahks["plz"] + " " + self.ahks["gmd"] + ", Germany"
        
    def prep_dems(self):
        self.demlocs = self.demands[["address", "projekt"]].drop_duplicates(subset=["address"])
        address_parts = self.demlocs["address"].str.split(",", expand=True)
        self.demlocs[["strhnr", "postmun", "country"]] = address_parts
        
        self.demlocs['street'] = self.demlocs['strhnr'].str.extract(r'^(.+?)\d+')
        self.demlocs['hnr'] = self.demlocs['strhnr'].str.extract('(\d[\d\sA-Za-z-]*)$')
        self.demlocs[["postal", "municipal"]] = self.demlocs["postmun"].str.strip().str.split(" ", n=1, expand=True)
        
    def prep_googleV3(self):
        self.googleV3 = self.googleV3[["address", "address_geolocator", "geometry"]].drop_duplicates(subset=["address"])

    def fuzz_street(self, threshold=80, export_excel='', manual_correct=False):
        plzs = set(self.demlocs["postal"]).intersection(set(self.ahks["plz"]))
        print("Addresses are compared within the postal codes:", plzs)
        export = []

        for plz in plzs:
            print("starting with postal code:", plz)
            demand_streets = set(self.demlocs[self.demlocs["postal"] == plz]["street"].dropna())
            ahk_streets = set(self.ahks[self.ahks["plz"] == plz]["str"].dropna())

            for street in demand_streets:
                best_match = process.extractOne(street, ahk_streets, scorer=fuzz.ratio)
                if best_match and best_match[1] >= threshold:
                    street_mask = (self.demlocs["postal"] == plz) & (self.demlocs["street"] == street)
                    self.demlocs.loc[street_mask, "fuzz_street"] = best_match[0]
                elif manual_correct:
                    print(f"manual correction of fuzz street matching. schore {best_match[1]}:")
                    print(f"Demand: {street}")
                    print(f"Match: {best_match[0]}\n")
                    
                    answer = input(f"| {street} VS {best_match[0]} | (y/n):")
                    if answer.lower() == 'y':
                        street_mask = (self.demlocs["postal"] == plz) & (self.demlocs["street"] == street)
                        self.demlocs.loc[street_mask, "fuzz_street"] = best_match[0]
                        print('matched!')
                    else:
                        print('Not matched!')
                    
                export.append([plz, street, best_match[0] if best_match else "", best_match[1] if best_match else ""])

        if export_excel:
            pd.DataFrame(export, columns=["plz", "dem_street", "ahk_street", "score_ratio"]).to_excel(f"{export_excel}.xlsx")

    # Similarly refactor other methods like fuzz_hnr, geolocate_demlocs, etc., following the patterns here.
    def fuzz_hnr(self, threshold=100, export_excel=''):
        print("Info: you should previously run: fuzz_street")
        export = []

        self.demlocs["postal_street"] = self.demlocs["postal"] + " " + self.demlocs["fuzz_street"]
        self.ahks["postal_street"] = self.ahks["plz"] + " " + self.ahks["str"]
        plzstrs = set(self.demlocs["postal_street"]).intersection(set(self.ahks["postal_street"]))

        for plzstr in plzstrs:
            print("----", plzstr, "----")
            demand_hnrs = set(self.demlocs[self.demlocs["postal_street"] == plzstr]["hnr"].dropna())
            ahk_hnrs = set(self.ahks[self.ahks["postal_street"] == plzstr]["hnr+"].dropna())

            for hnr in demand_hnrs:
                hnr_cleaned = hnr.replace(" ","").lower()
                hnr_splits = re.split(r'[^a-zA-Z0-9\s]', hnr_cleaned)

                best_match = [None, 0]
                for split_hnr in hnr_splits:
                    match = process.extractOne(split_hnr, ahk_hnrs, scorer=fuzz.ratio, score_cutoff=threshold)
                    if match and match[1] > best_match[1]:
                        best_match = match
                    
                    # Add step for manual correction
                    # for this dont extractOne but get top three matches and let the user decide by inputting between 1 and 3

                if best_match[0] and best_match[1] >= threshold:
                    street_mask = (self.demlocs["postal_street"] == plzstr) & (self.demlocs["hnr"] == hnr)
                    self.demlocs.loc[street_mask, "fuzz_hnr"] = best_match[0]

                export.append([plzstr, hnr, best_match[0] if best_match[0] else "", best_match[1]])

        if export_excel:
            results = pd.DataFrame(export, columns=["plzstr", "dem_hnr", "ahk_hnr", "score_ratio"])
            results.to_excel(f"{export_excel}.xlsx")

    def geolocate_demlocs(self, ahk=False, googleV3=False, ahk_short=True):
        
        self.demlocs["fuzzdress_short"] = self.demlocs["fuzz_street"] + " " + self.demlocs["fuzz_hnr"] + ", " + self.demlocs["postal"]
            
        if ahk:
            self.demlocs["fuzzdress"] = self.demlocs["fuzz_street"] + " " + self.demlocs["fuzz_hnr"] + ", " + self.demlocs["postal"] + " " + self.demlocs["municipal"] + ", Germany"
            self.demlocs = self.demlocs.reset_index().merge(
                self.ahks, how="left", left_on="fuzzdress", right_on="address", suffixes=('', '_ahk')
            ).drop_duplicates(["index"]).drop(columns=["index"])
            self.demlocs.loc[self.demlocs["geometry"].notna(), "georef_type"] = "ahk"

        if ahk_short:
            self.demlocs["fuzzdress_short"] = self.demlocs["fuzz_street"] + " " + self.demlocs["fuzz_hnr"] + ", " + self.demlocs["postal"]
            self.ahks['fuzzdress_short'] = self.ahks['address'].str.replace(r'(?<=\d{5}) [A-Za-z\s,]+.*$', '', regex=True)
            self.demlocs = self.demlocs.reset_index().merge(
                self.ahks, how="left", left_on="fuzzdress_short", right_on="fuzzdress_short", suffixes=('', '_ahk')
            ).drop_duplicates(["index"]).drop(columns=["index"])
            self.demlocs.loc[self.demlocs["geometry"].notna(), "georef_type"] = "ahk_short"

        if googleV3:
            self.demlocs = self.demlocs.reset_index().merge(
                self.googleV3[["geometry", "address"]], how="left", on="address", suffixes=('', '_GoogleV3')
            )
            if ahk:
                self.demlocs["geometry"] = self.demlocs["geometry"].fillna(self.demlocs["geometry_GoogleV3"])
                self.demlocs.loc[self.demlocs["georef_type"].isna() & self.demlocs["geometry"].notna(), "georef_type"] = "googleV3"
            else:
                self.demlocs.loc[self.demlocs["geometry"].notna(), "georef_type"] = "googleV3"
            self.demlocs = self.demlocs.drop_duplicates(["index"]).drop(columns=["index"])

        self.demlocs = gpd.GeoDataFrame(self.demlocs, geometry="geometry", crs=self.crs)
        
        
    def geolocate_demands(self):
        demlocs = self.demlocs[["address", "geometry", "fuzzdress_short"]]
        self.demands = self.demands.merge(demlocs, on="address", how="left")
        self.demands = gpd.GeoDataFrame(self.demands, geometry="geometry", crs=self.crs)
