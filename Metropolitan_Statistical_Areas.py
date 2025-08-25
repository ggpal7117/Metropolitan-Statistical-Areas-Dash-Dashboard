#http://127.0.0.1:7117/
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.offline import plot
from plotly.subplots import make_subplots
import dash
from dash import html, dcc
import seaborn as sns
import matplotlib.pyplot as plt
from dash.dependencies import Input, Output
import requests
import json
from pygris import core_based_statistical_areas
from pygris.utils import shift_geometry
import geopandas as gpd
import fredapi as fd
from fredapi import Fred
from datetime import datetime, timedelta
import pickle
import os
from dotenv import load_dotenv
from pathlib import Path


# Load the .env file
# load_dotenv()

# census_key = os.getenv("CENSUS_KEY")
# fred_key = os.getenv("FRED_KEY")

# print("CENSUS:", census_key)
# print("FRED:", fred_key)

# census_key = keys_df.query('Key_Name == "Census"').Key.values[0]
# fred_key = keys_df.query('Key_Name == "Fred"').Key.values[0]


census_key = os.getenv("CENSUS_API_KEY")
fred_key = os.getenv("FRED_API_KEY")


fred = Fred(api_key=fred_key)
metro_str = "metropolitan statistical area/micropolitan statistical area"


# -------------------------------------------------------------------- Functions ⬇️--------------------------------------------------------------------
# -- 1 --
def scrape_data(geo, ids, id_labels):
    """
    Get Data Depending On Geography and data by requesting series data through api request
    """
    # Scrape Most Recent Census Data
    HOST = "https://api.census.gov/data"
    year_recent = "2023"
    dataset = "acs/acs5"
    base_url = "/".join([HOST, year_recent, dataset])

    predicates = {}
    predicates['get'] = ",".join(ids)
    predicates["for"] = f"{geo}:*"
    predicates['key'] = census_key
        
    r = requests.get(base_url, params=predicates)

    return pd.DataFrame(r.json()[1:], columns = id_labels)
    


# ============ For Detailed Age/Gender Data ============
def change_bins(df, gender):
    """
    Transform Age Columns to Larger Bins
    Inputs: df, st: males or females
    Output: df
    """
    
    df[f"{gender}_under_9"] = df[f"{gender}_under_5"] + df[f'5-9_years_{gender}']
    df = df.drop(columns = [f"{gender}_under_5", f'5-9_years_{gender}'])

    df[f"10-17_years_{gender}"] = df[f"10-14_years_{gender}"] + df[f"15-17_years_{gender}"]
    df = df.drop(columns = [f"10-14_years_{gender}", f"15-17_years_{gender}"])

    df[f"18-24_years_{gender}"] = df[f"18-19_years_{gender}"] + df[f"20_years_{gender}"] + df[f"21_years_{gender}"] + df[f"22-24_years_{gender}"]
    df = df.drop(columns = [f"18-19_years_{gender}", f"20_years_{gender}", f"21_years_{gender}", f"22-24_years_{gender}"])

    df[f"25-34_years_{gender}"] = df[f"25-29_years_{gender}"] + df[f"30-34_years_{gender}"]
    df = df.drop(columns = [f"25-29_years_{gender}", f"30-34_years_{gender}"])

    df[f"35-44_years_{gender}"] = df[f"35-39_years_{gender}"] + df[f"40-44_years_{gender}"]
    df = df.drop(columns = [f"35-39_years_{gender}", f"40-44_years_{gender}"])

    df[f"45-54_years_{gender}"] = df[f"45-49_years_{gender}"] + df[f"50-54_years_{gender}"]
    df = df.drop(columns = [f"45-49_years_{gender}",f"50-54_years_{gender}"])

    df[f"55-66_years_{gender}"] = df[f"55-59_years_{gender}"] + df[f"60-61_years_{gender}"] + df[f"62-64_years_{gender}"] + df[f"65-66_years_{gender}"]
    df = df.drop(columns=[f"55-59_years_{gender}", f"60-61_years_{gender}", f"62-64_years_{gender}", f"65-66_years_{gender}"])

    df[f"67-79_years_{gender}"] = df[f"67-69_years_{gender}"] + df[f"70-74_years_{gender}"] + df[f"75-79_years_{gender}"]
    df = df.drop(columns = [f"67-69_years_{gender}", f"70-74_years_{gender}", f"75-79_years_{gender}"])

    df[f"{gender}_over_80"] = df[f"80-84_years_{gender}"] + df[f"{gender}_over_85"]
    df = df.drop(columns = [f"80-84_years_{gender}", f"{gender}_over_85"])

    return df
# ============ For Detailed Age/Gender Data ============


#====================================================== Race Demographics Page ==========================================
def process_region_df(region_df: pd.DataFrame, met_str: str, top_100_msa: list, scrape_data_fn) -> pd.DataFrame:
    """
    Processes regional immigrant data from Census API.
    """
    
    country_names = region_df.Country_Name.tolist()
    
    # Pull raw ACS data
    full_data = scrape_data_fn(
        met_str,
        ["NAME"] + region_df.Name.tolist(),
        ["Name"] + country_names + ["GEOID"]
    )
    
    # Trim MSA label suffix (usually ' Metro Area')
    full_data["Name"] = full_data["Name"].str[:-11]
    
    # Filter only valid MSA
    full_data = full_data[full_data["Name"].isin(top_100_msa)]
    
    # Ensure proper type
    full_data[country_names] = full_data[country_names].astype(int)
    
    # Melt for graphing
    melted = pd.melt(
        full_data,
        id_vars=["Name", "GEOID"],
        var_name="Country",
        value_name="Population"
    )
    return melted

def load_or_process_region_df(region_name: str, region_df: pd.DataFrame, met_str: str, top_100_msa: list, scrape_data_fn, cache_dir="cached_regions") -> pd.DataFrame:
    """
    Loads region immigrant data from CSV cache or processes and saves it.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{region_name}_immigrants.csv")

    if os.path.exists(cache_path):
        #print(f"Loaded cached data for {region_name} from {cache_path}")
        return pd.read_csv(cache_path)

    print(f"No cache for {region_name}, processing fresh data...")
    processed_df = process_region_df(region_df, met_str, top_100_msa, scrape_data_fn)
    processed_df.to_csv(cache_path, index=False)
    return processed_df
#====================================================== Race Demographics Page ==========================================

# ================================================================ Income Bins ================================================================




# -------------------------------------------------------------------- Functions ⬆️--------------------------------------------------------------------


# ---------------------------------------------------------------------  SECTION 1 ⬇️--------------------------------------------------------------------- #
# This Section is stricly for GETTING DATA, preprocessing that data, and merging data frames

# Population -- Wikepedia
# msa_pop = pd.read_html("https://en.wikipedia.org/wiki/Metropolitan_statistical_area")[1][:100]


url = 'https://en.wikipedia.org/wiki/Metropolitan_statistical_area'
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

response = requests.get(url, headers=headers)
tables = pd.read_html(response.text)

msa_pop = tables[1]

msa_pop = msa_pop[msa_pop.columns[:2]]
msa_pop.columns = ["Name", "Population_2024"]
msa_pop['Name'] = msa_pop.Name.str[:-4]
msa_pop['Name'] = msa_pop.Name.str.replace('–','-')

msa_pop = msa_pop.drop(index = 34)
nash_df = pd.DataFrame({
    "Name" : ["Nashville-Davidson--Murfreesboro--Franklin, TN"],
    "Population_2024": [2150553]
})

msa_pop = pd.concat([msa_pop, nash_df])
msa_pop = msa_pop.sort_values(by = "Population_2024", ascending = False)

#print(msa_pop.head())

######## MSA NAMES ########
top_100_msa = list(msa_pop.Name)
######## MSA NAMES ########


# MSA GEOMETRY DATA
msa_gdf = core_based_statistical_areas(cache = True, year = '2024')
msa_gdf = msa_gdf[
    msa_gdf.NAMELSAD.str.endswith("Metro Area")
]

msa_gdf['INTPTLAT'] = msa_gdf['INTPTLAT'].str[1:].astype(float)
msa_gdf["INTPTLON"] = msa_gdf["INTPTLON"].astype(float)
msa_gdf = msa_gdf[["NAME", "NAMELSAD", "INTPTLAT", "INTPTLON", "GEOID"]] # Get relevant columns
#msa_gdf["NAME"] = msa_gdf["NAME"].str.replace('--', '-')

# Create MSA df with POP and GEOID and GEOGRAPHY!!!!
msa_gdf_pop = msa_gdf.merge(msa_pop, left_on = "NAME", right_on = "Name")

##### GEOIDS #####
GEOIDS = msa_gdf_pop.GEOID
##### GEOIDS #####

name_to_geoid = dict(zip(msa_gdf_pop.Name, msa_gdf_pop.GEOID))

# print(msa_gdf_pop)
# print()

# # ------------------------------------------------------------------------------------------------------------------------------
# -- MEDIAN AGE
median_age_all = scrape_data(metro_str, ["NAME", "B01002_001E"],
           ["Name", "Median_Age", "GEOID"])

median_age_all['Name'] = median_age_all.Name.str[:-11]
median_age_all["Median_Age"] = median_age_all.Median_Age.astype(float)

median_age = median_age_all[median_age_all.Name.isin(top_100_msa)] # In the top 100
median_age = median_age.merge(msa_pop, on = 'Name')
median_age = median_age.merge(msa_gdf, on = "GEOID")


# -- Median Income
income = scrape_data(metro_str, ["NAME", "B06011_001E"], 
                     ["Name", "Median_Income", "GEOID"])

income['Name'] = income.Name.str[:-11]
income['Median_Income'] = income.Median_Income.astype(float)
income = income[income.Name.isin(top_100_msa)]

# Add
all_map_data = median_age.merge(income, on = ["GEOID", "Name"])



# -- Gender Dist
gender_pop = scrape_data(metro_str, ["NAME", "B01001_002E", "B01001_026E"], 
                     ["Name", "Male_Population", "Female_Population", "GEOID"])
gender_pop["Male_Population"] = gender_pop["Male_Population"].astype(int)
gender_pop["Female_Population"] = gender_pop["Female_Population"].astype(int)
 
gender_pop['Percent_Male'] =(100* gender_pop["Male_Population"] / (gender_pop["Male_Population"] + gender_pop["Female_Population"])).round(2)
gender_pop['Percent_Female'] =(100* gender_pop["Female_Population"] / (gender_pop["Male_Population"] + gender_pop["Female_Population"])).round(2)

gender_pop = gender_pop.drop(columns = ["Name", "Male_Population", "Female_Population"])
# Add
all_map_data = all_map_data.merge(gender_pop, on = 'GEOID')


# -- Unemp -- ONLY 92
file_path = os.path.join(os.path.dirname(__file__), 'unemp_fred_ids.csv')
fred_unemp_series = pd.read_csv(file_path)[["Series ID", "Region Name", "Region Code"]]
fred_unemp_series["GEOID"] = fred_unemp_series["Region Code"].astype(str)
fred_unemp_series = fred_unemp_series.drop(columns = ["Region Code"])

temp = msa_gdf_pop.merge(fred_unemp_series, on = 'GEOID')
temp = temp.drop_duplicates(subset = "GEOID")
# print(temp); print(); print(temp.shape); # print(); # print(fred_unemp_series)
# print(temp.head())
# print(temp.columns)
# print(temp.shape)
# print()
# Define cache path
cache_file = Path("cached_msa_unemployment.csv")
cache_file_all_unemp = Path("cache_all_unemployment.csv")

unemployment_df = None 
unemployment_historic = None

# os.remove("cached_msa_unemployment.csv")
# os.remove("cache_all_unemployment.csv")

cache_fresh = (
    cache_file.exists() and
    cache_file_all_unemp.exists() and
    (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)) < timedelta(days=7) and
    (datetime.now() - datetime.fromtimestamp(cache_file_all_unemp.stat().st_mtime)) < timedelta(days=7)
)

# Check if cache exists and is fresh (under 7 day old)
if cache_fresh:
    print("Reading Unemployment from cache")
    unemployment_df = pd.read_csv(cache_file)
    unemployment_historic = pd.read_csv(cache_file_all_unemp)

else:
    print("Cache not found or stale. Pulling fresh unemployment data...\n")
    # Get latest unemployment rate for each Series ID
    # unemployment_data = {
    #     "Series ID": [],
    #     "Unemployment_Rate": []
    # }

    # temp["Unemployment_Rate"] = temp['Series ID'].apply(lambda row: fred.get_series(row).dropna()[-1])

    temp_df = pd.DataFrame()
    unemp_rates = []

    for _, row in temp.iterrows():
        sid = row["Series ID"]
        name = row["Name"]
        geoid = row["GEOID"]

        try:
            series = fred.get_series(sid).dropna()

            # === Current Rate (mean of last 2 months)
            unemp_rates.append(np.round(series[-2:].mean(), 2))

            # === Historical
            series_df = series.reset_index()
            series_df.columns = ["Date", "Unemployment_Rate"]
            series_df["Name"] = name
            series_df["GEOID"] = geoid

            temp_df = pd.concat([temp_df, series_df], ignore_index=True)

        except Exception:
            unemp_rates.append(None)

    # Assign safely (length matches temp)
    temp["Unemployment_Rate"] = unemp_rates

    # Cache results
    unemployment_df = temp.copy()
    unemployment_df.to_csv(cache_file, index=False)

    unemployment_historic = temp_df.copy()
    unemployment_historic.to_csv(cache_file_all_unemp, index=False)


#Merge unemployment rate with MSA geometry/population
#fred_merged = msa_gdf_pop.merge(fred_unemp_series, on="GEOID").merge(unemployment_df, on="Series ID")

# print(unemployment_historic); print(); print(unemployment_historic.shape); print(); print(unemployment_historic.isna().sum())

# print(unemployment_df.columns)
# print(unemployment_df.shape)
# print(unemployment_df.isna().sum())
# print()
# print(fred_merged.isna().sum())
#-- Only 92

#-- Non Farm Jobs -- Only 94!!


file_path_nj = os.path.join(os.path.dirname(__file__), 'total_nonfarm_ids.xls')
non_farm_ids = pd.read_csv(file_path_nj)
non_farm_ids["GEOID"] = non_farm_ids["GEOID"].astype(str)

non_farm_ids.loc[non_farm_ids.Name.str.contains('Akron'), 'series_id'] = "AKRO439NA" # Fix Scraping Error
non_farm_ids.loc[non_farm_ids.Name.str.contains('Jackson, MS'), 'series_id'] = "JACK128NA" # Fix Scraping Error
non_farm_ids.loc[non_farm_ids.Name.str.contains("Rochester, NY"), 'series_id'] = "ROCH336NA" # Fix Scraping Error
non_farm_ids.loc[non_farm_ids.Name.str.contains("Austin-Round Rock-Georgetown, TX"), 'series_id'] = "AUSSA175MFRBDAL"
non_farm_ids.loc[non_farm_ids.Name.str.contains("Phil"), 'series_id'] = "PHIL942NA"
non_farm_ids.loc[non_farm_ids.Name.str.contains("Baltimore"), 'series_id'] = "BALT524NA"
#non_farm_ids.loc[non_farm_ids.Name.str.contains("Pough"), 'series_id'] = "SMS36288800000000001"

pough_temp_data = pd.DataFrame({
    "series_id": ["SMS36288800000000001"],
    "Name": ["Kiryas Joel–Poughkeepsie–Newburgh, NY"],
    "GEOID": "28880"
})

# non_farm_ids = pd.concat([non_farm_ids, pough_temp_data])

# print(non_farm_ids[non_farm_ids.Name.str.contains("Pough")])
# print(non_farm_ids.shape)

# non_farm_ids = non_farm_ids.merge(fred_unemp_series[["Region Name", "GEOID"]], 
#                        left_on = "Name", right_on = "Region Name")

non_farm_ids = non_farm_ids[["series_id", "GEOID"]].merge(msa_gdf_pop, on = "GEOID")
# print(non_farm_ids.head())
# print(non_farm_ids.columns)
# print(non_farm_ids.dtypes)

temp_non_farm = non_farm_ids.copy()
# os.remove("cached_nonfarm_workers_msa.csv")
# os.remove("cached_nonfarm_workers_msa_all.csv")

# Get Nonfarm workers
cached_file = Path("cached_nonfarm_workers_msa.csv")
cached_file_all = Path("cached_nonfarm_workers_msa_all.csv")

nonfarm_df = None
nonfarm_df_all = None



# Load from cache if under 7 days old
if cached_file.exists() and (datetime.now() - datetime.fromtimestamp(cached_file.stat().st_mtime)) < timedelta(days=7):
    print("Reading nonfarm from cache")
    nonfarm_df = pd.read_csv(cached_file)

if cached_file_all.exists() and (datetime.now() - datetime.fromtimestamp(cached_file_all.stat().st_mtime)) < timedelta(days=7):
    print("Reading historical nonfarm from cache")
    nonfarm_df_all = pd.read_csv(cached_file_all)

# If any is missing, compute both fresh
if nonfarm_df is None or nonfarm_df_all is None:
    print("Cache not found or stale. Pulling fresh nonfarm data...\n")

    workers = []
    time_series_non_farm = []

    for i, row in temp_non_farm.iterrows():
        try:
            sid = row["series_id"]
            series_data = fred.get_series(sid).dropna()
            workers.append(np.round(series_data[-2:].mean()))

            df_temp = series_data.to_frame("Non_Farm_Workers").reset_index()
            df_temp["Non_Farm_Workers"] *= 1000
            df_temp["Name"] = row["Name"]
            df_temp["GEOID"] = row["GEOID"]
            time_series_non_farm.append(df_temp)

        except Exception as e:
            workers.append(None)
            print(f"Failed on series {sid}: {e}")

    temp_non_farm["NonFarm_Workers"] = workers
    nonfarm_df = temp_non_farm
    nonfarm_df_all = pd.concat(time_series_non_farm, ignore_index=True)


    # Save to cache
    nonfarm_df.to_csv(cached_file, index=False)
    nonfarm_df_all.to_csv(cached_file_all, index=False)

#print()
nonfarm_df["NonFarm_Workers"] = np.round(nonfarm_df["NonFarm_Workers"]* 1000)
nonfarm_df["Percent_Working_Nonfarm_Jobs"] = np.round(100* nonfarm_df["NonFarm_Workers"] / nonfarm_df["Population_2024"], 2)



# print(nonfarm_df_all[nonfarm_df_all.Name.str.contains("Pough")])
# nonfarm_df = nonfarm_df.drop(2) # Drop Austin-Round Rock, TX, Data Corrupted

# print(nonfarm_df.columns)
# nonfarm_df = nonfarm_df.merge(msa_gdf_pop, on = "GEOID")

# print(len(nonfarm_df))
# #print(set(GEOIDS) - set(nonfarm_df.GEOID))
# print(nonfarm_df.GEOID.nunique())


# -- Immigrant
imm = scrape_data(metro_str, ["NAME", 'B05015_001E', "B01003_001E"], ["Name", "Immigrant", "Total", "GEOID"])

imm["Immigrant"] = imm["Immigrant"].astype(int)
imm["Total"] = imm["Total"].astype(int)

imm["Percent_Immigrant_Population"] = np.round(100* imm["Immigrant"]/imm["Total"], 2)
imm["Name"] = imm.Name.str[:-11]
imm = imm[imm.Name.isin(top_100_msa)].\
      sort_values(by = "Percent_Immigrant_Population", ascending = False)

# -- Addition Specific Immigration Info
immigrant_detailed = pd.read_html("https://api.census.gov/data/2023/acs/acs5/groups/B05006.html")[0][["Name", "Label", "Concept"]]
immigrant_detailed = immigrant_detailed[
    (immigrant_detailed.Label.str.contains("Estimate!!Total"))
    &
    (immigrant_detailed.Label.str.startswith("Estimate"))
]

immigrant_detailed['split_label'] = immigrant_detailed.Label.str.split("!!")

continents = immigrant_detailed[
    (immigrant_detailed.split_label.str.len() ==3)
]


names = continents.split_label.str[-1].tolist()
continent_names = [name[:-1] for name in names] # Continent Names Options

imm_continets = scrape_data(
    metro_str,
    ["NAME", "B05015_001E"] + list(continents.Name), 
    ["Name", "Total_Immigrant_Population"] + continent_names + ["GEOID"]
)
imm_continets["Name"] = imm_continets["Name"].str[:-11]
imm_continets = imm_continets[imm_continets.Name.isin(top_100_msa)]
imm_continets[["Total_Immigrant_Population"] + continent_names] = imm_continets[["Total_Immigrant_Population"] + continent_names].astype(int)

imm_continets = imm_continets[["Total_Immigrant_Population"] + continent_names + ["GEOID"]]
for name in continent_names:
    imm_continets[f"Percent_of_Immigrants_from_{name}"] =np.round(100* imm_continets[name]/imm_continets["Total_Immigrant_Population"], 2)
    imm_continets = imm_continets.drop(columns = name)

imm = imm.merge(imm_continets, on = "GEOID") # Add
imm = imm.merge(
    msa_gdf_pop[["INTPTLAT", "INTPTLON", "Population_2024", "GEOID"]], 
    on = "GEOID"
) # Add Map Data

#print(imm)

# -- Addition Specific Immigration Info


#all_map_data = all_map_data.merge(imm[['Percent_Immigrant_Population',"GEOID"]], on = "GEOID")

#-- Race
# Scrape Data
race_vars = [
  "NAME",              # area name
  "B01003_001E",       # Total
  "B02001_002E",       # White alone
  "B02001_003E",       # Black alone
  "B02001_004E",       # AIAN alone
  "B02001_005E",       # Asian alone
  "B02001_006E",       # NH-PI alone
  "B02001_007E"       # Some other
  ]


race_labels = [
  "Name",
  "Total",
  "White",
  "Black_or_African_American",
  "American_Indian_and_Alaska_Native",
  "Asian",
  "Native_Hawaiian_Other_Pacific_Islander",
  "Some_Other_Race"
]


race_df_whole = scrape_data("metropolitan statistical area/micropolitan statistical area", race_vars, race_labels + ["GEOID"])
race_df_whole["Name"] = race_df_whole["Name"].str[:-11]

# race_df = race_df_whole.copy(deep=True)

# Convert dtype
for race in race_labels[1:]:
    race_df_whole[race] = race_df_whole[race].astype(int)

# print(race_df_whole.head())


race_df = race_df_whole.copy(deep=True)
for race in race_labels[2:]:
    race_df[race] = np.round(100* race_df[race] / race_df['Total'], 2)
race_df = race_df.drop(columns = ["Total"])

race_df.columns = ["Name", "Percent_White", "Percent_Black_or_African_American",
                  "Percent_American_Indian", "Percent_Asian", "Percent_Hawaiian", "Percent_Other", "GEOID"]

race_df = race_df[race_df.Name.isin(top_100_msa)]
race_df = race_df.merge(all_map_data[["INTPTLAT", "INTPTLON", "GEOID"]], on = "GEOID")
race_df = race_df.merge(msa_pop, on ='Name')

#print(race_df)
#print(race_df.columns)
#print(race_df.Percent_Other)
#--Race

# -- Hispanic
race_vars = [
    "NAME",                  # MSA name
    "B01003_001E",           # Total
    "B01001I_001E"           # Hispanics Ethnicity ≠ Race
]

race_labels = [
    "Name",
    "Total",
    "Hispanic_or_Latino_Total"
]

hisp = scrape_data(metro_str, race_vars, race_labels+["GEOID"])
hisp["Name"] = hisp["Name"].str[:-11]
hisp[["Total", "Hispanic_or_Latino_Total"]] = hisp[["Total", "Hispanic_or_Latino_Total"]].astype(int)
hisp = hisp[hisp.Name.isin(top_100_msa)]
hisp["Non_Hispanic_or_Latino_Total"] = hisp["Total"] - hisp["Hispanic_or_Latino_Total"]
hisp["Percentage_Hispanic_or_Latino"] = np.round(100* hisp.Hispanic_or_Latino_Total/hisp.Total, 2)
hisp = hisp.merge(msa_gdf_pop[["INTPTLAT", "INTPTLON", "GEOID", "Population_2024"]], on ="GEOID") # == Add ==
# -- Hispanic

# -- Education
ed = scrape_data(metro_str, ["NAME", "B23006_001E", "B23006_002E", "B23006_009E", "B23006_016E", "B23006_023E"], 
                 ["Name", "Total_25-64", "Less_Than_High_School_Diplomna", "High_School_Diploma", "Some_College_or_Associates_Degree",
                  "Bachelor_Degree_or_Higher", "GEOID"])

ed["Name"] = ed["Name"].str[:-11]
num_ed_cols = ["Total_25-64", "Less_Than_High_School_Diplomna", "High_School_Diploma", "Some_College_or_Associates_Degree","Bachelor_Degree_or_Higher"]
ed[num_ed_cols] = ed[num_ed_cols].astype(int)
ed = ed[ed.Name.isin(top_100_msa)]

for col in num_ed_cols[1:]:
    ed[f"Percent_25-64_Year_Olds_With_{col}"] = np.round(100* ed[col] / ed['Total_25-64'], 2)
    ed = ed.drop(columns = col)

ed = ed.drop(columns = ["Name"]).merge(msa_gdf_pop, on ="GEOID")
# print(f"Education\n{ed.columns}")
# -- Education


# ---- Poverty Rate
poverty = scrape_data(metro_str, ["NAME", "B01003_001E", "B17020_002E"], ["Name", "Total", "Poverty", "GEOID"])
poverty[["Total", "Poverty"]] = poverty[["Total", "Poverty"]].astype(int)
poverty["Name"] = poverty["Name"].str[:-11]
poverty["Poverty_Rate"] = np.round(100* (poverty["Poverty"] / poverty["Total"]), 2)

poverty = poverty[poverty.GEOID.isin(GEOIDS)]
poverty_mapping = poverty[["GEOID", "Poverty_Rate"]].merge(msa_gdf_pop, on = "GEOID")
# ---------------------------------------------------------------------  SECTION 1 ⬆️--------------------------------------------------------------------- #


# ------------------------------------------------------------------ Demographics Data ------------------------------------------------------------------
# 1. Gender Pie Chart Orginization
gender_pie = all_map_data[["Name", "Percent_Male", "Percent_Female", "GEOID"]]
#print(gender_pie.head())

# 2. Race
race_pie = race_df_whole[race_df_whole.Name.isin(top_100_msa)]
#print(race_pie.head())

# 2. Hispanics
hisp_pie = hisp[["Name", "Hispanic_or_Latino_Total", "Non_Hispanic_or_Latino_Total"]]



# ======================================================= Time Series =======================================================
# Setup
# === CONFIG ===
years = np.arange(2010, 2024)
HOST = "https://api.census.gov/data"
dataset = "acs/acs5"
pop_id = "B01003_001E"       # Total population
inc_id = "B06011_001E"       # Median income

cache_dir = "census_cache"
os.makedirs(cache_dir, exist_ok=True)


# Corrections for 2023 GEOID anomalies
geoid_corrections_2023 = {
    "17460": "17410",  # Cleveland
    "39100": "28880",  # Poughkeepsie
}

name_corrections_2023 = {
    "Cleveland-Elyria, OH Metro Area": "Cleveland, OH",
    "Poughkeepsie-Newburgh-Middletown, NY Metro Area": "Poughkeepsie, NY"
}

# Combined GEOID list with alternates
extended_geoids = list(set(GEOIDS).union(geoid_corrections_2023.keys()))

# === OUTPUT ===
pop_time_series_data = pd.DataFrame()
inc_time_series_data = pd.DataFrame()
imm_time_series_data = pd.DataFrame()
poverty_rate_time_series_data = pd.DataFrame()

for year in years:
    temp_year = str(year)
    base_url = '/'.join([HOST, temp_year, dataset])

    # === POPULATION ===
    pop_cache_path = os.path.join(cache_dir, f"pop_{temp_year}.json")
    if os.path.exists(pop_cache_path):
        with open(pop_cache_path, 'r') as f:
            pop_data = json.load(f)
    else:
        pop_params = {
            'get': f"NAME,{pop_id}",
            'for': f"{metro_str}:*",
            'key': census_key
        }
        r = requests.get(base_url, params=pop_params)
        r.raise_for_status()
        pop_data = r.json()
        with open(pop_cache_path, 'w') as f:
            json.dump(pop_data, f)

    pop_temp = pd.DataFrame(pop_data[1:], columns=["Name", "Population", "GEOID"])
    pop_temp = pop_temp[pop_temp.GEOID.isin(extended_geoids)]
    pop_temp["GEOID"] = pop_temp["GEOID"].replace(geoid_corrections_2023)
    if year == 2023:
        pop_temp["Name"] = pop_temp["Name"].replace(name_corrections_2023)
    pop_temp["Name"] = pop_temp["Name"].str[:-11]
    pop_temp["Population"] = pop_temp["Population"].astype(int)
    pop_temp["Year"] = year
    pop_time_series_data = pd.concat([pop_time_series_data, pop_temp], ignore_index=True)

    # === INCOME ===
    inc_cache_path = os.path.join(cache_dir, f"income_{temp_year}.json")
    if os.path.exists(inc_cache_path):
        with open(inc_cache_path, 'r') as f:
            inc_data = json.load(f)
    else:
        inc_params = {
            'get': f"NAME,{inc_id}",
            'for': f"{metro_str}:*",
            'key': census_key
        }
        r = requests.get(base_url, params=inc_params)
        r.raise_for_status()
        inc_data = r.json()
        with open(inc_cache_path, 'w') as f:
            json.dump(inc_data, f)

    inc_temp = pd.DataFrame(inc_data[1:], columns=["Name", "Median_Income", "GEOID"])
    inc_temp = inc_temp[inc_temp.GEOID.isin(extended_geoids)]
    inc_temp["GEOID"] = inc_temp["GEOID"].replace(geoid_corrections_2023)
    if year == 2023:
        inc_temp["Name"] = inc_temp["Name"].replace(name_corrections_2023)
    inc_temp["Name"] = inc_temp["Name"].str[:-11]
    inc_temp["Median_Income"] = inc_temp["Median_Income"].astype(int)
    inc_temp["Year"] = year
    inc_time_series_data = pd.concat([inc_time_series_data, inc_temp], ignore_index=True)

    # === IMMIGRATION ===
    imm_id = 'B05006_001E' if year <= 2018 else 'B05015_001E'
    imm_cache_path = os.path.join(cache_dir, f"immigration_{temp_year}.json")
    if os.path.exists(imm_cache_path):
        with open(imm_cache_path, 'r') as f:
            imm_data = json.load(f)
    else:
        imm_params = {
            'get': f"NAME,{imm_id}",
            'for': f"{metro_str}:*",
            'key': census_key
        }
        r = requests.get(base_url, params=imm_params)
        r.raise_for_status()
        imm_data = r.json()
        with open(imm_cache_path, 'w') as f:
            json.dump(imm_data, f)

    imm_temp = pd.DataFrame(imm_data[1:], columns=["Name", "Total_Foreign_Born_Population", "GEOID"])
    imm_temp = imm_temp[imm_temp.GEOID.isin(extended_geoids)]
    imm_temp["GEOID"] = imm_temp["GEOID"].replace(geoid_corrections_2023)
    if year == 2023:
        imm_temp["Name"] = imm_temp["Name"].replace(name_corrections_2023)
    imm_temp["Name"] = imm_temp["Name"].str[:-11]
    imm_temp["Total_Foreign_Born_Population"] = imm_temp["Total_Foreign_Born_Population"].astype(int)
    imm_temp["Year"] = year
    imm_time_series_data = pd.concat([imm_time_series_data, imm_temp], ignore_index=True)


    # === POVERTY RATE ===
    if year in [2010, 2011, 2012]:
        poverty_ids = ["B01003_001E", "B14006_002E"]
    else:
        poverty_ids = ["B01003_001E", "B17020_002E"]
    
    poverty_cache_path = os.path.join(cache_dir, f"poverty_{temp_year}.json")

    if os.path.exists(poverty_cache_path):
        with open(poverty_cache_path, 'r') as f:
            poverty_data = json.load(f)
    else:
        poverty_params = {
            'get': f"NAME,{','.join(poverty_ids)}",
            'for': f"{metro_str}:*",
            'key': census_key
        }
        r = requests.get(base_url, params=poverty_params)
        r.raise_for_status()
        poverty_data = r.json()
        with open(poverty_cache_path, 'w') as f:
            json.dump(poverty_data, f)

    poverty_temp = pd.DataFrame(poverty_data[1:], columns=["Name", "Total", "Poverty", "GEOID"])
    poverty_temp = poverty_temp[poverty_temp.GEOID.isin(extended_geoids)]
    poverty_temp["GEOID"] = poverty_temp["GEOID"].replace(geoid_corrections_2023)
    if year == 2023:
        poverty_temp["Name"] = poverty_temp["Name"].replace(name_corrections_2023)
    poverty_temp["Name"] = poverty_temp["Name"].str[:-11]
    poverty_temp[["Total", "Poverty"]] = poverty_temp[["Total", "Poverty"]].astype(int)
    poverty_temp["Poverty_Rate"] = np.round(100 * poverty_temp["Poverty"] / poverty_temp["Total"], 2)
    poverty_temp["Year"] = year

    poverty_rate_time_series_data = pd.concat([poverty_rate_time_series_data, poverty_temp], ignore_index=True)



# === Append 2024 manually for population if available ===
if 'msa_gdf_pop' in locals():
    temp_2024 = msa_gdf_pop[["Name", "Population_2024", "GEOID"]].copy()
    temp_2024["Year"] = 2024
    temp_2024.columns = pop_time_series_data.columns
    pop_time_series_data = pd.concat([pop_time_series_data, temp_2024], ignore_index=True)

# === Final sorting ===
pop_time_series_data = pop_time_series_data.sort_values(by="Year").reset_index(drop=True)
inc_time_series_data = inc_time_series_data.sort_values(by="Year").reset_index(drop=True)
imm_time_series_data = imm_time_series_data.sort_values(by="Year").reset_index(drop=True)
poverty_rate_time_series_data = poverty_rate_time_series_data.sort_values(by="Year").reset_index(drop=True)

# print(f"POVERTY DF\n{poverty_rate_time_series_data}")


# print(imm_time_series_data.head());print()
# print(pop_time_series_data.GEOID.value_counts())
# import shutil
# shutil.rmtree("census_cache")
# ======================================================= Time Series: Population & Income =======================================================

# ===================================================================== Age =====================================================================
cols_names = ["5", '5-9', '10-14', '15-17', '18-19', '20', '21', '22-24', '25-29', '30-34', '35-39', '40-44', 
              '45-49', '50-54', '55-59', '60-61', '62-64', '65-66', '67-69', '70-74', '75-79', '80-84', '85']

get_vars = ["NAME", "B01003_001E"]
main_cols = ["Name", "Population"]

# == Males ==
male_age_vars = ["B01001_" + str(i).zfill(3) + "E" for i in range(3, 26)]
male_age_col_names = [
    "males_under_" + t if t == "5" else
    "males_over_" + t if t == "85" else
    t + "_years_males"
    for t in cols_names
]

male_age_data = scrape_data(
    metro_str,
    get_vars + male_age_vars,
    main_cols + male_age_col_names + ["GEOID"]
)
male_age_data[male_age_col_names] = male_age_data[male_age_col_names].astype(int)

male_age_data = change_bins(male_age_data, "males")
male_age_data["Name"] = male_age_data["Name"].str[:-11]
male_age_data = male_age_data[male_age_data.Name.isin(top_100_msa)]

# == Female ==
female_age_vars = ["B01001_" + str(i).zfill(3) + "E" for i in range(27, 50)] 
female_age_col_names = [
    "females_under_" + t if t == "5" else
    "females_over_" + t if t == "85" else
    t + "_years_females"
    for t in cols_names
]

female_age_data = scrape_data(
    metro_str,
    get_vars + female_age_vars,
    main_cols + female_age_col_names + ["GEOID"]
)
female_age_data[female_age_col_names] = female_age_data[female_age_col_names].astype(int)
female_age_data = change_bins(female_age_data, "females")
female_age_data["Name"] = female_age_data["Name"].str[:-11]
female_age_data = female_age_data[female_age_data.Name.isin(top_100_msa)]


# ============ Combine Ages ============
df_combined_age = pd.merge(male_age_data, female_age_data, on=["Name", "Population", "GEOID"])

# Combine male and female age groups
age_groups = [
    "under_9",
    "10-17_years",
    "18-24_years",
    "25-34_years",
    "35-44_years",
    "45-54_years",
    "55-66_years",
    "67-79_years",
    "over_80"
]

# Create new columns with total population per age group
for age in age_groups:
    if age == "under_9" or age == "over_80":
        df_combined_age[f"Total_{age}"] = df_combined_age[f"males_{age}"] + df_combined_age[f"females_{age}"]
    else:
        df_combined_age[f"Total_{age}"] = df_combined_age[f"{age}_males"] + df_combined_age[f"{age}_females"]


df_combined_age = df_combined_age[["Name", "GEOID", "Population"] + [f"Total_{age}" for age in age_groups]]

# Select columns excluding 'GEOID' and population
cols_to_use = [col for col in male_age_data.columns if col not in ["GEOID", "Population"]]
# Perform melt using those columns
# Select columns excluding 'GEOID'
cols_to_use = [col for col in male_age_data.columns if col not in ["GEOID", "Population"]]
# Perform melt using those columns
male_age_data = pd.melt(
    male_age_data[cols_to_use],
    id_vars=["Name"],
    var_name = "Age_Range",
    value_name = "Population"
)

cols_to_use = [col for col in female_age_data.columns if col not in ["GEOID", "Population"]]
female_age_data = pd.melt(
    female_age_data[cols_to_use],
    id_vars=["Name"],
    var_name = "Age_Range",
    value_name = "Population"
)

cols_to_use = [col for col in df_combined_age.columns if col not in ["GEOID", "Population"]]
df_combined_age = pd.melt(
    df_combined_age[cols_to_use],
    id_vars=["Name"],
    var_name = "Age_Range",
    value_name = "Population"
)
# print(male_age_data.head()); print()
# print(female_age_data.head());print()
# print(df_combined_age.head())
# ===================================================================== Age =====================================================================


# ==== Immigration
# 1. Total immigration
imm_total = imm[["Name", "Total_Immigrant_Population"]]



# =========== Foreign Born Pop
countries = immigrant_detailed[
    (immigrant_detailed.split_label.str.len() >= 5)
]
countries = countries[~countries['split_label'].apply(lambda x: any('n.e.c.' in item for item in x))]

country_names = countries.split_label.str[-1].tolist()
countries["Country_Name"] = [country.replace(':', '').strip() for country in country_names]
countries = countries[~countries.Country_Name.str.contains("Other")]

countries = countries.drop([28, 192, 556, 612, 652])

europe = countries[countries.Label.str.contains("Europe")]
asia = countries[countries.Label.str.contains("Asia")]
africa = countries[countries.Label.str.contains("Africa")]
americas = countries[countries.Label.str.contains("Americas")]


# Manually Get Oceainia ID's
oceiania = immigrant_detailed[immigrant_detailed.Label.str.contains("Oceania")]
oceiania = oceiania[~oceiania.Label.str.contains("Other")]
oceiania = oceiania.drop([544, 516])

oceiania["Country_Name"] = oceiania.split_label.str[-1]
oceiania["Country_Name"] = oceiania["Country_Name"].str.replace(':', '').str.strip()


europe_df = load_or_process_region_df("europe", europe, metro_str, top_100_msa, scrape_data)
asia_df = load_or_process_region_df("asia", asia, metro_str, top_100_msa, scrape_data)
africa_df = load_or_process_region_df("africa", africa, metro_str, top_100_msa, scrape_data)
americas_df = load_or_process_region_df("americas", americas, metro_str, top_100_msa, scrape_data)
oceania_df = load_or_process_region_df("oceiania", oceiania, metro_str, top_100_msa, scrape_data)

#oceiania = load_or_process_region_df("oceiania", oceiania, metro_str, top_100_msa, scrape_data)
# print(europe_df.head())
# print(asia_df.head())
# print(africa_df.head())
# print(oceania_df.head())
# print(americas_df.head())


# cache_dir = "cached_regions"
# regions = ["oceiania"]

# Delete cached files
# for region in regions:
#     cache_path = os.path.join(cache_dir, f"{region}_immigrants.csv")
#     if os.path.exists(cache_path):
#         os.remove(cache_path)
#         print(f"Deleted cache for {region}")
# =========== Foreign Born Pop


# ------------------------------------------------------------------ Demographics Data ------------------------------------------------------------------





# ---------------------------------------------------------------------  SECTION 2 ⬇️ --------------------------------------------------------------------- #
map_options = {
    "Population": "Population_2024",
    "Population by Gender":"Percent_Female",
    "Hispanic Population(Ethnicity)":"Percentage_Hispanic_or_Latino",
    "Race": "Percent_White",
    "Foreign Born Population": "Percent_Immigrant_Population",
    "Median Age": "Median_Age",
    "Education": "Less_Than_High_School",
    "Median Income": "Median_Income",
    "Poverty Rate": "Poverty_Rate",
    "Unemployment Rate": "Unemployment_Rate",
    "Total Nonfarm Workers": "NonFarm_Workers"
}
# ---------------------------------------------------------------------  SECTION 2 ⬆️--------------------------------------------------------------------- #




# -------------------------------------------------------------------- Econ Data --------------------------------------------------------------------
# ======================== Pughkeepsie Unemployment
# os.remove("poughkeepsie_manual_unemployment.csv")
cache_file = "poughkeepsie_manual_unemployment.csv"
poughkeepsie_lf_id = "B23025_003E"
poughkeepsie_ur_id = "B23025_005E"

if os.path.exists(cache_file):
    pough_ur = pd.read_csv(cache_file)
else:
    pough_ur = pd.DataFrame()

    for year in np.arange(2019, 2024):
        temp_year = str(year)
        base_url = '/'.join([HOST, temp_year, dataset])

        pop_params = {
            'get': f"NAME,{poughkeepsie_lf_id},{poughkeepsie_ur_id}",
            'for': f"{metro_str}:*",
            'key': census_key
        }

        r = requests.get(base_url, params=pop_params)
        r.raise_for_status()
        data = r.json()

        temp_df = pd.DataFrame(data[1:], columns=["Name", "LF", "Unemployed", "GEOID"])
        temp_df[["LF", "Unemployed"]] = temp_df[["LF", "Unemployed"]].dropna().astype(int)

        temp_df = temp_df[(temp_df["Name"].str.contains("Pough")) | (temp_df["GEOID"] == '39100')]
        temp_df["Year"] = year
        temp_df["Unemployment_Rate"] = np.round(100 * (temp_df.Unemployed / temp_df.LF), 2)

        pough_ur = pd.concat([pough_ur, temp_df])

    pough_ur["Name"] = "Kiryas Joel-Poughkeepsie-Newburgh, NY Metro Area"
    pough_ur["GEOID"] = "28880"

    pough_ur[["Name", "GEOID", "Unemployment_Rate", "Year"]].to_csv(cache_file, index=False)

# pough_ur is now ready to use
# print(pough_ur.head())



# ================================================ Income Distribution
income_brackets = [
    "<$10k", "$10k-$14.9k", "$15k-$19.9k", "$20k-$24.9k", "$25k-$29.9k",
    "$30k-$34.9k", "$35k-$39.9k", "$40k-$44.9k", "$45k-$49.9k",
    "$50k-$59.9k", "$60k-$74.9k", "$75k-$99.9k", "$100k-$124.9k",
    "$125k-$149.9k", "$150k-$199.9k", ">$200k"
]

table_map = {
    "B19001": "All",
    "B19001A": "White",
    "B19001B": "Black",
    "B19001C": "American Indian",
    "B19001D": "Asian",
    "B19001E": "Hawaiian",
    "B19001G": "Two or More Races",
    "B19001I": "Hispanic or Latino"
}

# Caching file path
cache_file = "income_distribution_by_race.csv"

# os.remove("income_distribution_by_race.csv")

if os.path.exists(cache_file):
    income_distribution_by_race = pd.read_csv(cache_file)
else:
    race_dfs = []

    for table_id, race_label in table_map.items():
        df_raw = pd.read_html(f"https://api.census.gov/data/2023/acs/acs5/groups/{table_id}.html")[0]
        df_raw = df_raw[df_raw.Name.str.endswith('E')][1:-1]  # Drop total and margin of error
        vars_inc = df_raw.Name.tolist()

        df = scrape_data(metro_str, ["NAME"] + vars_inc, ["Name"] + income_brackets + ["GEOID"])

        df = df.query('GEOID in @GEOIDS')
        df[income_brackets] = df[income_brackets].astype(int)
        df["Name"] = df["Name"].str[:-11]

        df_long = pd.melt(df, id_vars=["Name", "GEOID"], value_name="Count", var_name="Income_Bracket")
        df_long["Race"] = race_label
        race_dfs.append(df_long)

    # Final combined DataFrame
    income_distribution_by_race = pd.concat(race_dfs, ignore_index=True)
    
    # Cache to CSV
    income_distribution_by_race.to_csv(cache_file, index=False)


# print(f"INC DIST\n{income_distribution_by_race}")


# =========================== Workers by Industry
raw = pd.read_html("https://api.census.gov/data/2023/acs/acs5/groups/C24070.html")[0]
vars_df = raw[
    raw.Label.str.startswith("Estimate!!Total:!!")
    & raw.Name.str.endswith("E")
]

vars_df['split_label'] = vars_df['Label'].str.split('!!')
vars_df = vars_df[vars_df.split_label.str.len() == 3]
vars_df["Industry"] = vars_df['split_label'].str[-1]
vars_df = vars_df[["Name", "Label", "Industry"]]
vars_df['Industry'] = vars_df['Industry'].str.replace(':', '').str.strip()

industry_data = scrape_data(
    metro_str,
    ["NAME"] + vars_df['Name'].tolist(),
    ["Name"] + vars_df['Industry'].tolist() + ["GEOID"]
)

industry_data = industry_data.query("GEOID in @GEOIDS")
industry_data[vars_df['Industry'].tolist()] = industry_data[vars_df['Industry'].tolist()].astype(int)
industry_data["Name"] = industry_data["Name"].str[:-11]

industry_long = pd.melt(
    industry_data,
    id_vars=["Name", "GEOID"],
    var_name="Industry",
    value_name="Workers"
)


# print(industry_long)
# -------------------------------------------------------------------- Econ Data --------------------------------------------------------------------


# ============================================================= Social =============================================================
# Households for boxes
households = pd.read_html("https://api.census.gov/data/2023/acs/acs5/groups/B11001.html")[0]
households = households[households.Name.str.endswith("E")]
households = households[:-1]
households["Title"] = households["Label"].str.split('!!').str[-1].str.replace(':', "").str.strip()

households_data = scrape_data(metro_str, ["NAME"]+households.Name.tolist(), ["Name"]+households.Title.tolist()+["GEOID"])
households_data[households.Title.tolist()] = households_data[households.Title.tolist()].astype(int)
households_data["Name"] = households_data["Name"].str[:-11]
households_data = households_data.query('GEOID in @GEOIDS')

# Household sizes
household_size = pd.read_html("https://api.census.gov/data/2023/acs/acs5/groups/B11016.html")[0]
household_size = household_size[household_size.Name.str.endswith("E")][2:-1]
household_size = household_size.drop(32)
household_size["Type"] =  ["Family Household" if "Family households" in lab else "Nonfamily Households" 
                           for lab in household_size.Label.tolist()]

household_size = household_size[["Name", "Label", "Type"]]

household_size["Size"] = household_size.Label.str.split("!!").str[-1].str.replace(':', '').str.strip()
household_size["Label_Name"] = household_size["Type"] + " - " + household_size["Size"]

fam_size = scrape_data(metro_str, ["NAME"] + household_size.Name.tolist(), ["Name"] + household_size.Label_Name.tolist() + ["GEOID"])
fam_size["Name"] = fam_size["Name"].str[:-11] 
fam_size = fam_size.query("GEOID in @GEOIDS")
fam_size[household_size.Label_Name] = fam_size[household_size.Label_Name].astype(int)

fam_size = pd.melt(fam_size, id_vars = ["Name", "GEOID"], value_name = "Count", var_name = "Size")



# ================================================ Language
# Languages 
lang_ids = pd.read_html("https://api.census.gov/data/2023/acs/acs5/groups/B16002.html")[0]
lang_ids = lang_ids[lang_ids.Name.str.endswith("E")]
lang_ids = lang_ids[1:-1]
lang_ids["split_label"] = lang_ids["Label"].str.split("!!")
lang_ids = lang_ids[lang_ids.split_label.str.len() == 3]
lang_ids["Language"] = lang_ids["split_label"].str[-1]
lang_ids["Language"] = lang_ids["Language"].str.replace(":", "").str.strip()

language = scrape_data(metro_str, ["NAME"]+lang_ids.Name.tolist(), ["Name"]+lang_ids.Language.tolist()+["GEOID"])
language[lang_ids.Language] = language[lang_ids.Language].astype(int)
language = language.query("GEOID in @GEOIDS")
language = pd.melt(language, id_vars = ["Name", "GEOID"], var_name = "Language", value_name = "Count")
language["Name"] = language["Name"].str[:-11] 


# -------------------------------- Education
ed_all = ed[["Name", "Percent_25-64_Year_Olds_With_Less_Than_High_School_Diplomna", "Percent_25-64_Year_Olds_With_High_School_Diploma",
            "Percent_25-64_Year_Olds_With_Some_College_or_Associates_Degree", "Percent_25-64_Year_Olds_With_Bachelor_Degree_or_Higher", "GEOID"]]
ed_all = pd.melt(ed_all, id_vars = ["Name", "GEOID"], var_name = "Degree", value_name = "Percent_25-64")


# Education detailed
# -- 1. Earnings
earning_ids = pd.read_html("https://api.census.gov/data/2023/acs/acs5/groups/B20004.html")[0]
earning_ids = earning_ids[
    (earning_ids.Name.str.endswith("E"))
    &
    (~earning_ids.Label.str.contains("Male"))
    &
    (~earning_ids.Label.str.contains("Female"))
][1:-1]

earning_ids["Degree"] = earning_ids["Label"].str.split("!!").str[-1]

income_by_edu = scrape_data(metro_str, ["NAME"]+earning_ids["Name"].tolist(), ["Name"]+earning_ids.Degree.tolist()+["GEOID"])
income_by_edu["Name"] = income_by_edu["Name"].str[:-11]
income_by_edu[earning_ids.Degree.tolist()] = income_by_edu[earning_ids.Degree.tolist()].astype(int)
income_by_edu = income_by_edu.query('GEOID in @GEOIDS')
income_by_edu["Bachelor's Degree or Higher"] = np.round((income_by_edu["Bachelor's degree"] + income_by_edu["Graduate or professional degree"])/2, 2)
income_by_edu = income_by_edu.drop(columns = ["Bachelor's degree", "Graduate or professional degree"])


income_by_edu = pd.melt(income_by_edu, id_vars = ["Name", "GEOID"], var_name = "Degree", value_name = "Median_Income")

# ============================================================= Social =============================================================


# ****** -------------------------------------------------------------- DASH APP ⬇️-------------------------------------------------------------- *****
# Create Dash App
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# -**---------------------------------------------------------------- App Layout ----------------------------------------------------------------**
app.layout = html.Div(
    children=[
        dcc.Location(id='url'),

        # Navigation bar
        html.Div(
            children=[
                dcc.Link("🏠 Home", href='/', className='nav-link'),
                dcc.Link("📍 Maps", href='/Maps', className='nav-link'),
                dcc.Link("👥 Demographics", href='/Demographics', className='nav-link'),
                dcc.Link("🏛️ Economy", href='/economy', className='nav-link'),
                dcc.Link("👨‍👩‍👧‍👦 Social", href='/social', className='nav-link')
            ],
            style={
                'display': 'flex',
                'justifyContent': 'center',
                'alignItems': 'center',
                'gap': '30px',
                'backgroundColor': '#f4f4f4',
                'padding': '15px 0',
                'borderBottom': '1px solid #ccc',
            }
        ),

        html.Div(id="page-content")
    ],
    style={'width': '100%', 'maxWidth': '1400px', 'margin': '0 auto'}
)

# **-----------------------------------------------------------------  HOME PAGE -----------------------------------------------------------------**
home_page = html.Div(
    children=[
        html.Img(
            src="https://wallpapercave.com/wp/wp7846033.jpg",
            style= {
                'width': '100%',
                'maxHeight': '250px',
                'objectFit': 'cover',
                'borderRadius': '10px',
                'marginBottom': '20px'
            }
        ),
        
        html.H1("US. Metropolitan Statistical Area Analysis(2023-25)", style={'border':'3px solid black'}),
        html.H2("Explore Data and Trends Amongst America's 100 Largest Metropolitan Statistical Areas"),

        html.Div(
            children=[
                dcc.Link('🌍 View MSA Maps', href='/Maps', style={
                    'padding': '10px 20px',
                    'backgroundColor': '#2980B9',
                    'color': 'white',
                    'textDecoration': 'none',
                    'borderRadius': '5px'
                }),
                dcc.Link('🧑‍🤝‍🧑 View Demographics', href='/Demographics', style={
                    'padding': '10px 20px',
                    'backgroundColor': '#2980B9',
                    'color': 'white',
                    'textDecoration': 'none',
                    'borderRadius': '5px'
                }),
                dcc.Link('💸 View Economy', href='/economy', style={
                    'padding': '10px 20px',
                    'backgroundColor': '#2980B9',
                    'color': 'white',
                    'textDecoration': 'none',
                    'borderRadius': '5px'
                }),
                dcc.Link('📱 View Social', href='/social', style={
                    'padding': '10px 20px',
                    'backgroundColor': '#2980B9',
                    'color': 'white',
                    'textDecoration': 'none',
                    'borderRadius': '5px'
                })
            ],
            style={
                'marginTop': '30px',
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'center',
                'gap': '20px'
            }
        ),

        html.Br(),

        html.P(
            "There are 393 Metropolitan Statistical Areas (MSA) in the United States. These MSAs are defined by the U.S. Census Bureau and are based on counties or equivalent entities,"
            "with at least one urbanized area of 50,000 or more population, and adjacent counties having a high degree of social and economic integration with the urbanized area."
            " This dashboard will provide data from the U.S. Census Bureau dating back from early 2000's-2023(No more recent data from US Census), aswell as the Federal Reserve Economic Database."
            " These areas are home to the majority of American Residents as all 393 MSA's have a total of 293,884,621 residents. The top 100 consist of nearly 80% of that with a total population of 228,366,748 "
            "which is 65% of the total US Population, which is why we will only look at the top 100 MSA's.",
        
            style={
                'maxWidth': '800px',
                'margin': '0 auto',
                'fontSize': '1.2em',
                'lineHeight': '1.6',
                'color': '#555'
            }
        ),

        html.Br(),

        html.Div([
            html.Img(
                src="https://play-lh.googleusercontent.com/ibZuCLir234b1v0q1XZ9UwqpcfFIakXHdkWswCH-uxZQ5e2wXdVXZhGo2zoyehMes41x",
                style={'height': '80px', 'margin': '10px'}
            ),
            html.Img(
                src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTDy-EvDPwgFglMxSbxNRP15Ltw32mo0DI2ug&s",
                style={'height': '80px', 'margin': '10px'}
            )
        ], style={'textAlign': 'center', 'paddingBottom': '20px'})
        
    ],
    style={'text-align':'center', 'display':'inline-block', 'width':'100%'} 
)
# **------------------------------------------------------------------------------------------------------------------------------------------**


# ------------------------------------------------------------------------ MAPS PAGE ------------------------------------------------------------------------
maps = html.Div(
    children=[
        html.H1("MSA Maps", style={'border':'3px solid black'}),
        html.H2(
            "Explore Interactive Maps of the Top 100 Most Populated Metropolitan Areas",
            style={'fontSize': '16px', 'color': '#555', 'marginBottom': '20px'}
        ),

        html.Div(
            children=[
                dcc.Dropdown(
                    id='dropdown_maps',
                    value="Population",
                    options=[{"label": k, "value": k} for k in map_options.keys()],
                    style={
                        'width': '300px',
                        'margin': '0 auto',
                        'padding': '4px',
                        'border': '1px solid #2C3E50',
                        'borderRadius': '4px',
                        'fontSize': '14px',
                        'textAlign': 'left',
                        'boxShadow': '1px 1px 3px rgba(0,0,0,0.1)'
                    }
                )
            ],
            style={'display': 'flex', 'justifyContent': 'center', 'padding': '10px'}
        ),

        html.Br(),

        # --- Race Radio Buttons (Visible only if 'Race' selected) ---
        html.Div(
            id='race_radio_section',
            children=[
                html.Label("Select Race to Visualize: \n", style={'fontWeight': 'bold'}),
                html.Label("\nHispanic is not a race(Ethnicity ≠ Race), so adding it to the races double-counts people. \n", style={'fontWeight': 'bold'}),
                html.Label("\nCheck Hispanic Tab to see Hispanic Population.", style={'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='race_radio',
                    options=[
                        {'label': "White", 'value': 'Percent_White'},
                        {'label': 'Black', 'value': 'Percent_Black_or_African_American'},
                        {'label': 'American Indian', 'value': 'Percent_American_Indian'},
                        {'label': 'Asian', 'value': 'Percent_Asian'},
                        {'label': 'Hawaiian', 'value': 'Percent_Hawaiian'},
                        {'label': 'Other', 'value': 'Percent_Other'}
                    ],
                    value='Percent_White',
                    labelStyle={'display': 'inline-block', 'margin': '0 12px'},
                    style={'marginTop': '10px'}
                )
            ],
            style={'display': 'none', 'textAlign': 'center', 'marginTop': '20px'}
        ),

        # --- Gender Radio Buttons (Visible only if 'Race' selected) ---
        html.Div(
            id='gender_radio_section',
            children=[
                html.Label("Select Gender to Visualize:", style={'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='gender_radio',
                    options=[
                        {'label': "Female", 'value': 'Percent_Female'},
                        {'label': 'Male', 'value': 'Percent_Male'}
                    ],
                    value='Percent_Female',
                    labelStyle={'display': 'inline-block', 'margin': '0 12px'},
                    style={'marginTop': '10px'}
                )
            ],
            style={'display': 'none', 'textAlign': 'center', 'marginTop': '20px'}
        ),

        # --- Radio Button Education if "Education" is Selected ---
        html.Div(
            id = 'education_button_sec',
            children=[
                html.Label("Select Education Level attained by 25-64 Year Olds", style={'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id = 'education_buttons',
                    options=[
                        {"label": 'Less Than High School', 'value':"Percent_25-64_Year_Olds_With_Less_Than_High_School_Diplomna"},
                        {'label':"High School Diploma", 'value':"Percent_25-64_Year_Olds_With_High_School_Diploma"},
                        {'label': "Some College or Associates Degree", "value":"Percent_25-64_Year_Olds_With_Some_College_or_Associates_Degree"},
                        {'label':"Bachelors Degree or Higher", "value":"Percent_25-64_Year_Olds_With_Bachelor_Degree_or_Higher"}
                    ],
                    value="Percent_25-64_Year_Olds_With_Less_Than_High_School_Diplomna",
                    labelStyle={'display': 'inline-block', 'margin': '0 12px'},
                    style={'marginTop': '10px'}                    
                )
            ],
            style={'display': 'none', 'textAlign': 'center', 'marginTop': '20px'}
        ),

        # --- Drop Down (Visible Only if 'Foreign Born Population is selected) ---
        html.Div(
            id = 'foreign_born_dropdown_sec',
            children=[
                html.Label("Select Continent/Region to Visualize\n**NOTE** data is based off of Total Immigrant Population",
                           style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id = 'imm_dropdown',
                    options= [{'label':'All', 'value':'Percent_Immigrant_Population'}] + [{'label':cont, 'value':f"Percent_of_Immigrants_from_{cont}"} for cont in continent_names],
                    value = 'Percent_Immigrant_Population',
                    style={'marginTop': '10px'}
                )
            ],
            style={'display': 'none', 'textAlign': 'center', 'marginTop': '20px'}
        ),

        html.Br(),

        html.Div(
            children=[
                dcc.Graph(id = "Maps", style={
                    'height': '700px',
                    'border': '1px solid #dee2e6',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                }), 
                
            ],
            style={'text-align':'center', 'display':'inline-block', 'width':'75%'}  
        ),

        html.Br(), html.Br(), html.Br(), html.Br(), html.Br() ## Add Many Break Lines For Space
    ],
    style={'text-align':'center', 'display':'inline-block', 'width':'100%'} 
)
# ------------------------------------------------------------------------ MAPS PAGE ------------------------------------------------------------------------

# ------------------------------------------------------------------------ DEMOGRAPHICS PAGE ------------------------------------------------------------------------
demographics = html.Div(children=[
    html.H1("Demographics", style={'border': '3px solid black'}),

    dcc.Dropdown(
        id="Dem1_Dropdown",
        options=[{"label": name, "value": name} for name in top_100_msa],
        value=top_100_msa[0],
        style={'width': '520px', 'height': '67px', 'border': '3px solid black', 'margin': '0 auto'}
    ),

    html.Br(),
    html.H2("General Information", style={'fontSize': '24px', 'color': '#555', 'marginBottom': '20px'}),

    html.Div(
        id="msa_summary_metrics",
        children=[
            html.Div([
                html.Div("\U0001F465", style={'fontSize': '28px'}),
                html.H4("Population", style={'margin': '4px 0', 'color': '#2C3E50'}),
                html.Div(id="metric_population", style={'fontSize': '20px', 'fontWeight': 'bold', 'color': '#1F618D'})
            ], style={
                'flex': '1',
                'padding': '16px',
                'borderRadius': '12px',
                'backgroundColor': '#EAF2F8',
                'boxShadow': '0 1px 4px rgba(0,0,0,0.1)',
                'margin': '0 10px'
            }),

            html.Div([
                html.Div("\U0001F4B0", style={'fontSize': '28px'}),
                html.H4("Median Single-Earner Income", style={'margin': '4px 0', 'color': '#2C3E50'}),
                html.Div(id="metric_income", style={'fontSize': '20px', 'fontWeight': 'bold', 'color': '#117864'})
            ], style={
                'flex': '1',
                'padding': '16px',
                'borderRadius': '12px',
                'backgroundColor': '#E8F8F5',
                'boxShadow': '0 1px 4px rgba(0,0,0,0.1)',
                'margin': '0 10px'
            }),

            html.Div([
                html.Div("\U0001F4CA", style={'fontSize': '28px'}),
                html.H4("Median Age", style={'margin': '4px 0', 'color': '#2C3E50'}),
                html.Div(id="metric_age", style={'fontSize': '20px', 'fontWeight': 'bold', 'color': '#6C3483'})
            ], style={
                'flex': '1',
                'padding': '16px',
                'borderRadius': '12px',
                'backgroundColor': '#F5EEF8',
                'boxShadow': '0 1px 4px rgba(0,0,0,0.1)',
                'margin': '0 10px'
            })
        ],
        style={
            'display': 'flex',
            'justifyContent': 'center',
            'textAlign': 'center',
            'marginBottom': '30px',
            'marginTop': '20px'
        }
    ),

    html.Div([
        dcc.Graph(id="Pies_Dem", style={
            'height': '700px',
            'border': '1px solid #dee2e6',
            'borderRadius': '8px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
        })
    ], style={'textAlign': 'center', 'width': '92.5%', 'margin': '0 auto'}),

    html.Br(),

    html.Div([
        html.H2("Time Series & Age", style={
            'fontSize': '24px',
            'color': '#555',
            'textAlign': 'center',
            'width': '100%'
        }),

        html.Div([
            html.Div([
                dcc.RadioItems(
                    id='time_series_buttons',
                    options=[
                        {'label': "Population", 'value': "Population"},
                        {'label': "Median Income", 'value': "Median_Income"}
                    ],
                    value="Population",
                    labelStyle={'display': 'inline-block', 'marginRight': '20px'},
                    style={'marginBottom': '20px', 'textAlign': 'center'}
                ),
                dcc.Graph(id='time_series_graph', config={'responsive': True})
            ], style={
                'width': '49%',
                'minWidth': '350px',
                'maxWidth': '600px',
                'padding': '10px',
                'boxSizing': 'border-box',
                'flexShrink': '0'
            }),

            html.Div([
                dcc.RadioItems(
                    id='age_gender_selector',
                    options=[
                        {'label': 'All', 'value': 'All'},
                        {'label': 'Male', 'value': 'Male'},
                        {'label': 'Female', 'value': 'Female'}
                    ],
                    value='All',
                    labelStyle={'display': 'inline-block', 'marginRight': '20px'},
                    style={'marginBottom': '20px', 'textAlign': 'center'}
                ),
                dcc.Graph(id='age_distribution_graph', config={'responsive': True})
            ], style={
                'width': '49%',
                'minWidth': '350px',
                'maxWidth': '600px',
                'padding': '10px',
                'boxSizing': 'border-box',
                'flexShrink': '0'
            })
        ], style={
            'display': 'flex',
            'flexDirection': 'row',
            'flexWrap': 'wrap',
            'justifyContent': 'space-between',
            'alignItems': 'flex-start',
            'width': '100%',
            'maxWidth': '1200px',
            'margin': '0 auto'
        })
    ]),

    html.Br(),

    html.Div([
        html.H2("Immigration Overview", style={
            'fontSize': '26px',
            'color': '#333',
            'marginBottom': '20px',
            'textAlign': 'center'
        }),

        html.Div([
            html.Div([
                html.Div(id='foreign_born_box', style={
                    'backgroundColor': '#ffffff',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.06)',
                    'textAlign': 'center',
                    'border': '1px solid #e1e1e1',
                    'fontSize': '16px',
                    'color': '#333'
                })
            ], style={
                'flex': '1 1 250px',
                'minWidth': '240px',
                'maxWidth': '280px',
                'alignSelf': 'flex-start'
            }),

            html.Div([
                dcc.Graph(
                    id='immigration_over_time',
                    config={'displayModeBar': False},
                    style={'height': '400px', 'width': '100%'}
                )
            ], style={
                'flex': '2 1 400px',
                'minWidth': '350px',
                'maxWidth': '500px'
            }),

            html.Div([
                dcc.Dropdown(
                    id="continent_selector",
                    options=[
                        {'label': label, 'value': value}
                        for label, value in [
                            ('Asia', 'asia'), ('Europe', 'europe'), ('Africa', 'africa'),
                            ('Americas', 'americas'), ('Oceania', 'oceania')
                        ]
                    ],
                    value='asia',
                    placeholder="Select a continent",
                    style={'marginBottom': '15px'}
                ),
                dcc.Graph(
                    id='immigration_by_continent',
                    config={'displayModeBar': False},
                    style={'height': '400px', 'width': '100%'}
                )
            ], style={
                'flex': '2 1 400px',
                'minWidth': '350px',
                'maxWidth': '500px'
            })

        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'gap': '25px',
            'justifyContent': 'center',
            'padding': '20px',
            'backgroundColor': '#f9f9f9',
            'borderRadius': '8px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.03)'
        })
    ])

], style={
    'textAlign': 'center',
    'width': '100%',
    'maxWidth': '1400px',
    'margin': '0 auto'
})



house_income = scrape_data(metro_str, ["NAME", "B19013_001E"], ["Name", "Median_Household_Income", "GEOID"])
house_income = house_income.query("GEOID in @GEOIDS")

house_income["Name"] = house_income["Name"].str[:-11]
house_income["Median_Household_Income"] = house_income["Median_Household_Income"].astype(int)

# -------------------------------------------------------------------------- ECONOMY PAGE --------------------------------------------------------------------------
economy = html.Div(children=[
    html.H1("Economy Data", style={'border':'3px solid black'}),
    dcc.Dropdown(
        id = "Econ1_Dropdown",
        options=[{"label":name, "value":name} for name in top_100_msa],
        value = top_100_msa[0],
        style={'width':'520px', 'height':'67px', 'border':'3px solid black', 'margin':'0 auto'}
    ),
    html.Br(),

    html.Div([
        dcc.RadioItems(
            id="econ_metric_selector",
            options=[
                {"label": "Total Non-Farm Workers", "value": "nonfarm"},
                {"label": "Unemployment Rate", "value": "unemployment"}
            ],
            value="nonfarm",
            labelStyle={'display': 'inline-block', 'marginRight': '20px'},
            style={'textAlign': 'center', 'marginBottom': '20px'}
        ),

        dcc.Graph(id="non_farm", style={
        'height': '600px',  
        'border': '1px solid #dee2e6',
        'borderRadius': '8px',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
        'marginBottom': '0px',
        'paddingBottom': '0px',
        'paddingTop': '0px',
        'marginTop': '0px'
    })
    ], style={'text-align': 'center', 'display': 'inline-block', 'width': '92.5%'}),

    html.Br(),

    html.Div([
        html.H2("Income & Poverty", style={'fontSize': '24px', 'color': '#555', 'marginTop': '30px'}),

        html.Div([
            # Median Household Income Box
            html.Div([
                html.Div("💵", style={'fontSize': '28px'}),
                html.H4("Median Household Income", style={'margin': '4px 0', 'color': '#2C3E50'}),
                html.Div(id="metric_median_income", style={
                    'fontSize': '20px',
                    'fontWeight': 'bold',
                    'color': '#117A65'
                })
            ], style={
                'flex': '1',
                'padding': '16px',
                'borderRadius': '12px',
                'backgroundColor': '#E8F8F5',
                'boxShadow': '0 1px 4px rgba(0,0,0,0.1)',
                'margin': '0 10px'
            }),

            # Poverty Rate Box
            html.Div([
                html.Div("📉", style={'fontSize': '28px'}),
                html.H4("Poverty Rate", style={'margin': '4px 0', 'color': '#2C3E50'}),
                html.Div(id="metric_poverty_rate", style={
                    'fontSize': '20px',
                    'fontWeight': 'bold',
                    'color': '#C0392B'
                })
            ], style={
                'flex': '1',
                'padding': '16px',
                'borderRadius': '12px',
                'backgroundColor': '#FDEDEC',
                'boxShadow': '0 1px 4px rgba(0,0,0,0.1)',
                'margin': '0 10px'
            })
        ], style={
            'display': 'flex',
            'justifyContent': 'center',
            'textAlign': 'center',
            'marginTop': '20px',
            'marginBottom': '30px'
        })
    ]),

    html.Br(),

    html.Div([
        html.Div([
            # === Income Distribution Graph ===
            html.Div([
                html.H4("Income Distribution by Race/Ethnicity", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id="race_income_selector",
                    options=[{"label": r, "value": r} for r in income_distribution_by_race["Race"].unique()],
                    value="All",
                    style={'width': '90%', 'margin': '10px auto'}
                ),
                dcc.Graph(id="income_dist_graph", style={'height': '500px'})
            ], style={
                'flex': '1',
                'margin': '10px',
                'padding': '15px',
                'border': '1px solid #ccc',
                'borderRadius': '10px',
                'boxShadow': '0 2px 6px rgba(0,0,0,0.05)',
                'backgroundColor': '#fdfdfd'
            }),

            # === Poverty Rate Time Series ===
            html.Div([
                html.H4("Poverty Rate Over Time", style={'textAlign': 'center'}),
                dcc.Graph(id="poverty_rate_graph", style={'height': '550px'})
            ], style={
                'flex': '1',
                'margin': '10px',
                'padding': '15px',
                'border': '1px solid #ccc',
                'borderRadius': '10px',
                'boxShadow': '0 2px 6px rgba(0,0,0,0.05)',
                'backgroundColor': '#fdfdfd'
            })

        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})
    ]),

    html.Br(),

    html.Div([
        html.H2("Industry Distribution", style={'fontSize': '24px', 'color': '#555'}),
        dcc.Graph(id="industry_treemap", style={'height': '600px'})
    ])


], style={'text-align':'center', 'display':'inline-block', 'width':'100%'})



# =================================================================== Education Social Page ===================================================================
social = html.Div(children=[
    html.H1("Social, Family, and Education Data", style={'border':'3px solid black'}),
    dcc.Dropdown(
        id = "Social1_Dropdown",
        options=[{"label":name, "value":name} for name in top_100_msa],
        value = top_100_msa[0],
        style={'width':'520px', 'height':'67px', 'border':'3px solid black', 'margin':'0 auto'}
    ),
    html.Br(),

    html.Div([
        html.H2("Household Overview", style={'fontSize': '24px', 'color': '#555'}),

        html.Div([
            html.Div([
                html.H4("Total Households", style={'marginBottom': '5px'}),
                html.Div(id="total_households_box", style={
                    'fontSize': '20px',
                    'fontWeight': 'bold',
                    'color': '#1F618D'
                })
            ], style={
                'flex': '1',
                'padding': '20px',
                'borderRadius': '12px',
                'backgroundColor': '#EAF2F8',
                'boxShadow': '0 1px 4px rgba(0,0,0,0.1)',
                'marginRight': '20px',
                'textAlign': 'center',
                'minWidth': '240px'
            }),

            html.Div([
                html.H4("Family Households", style={'marginBottom': '5px'}),
                html.Div(id="family_households_box", style={
                    'fontSize': '20px',
                    'fontWeight': 'bold',
                    'color': '#117864'
                })
            ], style={
                'flex': '1',
                'padding': '20px',
                'borderRadius': '12px',
                'backgroundColor': '#E8F8F5',
                'boxShadow': '0 1px 4px rgba(0,0,0,0.1)',
                'textAlign': 'center',
                'minWidth': '240px'
            }),
        ], style={
            'display': 'flex',
            'justifyContent': 'center',
            'marginBottom': '40px',
            'flexWrap': 'wrap'
        }),

        html.Div([
            html.Div([
                html.H3("Household Size Distribution by Type", style={'fontSize': '20px', 'marginBottom': '10px'}),
                dcc.RadioItems(
                    id="household_type_selector",
                    options=[
                        {'label': 'Family Households', 'value': 'Family Household'},
                        {'label': 'Nonfamily Households', 'value': 'Nonfamily Households'}
                    ],
                    value='Family Household',
                    labelStyle={'display': 'inline-block', 'marginRight': '20px'}
                ),
                dcc.Graph(id="household_size_dist", style={'height': '500px'})
            ], style={'flex': '1', 'padding': '10px', 'minWidth': '50%'}),

            html.Div([
                html.H3("Languages Spoken at Home", style={'fontSize': '20px', 'marginBottom': '10px'}),
                dcc.Graph(id="language_spoken_pie", style={'height': '500px'})
            ], style={'flex': '1', 'padding': '10px', 'minWidth': '50%'})

        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})
    ]),

    html.Div([
        html.H2("Education Levels of Population (25-64)", style={'fontSize': '24px', 'color': '#555'}),

        html.Div([
            dcc.Graph(id="education_overall_bar", style={'height': '600px', 'width': '48%'}),

            dcc.Graph(id="income_by_edu_bar", style={'height': '600px', 'width': '48%'})
        ], style={
            'display': 'flex',
            'justifyContent': 'space-around',
            'flexWrap': 'wrap',
            'marginTop': '20px'
        })
    ])


], style={'text-align':'center', 'display':'inline-block', 'width':'100%'})



## PAGE LINK CALLBACK ###########################################
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def page_toggle(pathname):
    if pathname == "/Maps":
        return maps
    
    elif pathname == "/Demographics":
        return demographics
    
    elif pathname == "/economy":
        return economy
    
    elif pathname == "/social":
        return social

    else:
        return home_page
## PAGE LINK CALLBACK ###########################################

# -- Map Page Race Radios
@app.callback(
    Output('race_radio_section', 'style'),
    Input('dropdown_maps', 'value')
)
def toggle_race_selector(selected):
    if selected == "Race":
        return {'display': 'block', 'textAlign': 'center', 'marginTop': '20px'}
    return {'display': 'none'}
# -- Map Page Race Radios

# -- Map Page Gender Radios
@app.callback(
    Output('gender_radio_section', 'style'),
    Input('dropdown_maps', 'value')
)
def toggle_gender_selector(selected):
    if selected == "Population by Gender":
        return {'display': 'block', 'textAlign': 'center', 'marginTop': '20px'}
    return {'display': 'none'}
# -- Map Page Gender Radios


# -- Map Page Education Buttons
@app.callback(
    Output('education_button_sec', 'style'),
    Input('dropdown_maps', 'value')
)
def toggle_ed_selector(selected):
    if selected == "Education":
        return {'display': 'block', 'textAlign': 'center', 'marginTop': '20px'}
    return {'display': 'none'}
# -- Map Page Education Buttons


# -- Map Page Foreign Pop Dropdown
@app.callback(
    Output('foreign_born_dropdown_sec', 'style'),
    Input('dropdown_maps', 'value')
)
def toggle_race_selector(selected):
    if selected == "Foreign Born Population":
        return {'display': 'block', 'textAlign': 'center', 'marginTop': '20px'}
    return {'display': 'none'}
# -- Map Page Foreign Pop Dropdown



# MAPS CALLBACK #################################################
@app.callback(
    Output('Maps', 'figure'),
    Input('dropdown_maps', 'value'),
    Input('race_radio', 'value'),
    Input('gender_radio', 'value'),
    Input('education_buttons','value'),
    Input('imm_dropdown', 'value')
)
def update_map(selected_option, selected_race, selected_gender, selected_ed_level, selected_region):
    extra_col = None # For Extra Info For Immigrants

    if not selected_option:
        return go.Figure()

    if selected_option == "Race":
        df = race_df
        col = selected_race
    elif selected_option == "Population by Gender":
        df=all_map_data
        col = selected_gender
    elif selected_option == "Hispanic Population(Ethnicity)":
        df = hisp
        col = map_options[selected_option]
        extra_col = "Hispanic_or_Latino_Total"
    elif selected_option == "Unemployment Rate":
        df = unemployment_df
        col = "Unemployment_Rate"
    elif selected_option == "Total Nonfarm Workers":
        df = nonfarm_df
        extra_col = map_options[selected_option]
        col = "Percent_Working_Nonfarm_Jobs"
    elif selected_option == "Foreign Born Population":
        df = imm
        col = selected_region
        extra_col = "Total_Immigrant_Population"
    elif selected_option == "Poverty Rate":
        df = poverty_mapping
        col = map_options[selected_option]
    elif selected_option == "Education":
        df = ed 
        col = selected_ed_level
        extra_col = "Total_25-64"
    else:
        df = all_map_data
        col = map_options[selected_option]

    if not extra_col:
        fig = px.scatter_map(
            df,
            lat='INTPTLAT',
            lon='INTPTLON',
            color=col,
            size='Population_2024',
            hover_name='Name',
            hover_data={
                col: True,
                'Population_2024': True,
                'INTPTLAT': False,
                'INTPTLON': False
            },
            zoom=(2 if selected_option != "Education" else 1.5),
            height=500
        )
        fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        return fig
    
    else:
        fig = px.scatter_map(
            df,
            lat='INTPTLAT',
            lon='INTPTLON',
            color=col,
            size='Population_2024',
            hover_name='Name',
            hover_data={
                col: True, extra_col: True,
                'Population_2024': True,
                'INTPTLAT': False,
                'INTPTLON': False
            },
            zoom=(2 if selected_option != "Education" else 1.5),
            height=500
        )
        fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        return fig


# =========================== Demographics Page ===========================
@app.callback(
    Output("Pies_Dem", "figure"),
    Input("Dem1_Dropdown", 'value')
)
def pie_charts(selected_msa):
    if not selected_msa:
        return go.Figure()
    
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "xy"}, {"type": "domain"}, {"type": "domain"}]],
        subplot_titles=[
            "Race Distribution",
            "Hispanic vs Non-Hispanic",
            "Gender Distribution"
        ],
        horizontal_spacing=0.1875
    )

    # === Gender Pie Data ===
    temp_gender = gender_pie.query("Name == @selected_msa")
    temp_gender = pd.melt(temp_gender, id_vars=["Name", "GEOID"], var_name="Gender", value_name="Percent")

    # === Hispanic Pie ===
    temp_hisp = hisp_pie.query("Name == @selected_msa")
    temp_hisp = pd.melt(temp_hisp, id_vars=["Name"], value_name="Total", var_name="Hispanic")

    # === Race Bar Data ===
    race_cols = [
        "White", "Black_or_African_American", "American_Indian_and_Alaska_Native",
        "Asian", "Native_Hawaiian_Other_Pacific_Islander", "Some_Other_Race"
    ]
    temp_race = race_pie.query("Name == @selected_msa")[["Name"] + race_cols]
    temp_race = pd.melt(temp_race, id_vars=["Name"], var_name="Race", value_name="Amount")

    # === Race Bar Chart ===
    fig.add_trace(
        go.Bar(
            x=temp_race["Race"],
            y=temp_race["Amount"],
            marker_color='#3498db',
            name="Race"
        ),
        row=1, col=1
    )

    # === Hispanic Pie Chart ===
    fig.add_trace(
        go.Pie(
            labels=temp_hisp["Hispanic"],
            values=temp_hisp["Total"],
            hole=0.4,
            textinfo='percent+label',
            name="Hispanic",
            insidetextorientation='radial',
            textfont_size=11
        ),
        row=1, col=2
    )

    # === Gender Pie Chart ===
    fig.add_trace(
        go.Pie(
            labels=temp_gender["Gender"],
            values=temp_gender["Percent"],
            hole=0.4,
            textinfo='percent+label',
            name="Gender",
            insidetextorientation='radial'
        ),
        row=1, col=3
    )

    fig.update_layout(
        title_text=f"Demographics Breakdown: {selected_msa}",
        title_x=0.5,
        height=600,
        margin=dict(t=60, b=40, l=40, r=40),
        uniformtext_minsize=10,
        uniformtext_mode='hide'
    )

    fig.update_xaxes(title_text="Race Categories", row=1, col=1, tickfont=dict(size=7.5), tickangle=90)
    fig.update_yaxes(title_text="Population", row=1, col=1)

    return fig

#                                      Name  GEOID          Gender  Percent
# 0  New York-Newark-Jersey City, NY-NJ  35620    Percent_Male    48.71
# 1  New York-Newark-Jersey City, NY-NJ  35620  Percent_Female    51.29
#                                  Name                               Race   Amount
# 0  New York-Newark-Jersey City, NY-NJ                              White  9556729
# 1  New York-Newark-Jersey City, NY-NJ          Black_or_African_American  3218889
# 2  New York-Newark-Jersey City, NY-NJ  American_Indian_and_Alaska_Native   107515
# 3  New York-Newark-Jersey City, NY-NJ                              Asian  2332293
# 4  New York-Newark-Jersey City, NY-NJ                           Hawaiian     8622

@app.callback(
    Output("metric_population", "children"),
    Output("metric_income", "children"),
    Output("metric_age", "children"),
    Input("Dem1_Dropdown", "value")
)
def update_metrics(msa_name):
    if not msa_name or msa_name not in name_to_geoid:
        return "N/A", "N/A", "N/A"

    row = all_map_data.query("Name == @msa_name").iloc[0]
    pop = f"{int(row['Population_2024']):,}"
    income = f"${int(row['Median_Income']):,}"
    age = f"{row['Median_Age']:.1f}"
    return pop, income, age

# ============== Time Series ==============
@app.callback(
    Output('time_series_graph', 'figure'),
    Input('Dem1_Dropdown', 'value'),
    Input('time_series_buttons', 'value')
)
def update_time_series(selected_msa, selected_metric):
    if not selected_msa or selected_msa not in name_to_geoid:
        return go.Figure()
    
    if selected_metric == "Median_Income":
        df = inc_time_series_data
        y_col = "Median_Income"
    else:
        df = pop_time_series_data
        y_col = "Population"

    temp_id = str(name_to_geoid[selected_msa])
    filtered_df = df[df["GEOID"] == temp_id]

    fig = px.line(
        filtered_df,
        x='Year',
        y=y_col,
        title=f"{selected_metric.replace('_', ' ')} Over Time in {selected_msa}",
        markers=True
    )
    fig.update_layout(margin=dict(t=40), height=400, title_x=0.5)
    return fig



# ======= Age Distribution Callback =======
@app.callback(
    Output('age_distribution_graph', 'figure'),
    Input('Dem1_Dropdown', 'value'),
    Input('age_gender_selector', 'value')
)
def update_age_distribution(selected_msa, selected_gender):
    if not selected_msa:
        return go.Figure()
    if selected_gender == 'Male':
        df = male_age_data
    elif selected_gender == 'Female':
        df = female_age_data
    else:
        df = df_combined_age

    filtered = df[df["Name"] == selected_msa]
    fig = px.bar(
        filtered,
        x='Age_Range',
        y='Population',
        title=f"Age Distribution ({selected_gender}) in {selected_msa}",
        labels={'Population': 'Population Count'}
    )
    fig.update_layout(
        title_x=0.5,
        margin=dict(t=40),
        height=400
    )
    return fig


# ====== Immigration Box Callback
@app.callback(
    Output('foreign_born_box', 'children'),
    Input('Dem1_Dropdown', 'value')
)
def update_foreign_born_box(selected_msa):
    pop = imm_total.query("Name == @selected_msa")["Total_Immigrant_Population"]
    if not pop.empty:
        return [
            html.H5("Foreign Born Population", style={'color': '#2c3e50', 'marginBottom': '5px'}),
            html.H3(f"{int(pop.values[0]):,}", style={'fontSize': '22px', 'color': '#2980B9', 'margin': 0})
        ]
    else:
        return [
            html.H5("Foreign Born Population", style={'color': '#2c3e50'}),
            html.P("Data not available", style={'fontSize': '16px', 'color': '#999'})
        ]


# =============== Immigration Time Series 
@app.callback(
    Output('immigration_over_time', 'figure'),
    Input('Dem1_Dropdown', 'value')
)
def update_immigrant_time_series(selected_msa):
    if not selected_msa or selected_msa not in name_to_geoid:
        return go.Figure()
    temp_id = name_to_geoid[selected_msa]
    plotting_data = imm_time_series_data.query('GEOID == @temp_id')

    fig = px.line(
        plotting_data,
        x='Year',
        y="Total_Foreign_Born_Population",
        title=f"Foreign Born Population Over Time in {selected_msa}",
        markers=True
    )
    fig.update_layout(margin=dict(t=40), height=400, title_x=0.5, title_font = dict(size = 8), yaxis_title="Foreign Born Population")
    return fig



# ================ Immigrant by country callback
@app.callback(
    Output('immigration_by_continent', 'figure'),
    Input('Dem1_Dropdown', 'value'),
    Input('continent_selector', 'value')
)
def update_immigration_by_country(selected_msa, selected_continent):
    if not selected_msa or not selected_continent:
        return go.Figure()
    region_map = {
        'europe': europe_df,
        'asia': asia_df,
        'africa': africa_df,
        'americas': americas_df,
        'oceania': oceania_df
    }

    df = region_map[selected_continent]
    filtered = df[df["Name"] == selected_msa]

    if selected_continent != "Oceania":
        num_bars = (int(np.round(len(filtered)*0.35)))
    else:
        num_bars = len(oceania_df)

    fig = px.bar(
        filtered.sort_values("Population", ascending=False).head(num_bars),
        x="Country",
        y="Population",
        title=f"{selected_msa} — Foreign-Born Population from {selected_continent.capitalize()}",
        labels={"Population": "Population", "Country": "Country of Origin"}
    )
    fig.update_layout(title_x=0.5, height=400, title_font = dict(size = 8))
    return fig

# ================================== Econ
@app.callback(
    Output("non_farm", "figure"),
    Input("Econ1_Dropdown", "value"),
    Input("econ_metric_selector", "value")
)
def update_econ_graph(selected_msa, selected_metric):
    if not selected_msa:
        return go.Figure()
    
    if selected_metric == "unemployment" and selected_msa == "Kiryas Joel-Poughkeepsie-Newburgh, NY":
        fig = px.bar(
            data_frame= pough_ur,
            x = "Year", y = "Unemployment_Rate",
            title = "Unemployment Rate in Kiryas Joel-Poughkeepsie-Newburgh, NY",
        )
        fig.update_layout(title_x=0.5, height=500, margin=dict(t=40), yaxis_title="Unemployment Rate (%)")
        return fig

    if selected_metric == "nonfarm":
        df = nonfarm_df_all.query("Name == @selected_msa").copy()
        y_col = "Non_Farm_Workers"
        title = f"Total Non-Farm Workers in {selected_msa}"
        y_title = "Non-Farm Workers"

    elif selected_metric == "unemployment":
        df = unemployment_historic.query("Name == @selected_msa").copy()
        y_col = "Unemployment_Rate"
        title = f"Unemployment Rate in {selected_msa}"
        y_title = "Unemployment Rate (%)"


    # Handle column rename
    if 'index' in df.columns:
        df.rename(columns={"index": "Date"}, inplace=True)
    elif 'DATE' in df.columns:
        df.rename(columns={"DATE": "Date"}, inplace=True)

    # df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    fig = px.line(
        df,
        x="Date",
        y=y_col,
        title=title,
        markers=True
    )
    fig.update_layout(title_x=0.5, height=500, margin=dict(t=40), yaxis_title=y_title)
    return fig

# ===================== Econ text boxes =====================
@app.callback(
    Output("metric_median_income", "children"),
    Output("metric_poverty_rate", "children"),
    Input("Econ1_Dropdown", "value")
)
def update_income_poverty(selected_msa):
    if not selected_msa:
        return "N/A", "N/A"
    income_row = house_income.query("Name == @selected_msa")
    poverty_row = poverty.query("Name == @selected_msa")

    median_income = "${:,.0f}".format(income_row["Median_Household_Income"].values[0]) if not income_row.empty else "N/A"
    poverty_rate = "{:.1f}%".format(poverty_row["Poverty_Rate"].values[0]) if not poverty_row.empty else "N/A"

    return median_income, poverty_rate


# ========================== Income Distribution ========================
@app.callback(
    Output("income_dist_graph", "figure"),
    Input("Econ1_Dropdown", "value"),
    Input("race_income_selector", "value")
)
def update_income_dist(selected_msa, selected_race):
    df = income_distribution_by_race.query("Name == @selected_msa and Race == @selected_race")

    fig = px.bar(
        df,
        x="Income_Bracket",
        y="Count",
        title=f"Income Distribution – {selected_race} Households in {selected_msa}",
        labels={"Count": "Number of Households", "Income_Bracket": "Income Bracket"}
    )
    fig.update_layout(title_x=0.5, xaxis_tickangle=-30)
    return fig


# ====================== POverty time series
@app.callback(
    Output("poverty_rate_graph", "figure"),
    Input("Econ1_Dropdown", "value")
)
def update_poverty_graph(selected_msa):
    if not selected_msa or selected_msa not in name_to_geoid:
        return go.Figure()
    temp_id = name_to_geoid[selected_msa]
    temp = poverty_rate_time_series_data.query('GEOID == @temp_id').copy()
    temp = temp.sort_values(by="Year")

    fig = px.line(
        temp,
        x="Year",
        y="Poverty_Rate",
        markers=True,
        title=f"Poverty Rate in {selected_msa} Over Time"
    )
    fig.update_layout(title_x=0.5, height=500, margin=dict(t=40), yaxis_title="Poverty Rate (%)")
    return fig



@app.callback(
    Output("industry_treemap", "figure"),
    Input("Econ1_Dropdown", "value")
)
def update_industry_pie(selected_msa):
    if not selected_msa:
        return go.Figure()
    temp_df = industry_long.query('Name == @selected_msa')
    fig = px.pie(data_frame=temp_df, values="Workers", names = "Industry")
    fig.update_traces(hovertemplate="%{label}")

    return fig

# =========================== Family Boxes
@app.callback(
    Output("total_households_box", "children"),
    Output("family_households_box", "children"),
    Input("Social1_Dropdown", "value")
)
def update_household_metrics(selected_msa):
    row = households_data[households_data["Name"] == selected_msa]
    if row.empty:
        return "N/A", "N/A"
    total = int(row["Total"])
    family = int(row["Family households"])
    return f"{total:,}", f"{family:,}"


# ================================= Family Size Dist
@app.callback(
    Output("household_size_dist", "figure"),
    Input("Social1_Dropdown", "value"),
    Input("household_type_selector", "value")
)
def update_household_size_graph(selected_msa, household_type):
    subset = fam_size[
        (fam_size["Name"] == selected_msa) &
        (fam_size["Size"].str.startswith(household_type))
    ].copy()

    if subset.empty:
        return go.Figure()

    subset["Size Category"] = subset["Size"].str.extract(r"(\d+-person|7-or-more person)")
    subset["Size Category"] = subset["Size Category"].replace("7-or-more person", "7+")
    
    fig = px.bar(
        subset,
        x="Size Category",
        y="Count",
        title=f"{household_type} Household Size in {selected_msa}"
    )
    fig.update_layout(title_x=0.5, yaxis_title="Household Count", xaxis_title="Size")
    return fig


# ===================================== Language Pie chart
@app.callback(
    Output("language_spoken_pie", "figure"),
    Input("Social1_Dropdown", "value")
)
def update_language_pie(selected_msa):
    df = language[language["Name"] == selected_msa]
    if df.empty:
        return px.pie(values=[1], names=["No Data"])

    # Optional: Combine languages with <2% into "Other"
    total = df["Count"].sum()
    df["Percent"] = df["Count"] / total
    df["Language_Category"] = df.apply(lambda row: row["Language"] if row["Percent"] >= 0.02 else "Other", axis=1)
    df_grouped = df.groupby("Language_Category", as_index=False)["Count"].sum()

    fig = px.pie(df_grouped, values="Count", names="Language_Category", title=f"Languages Spoken at Home in {selected_msa}")
    fig.update_layout(title_x=0.5, height=400, margin=dict(t=40))
    return fig

# =================================== Education Pie Chart
@app.callback(
    Output("education_overall_bar", "figure"),
    Input("Social1_Dropdown", "value")
)
def update_education_bar(selected_msa):
    temp_df = ed_all[ed_all["Name"] == selected_msa].copy()
    if temp_df.empty:
        return px.bar(title="No data available")

    fig = px.bar(
        temp_df,
        x="Degree",
        y="Percent_25-64",
        title=f"Education Attainment (25-64) in {selected_msa}",
        labels={"Percent_25-64": "% of Population", "Degree": "Educational Attainment"},
        text="Percent_25-64"
    )

    fig.update_layout(
        title_x=0.5,
        height=650,
        margin=dict(t=40),
        xaxis_tickangle=-30,
        xaxis_tickfont=dict(size=10),
        font=dict(size=9),
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    return fig


@app.callback(
    Output("income_by_edu_bar", "figure"),
    Input("Social1_Dropdown", "value")
)
def update_income_by_edu_bar(selected_msa):
    temp_df = income_by_edu[income_by_edu["Name"] == selected_msa].copy()
    if temp_df.empty:
        return px.bar(title="No data available")

    fig = px.bar(
        temp_df,
        x="Degree",
        y="Median_Income",
        title=f"Median Income by Education in {selected_msa}",
        labels={"Median_Income": "Median Income ($)", "Degree": "Educational Attainment"},
        text="Median_Income"
    )
    fig.update_layout(
        title_x=0.5,
        height=550,
        margin=dict(t=40),
        xaxis_tickangle=30
    )
    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    return fig



# # # plot(fig_msa_pop)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
    ## app.run(debug = True, port = 7117)
    







