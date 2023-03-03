import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rasterio

from rasterio.mask import mask

def open_raster(file):
    return rasterio.open(file)

def open_cdl(year):
    raster_path = r'\\ac-geneva-24\E\grains trading\Streamlit\LocalData\CDL\\'+ str(year) + "/" + str(year) + "_30m_cdls.tif"
    return rasterio.open(raster_path)

def get_geometry(country='usa', aggregate_level='state'):
    '''
    aggregate_level='national', 'state', 'county'

    '''
    folder=r'\\ac-geneva-24\E\grains trading\Streamlit\LocalData\geometry\\'

    if country == 'usa':
        folder=folder+'usa\\'

        if aggregate_level=='state':
            file = 'gadm36_USA_1.shp'
            file_path = folder+file
    
        geometry = gpd.read_file(filename=file_path)
        geometry = geometry.to_crs("EPSG:5070")

    return geometry

def clip_raster_by_shapefile(raster, shapefile, polygon_name_col='NAME_1', suffix=''):
    """
        
    """
    folder=r'\\ac-geneva-24\E\grains trading\Streamlit\LocalData\results\\'
    for i, row in shapefile.iterrows():
        name = row[polygon_name_col].lower()
        print('Clip:', name)

        try:
            output_file = folder + name + suffix+ ".tif"

            data, _ = mask(raster, [row["geometry"]], crop=True) #Keep only the data that fall inside the boundary of the current county
            data = data[0, :, :]

            profile = raster.profile
            profile["height"] = data.shape[0]
            profile["width"] = data.shape[1]
            with rasterio.open(output_file, "w", **profile) as dst:
                dst.write(data, 1)
                dst.write_colormap(1, raster.colormap(1))
            print('Success:', name)

        except Exception as e: #Some counties do not have any ratser data (Northern islands, Puerto Rico, ...)
            print('Issues with:', name)
            print(e)

def get_cmap(raster):
    # https://stackoverflow.com/questions/37719304/python-imshow-set-certain-value-to-defined-color
    cmap = raster.colormap(1)
    colors = [[v for v in list(value)] for key, value in cmap.items()]
    return np.array(colors)

def plot_raster(data, cmap, size=6):
    # https://stackoverflow.com/questions/37719304/python-imshow-set-certain-value-to-defined-color
    RGB = cmap[data]
    fig, ax = plt.subplots(figsize=(size, size))
    plt.imshow(RGB)
    plt.show()

def get_legend():
    legend_path = r'\\ac-geneva-24\E\grains trading\Streamlit\LocalData\CDL\\'+  "legend.csv"
    legend = pd.read_csv(legend_path).rename(columns={"Class_Names": "class", "Codes":"code"}).set_index("code")

    legend["main"] = False
    legend.loc[legend["class"].isin(["Soybeans", "Corn", "Cotton", "Spring Wheat", "Durum Wheat", "Winter Wheat"]), "main"] = True

    return legend

# ------------------------------------------------------------------------------------------------------------------------------------

def get_counties_geometry():
    """
        Loads and returns a GeoDataFrame contaning the geometry of US counties.
    """
    counties_geometry = gpd.read_file("../../data/geometry/cb_2021_us_county_500k.shp")
    counties_geometry = counties_geometry.to_crs("EPSG:5070")
    return counties_geometry

def split_rasters_by_counties(counties_geometry):
    """
        For each year, split the CDL raster into smaller rasters representing each county
    """
    for year in range(2008, 2023):
        raster_path = "../../data/rasters/" + str(year) + "/" + str(year) + "_30m_cdls.tif"

        with rasterio.open(raster_path) as src:
            for i, row in counties_geometry.iterrows():
                state = row["STATE_NAME"].lower()
                county = row["NAME"].lower()

                try:
                    output_path = "../../data/rasters/tiles/" + state + "/" + county + "/"
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    data, _ = mask(src, [row["geometry"]], crop=True) #Keep only the data that fall inside the boundary of the current county
                    data = data[0, :, :]

                    profile = src.profile
                    profile["height"] = data.shape[0]
                    profile["width"] = data.shape[1]
                    with rasterio.open(output_path + str(year) + ".tif", "w", **profile) as dst:
                        dst.write(data, 1)
                        dst.write_colormap(1, src.colormap(1))
                except Exception as e: #Some counties do not have any ratser data (Northern islands, Puerto Rico, ...)
                    print(state)
                    print(county)
                    print(e)

def get_raster(year, state, county):
    """
        Loads and returns the raster corresponding the specified 'year', 'state' and 'county'
    """
    filepath = "../../data/rasters/tiles/" + state +"/" + county + "/" + str(year) + ".tif"

    with rasterio.open(filepath) as src:
        data = src.read(1)
        
        cmap = src.colormap(1)
        cmap = [tuple([v / 255 for v in list(value)]) for key, value in cmap.items()]
        cmap = matplotlib.colors.ListedColormap(cmap)
        
        return data, cmap

def get_rasters_for_county(state, county):
    """
        Get all the rasters fror a given 'state' and 'county'
        'rasters' is a list of tuples containing the actual raster, the year, and the county
    """
    rasters = []
    for year in range(2008, 2023):
        raster, cmap = get_raster(year, state, county)
        rasters.append((raster, year, county))
    return rasters, cmap
    
def get_rasters_for_state(state):
    """
        Get rasters for every years and every counties for a given 'state'. 
        'rasters' is a list of tuples containing the actual raster, the year, and the county
    """
    prefix =  "../../data/rasters/tiles/" + state +"/"

    cmap = None
    rasters = []

    for county in os.listdir(prefix):
        county_rasters, cmap = get_rasters_for_county(state, county)
        rasters = rasters + county_rasters

    return rasters, cmap





def area_metrics(raster, legend):
    """
        For a given raster, computes the #pixel for each categories and convert it to acres.
    """
    values, occurences = np.unique(raster, return_counts=True)
    summary = legend[legend.index.isin(values)].copy()
    summary["count"] = occurences
    summary["acre"] = summary["count"] * 0.222394
    summary["ratio_arable"] = 100 * summary["acre"] / summary[summary["arable"]]["acre"].sum()
    summary["ratio_total"] = 100 * summary["acre"] / summary["acre"].sum()
    summary.loc[~summary["arable"], "ratio_arable"] = np.nan

    return summary

def stacked_historical(summary):
    """
        Display the evolution of 'area_metrics' over the years as an area chart.
    """
    summary = summary[summary["ratio_arable"] >= 1.5]
    summary = summary[summary["arable"]][["class", "acre", "year"]]
    summary.pivot_table(columns="year", index="class").fillna(0).droplevel(0, axis=1).T.plot.area(figsize=(20, 12))

def united_states_area_summary():
    """
        Compute and save the 'area_metrics' of every state/counties for every available year.
    """
    if os.path.exists("./output/summaries/usa.csv"):
        summary = pd.read_csv("./output/summaries/usa.csv")
        return summary
    
    area_summaries = []
    for state in os.listdir("../../data/rasters/tiles/"):
        try:
            rasters, cmap = get_rasters_for_state(state)
            legend = get_legend(cmap)

            for (raster, year, county) in rasters:
                summary = area_metrics(raster, legend)
                summary["year"] = year
                summary["county"] = county
                summary["state"] = state
                area_summaries.append(summary)
        except Exception as e:
            print(e)
            print(state)

    summary = pd.concat(area_summaries)
    summary.to_csv("./output/summaries/usa.csv", index=False)

    return summary

def px_distribution(state, county, rasters, legend, save=True, plot=False):
    """
        Computes, saves and returns the distribution of represented categories over the years for every pixel of the rasters for a given 'state' and 'county'
    """
    class_subset = legend[(legend["main"])]
    class_subset_codes = class_subset.index.tolist()
    stacked = np.stack(rasters, axis=2)

    stacked_subset = stacked.reshape(-1, len(rasters))
    stacked_subset = stacked_subset[np.isin(stacked_subset, class_subset_codes).sum(axis=1) >= 3] #More than twice a main commodity in the last 15 years

    px_cat_distribution = np.apply_along_axis(lambda x: len(set(x)), 1, stacked_subset)

    output_path = "./output/distributions/" + state + "/" + county + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if save:
        with open(output_path + "distribution_nb_categories.npy", 'wb') as f:
            np.save(f, px_cat_distribution)

    if plot:
        plt.title("Distribution of the number of different categories a given arable pixel can represent")
        plt.hist(px_cat_distribution)
        plt.show()

    non_main_subset = set(legend[~legend["main"]].index.tolist())
    px_main_cat_distribution = np.apply_along_axis(lambda x: len(set(x) - non_main_subset), 1, stacked_subset)

    if save:
        with open(output_path + "distribution_nb_categories_main_crop.npy", 'wb') as f:
            np.save(f, px_main_cat_distribution)

    if plot:
        plt.title("Distribution of the number of main commodity a pixel that had main commodity can represent")
        plt.hist(px_main_cat_distribution)
        plt.show()

    return px_cat_distribution, px_main_cat_distribution


def px_pb_corn_soy_rotation_distribution(state, county, rasters, save=True, plot=False):
    """
        Computes, saves and returns the distribution of the number of times the rotation of corn/soybeans has been made over the years for every pixel of the rasters for a given 'state' and 'county'
    """
    output_path = "./output/distributions/" + state + "/" + county + "/"

    stacked = np.stack(rasters, axis=2)
    
    img = stacked.reshape(-1, len(rasters))
    img = img[np.isin(img[:, :], [1, 5]).any(axis=1)] # Keep soybean and corn

    pbs = []

    for commo_code in [1, 5]:
        commo = "corn" if commo_code == 1 else "soybean"

        img = img[np.count_nonzero(img==commo_code, axis=1) > 3] # Keep pixel that had at least corn four times in the last 15 years

        zipped = np.dstack((img[:, 1:], img[:, :-1])) # Group pixel (t, t+1)
        summed = np.sum(zipped, axis=2)

        pbs_rotation = np.count_nonzero(summed==6, axis=1) / summed.shape[1]

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if save:
            with open(output_path + "distribution_pb_rotation_one_" + commo + ".npy", 'wb') as f:
                np.save(f, pbs_rotation)

        if plot:
            plt.title("Probability that a pixel respect the rotation soybean/corn")
            plt.hist(pbs_rotation, bins=15)
            plt.show()

        ratio_rotation_hist = (summed == 6).sum(axis=0) / summed.shape[0]

        if save:
            with open(output_path + "ratio_rotation_hist_" + commo + ".npy", 'wb') as f:
                np.save(f, ratio_rotation_hist)

        pbs.append((pbs_rotation, ratio_rotation_hist))
    
    return pbs

def produce_distributions():
    legend = None
    path = "../../data/rasters/tiles/"
    for state in os.listdir(path):
        try:
            for county in os.listdir(path + state + "/"):
                rasters, cmap = get_rasters_for_county(state, county)
                if legend is None:
                    legend = get_legend(cmap)
                rasters = [r[0] for r in rasters]
                px_cat_distribution, px_main_cat_distribution = px_distribution(state, county, rasters, legend)
                [(pbs_rotation_at_least_one_corn, pb_series_at_least_one_corn), (pbs_rotation_at_least_one_soybean, pb_series_at_least_one_soybean)] = px_pb_corn_soy_rotation_distribution(state, county, rasters)
        except Exception as e:
            print(e)
            pass

non_cultivated = [
    "Other Hay/Non Alfalfa",
    "Sod/Grass Seed",
    "Switchgrass",
    "Forest",
    "Shrubland",
    "Barren",
    "Clouds/No Data",
    "Developed",
    "Water",
    "Wetlands",
    "Nonag/Undefined",
    "Aquaculture",
    "Open Water",
    "Perennial Ice/Snow",
    "Developed/Open Space",
    "Developed/Low Intensity",
    "Developed/Med Intensity",
    "Developed/High Intensity",
    "Barren",
    "Deciduous Forest",
    "Evergreen Forest",
    "Mixed Forest",
    "Shrubland",
    "Grass/Pasture",
    "Woody Wetlands",
    "Herbaceous Wetlands",
]

##### Snippets

"""
rasters, cmap = get_rasters_for_county("alabama", "autauga")
legend = get_legend(cmap)

summaries = []
for i, raster in enumerate(rasters):
    summary = area_metrics(raster, legend)
    summary["year"] = 2008 + i
    summaries.append(summary)
summaries = pd.concat(summaries)
stacked_historical(summaries)
#plot_raster(raster, cmap)
"""

"""
state="alabama"
county="autauga"

summary = united_states_area_summary()
tmp = summary[(summary["state"] == state) & (summary["county"] == county)].groupby(["year", "class"]).sum().reset_index()
tmp = tmp[tmp["class"].isin(["Corn", "Soybeans", "Cotton", "Wheat"])]
tmp = tmp.groupby(["year", "class"]).sum().reset_index()
tmp["arable"] = True
stacked_historical(tmp)

summary = united_states_area_summary()
summary_county = summary[(summary["state"] == state) & (summary["county"] == county)].groupby(["class"]).sum(numeric_only=True)
summary_county["weight"] = 100 * summary_county["acre"] / summary_county["acre"].sum()
"""

"""
img = stacked.reshape(-1, 15)

count_corn_ppx = np.count_nonzero(img==1, axis=1)
count_corn_ppx = count_corn_ppx[count_corn_ppx > 1]

count_soybean_ppx = np.count_nonzero(img==5, axis=1)
count_soybean_ppx = count_soybean_ppx[count_soybean_ppx > 1]

plt.title("# times a pixel has been used for a commodity over last 15 years")
plt.hist(count_corn_ppx, alpha=0.5, label="corn")
plt.hist(count_soybean_ppx, alpha=0.5, label="soybean")
plt.legend()
plt.show()

"""

def clip_cdl_by_state(year):
    src=open_cdl(year)
    usa_states = get_geometry(country='usa', aggregate_level='state')
    clip_raster_by_shapefile(src, usa_states,polygon_name_col='NAME_1', suffix='_'+str(year))


if __name__=='__main__':
    for year in range(2008,2023):
        clip_cdl_by_state(year)