#This file adjusts the dataset to a more manageable form and scope
#Specifically, it filters the dataset to only animals in TX and MA
#It also focuses on predicting animal genus instead of species
#The output patches are organized in a way that is easy for TensorFlow to read in later

import geopandas
import pandas
import os
import matplotlib.pyplot as plt
import zipfile
import shutil

base_data_dir = "../geolife"
output_dir = "data/us_patches_reduced"
us_patches_dir = os.path.join("/Volumes", "LaCie", "geolife", "patches-us")
metadata_dir = os.path.join(base_data_dir, "metadata")

species_details_df = pandas.read_csv(os.path.join(metadata_dir, "species_details.csv"), sep=';')
species_details_df = species_details_df.drop(columns=["GBIF_species_name", "GBIF_species_id"]) #Drop irrelevant columns
																							   #species_id is different from GBIF_species_id

observations_us_raw_df = pandas.read_csv(os.path.join(base_data_dir, "observations", "observations_us_train.csv"), sep=';')

observations_us_raw_gdf = geopandas.GeoDataFrame(observations_us_raw_df, geometry=geopandas.points_from_xy(
	observations_us_raw_df.longitude, observations_us_raw_df.latitude), crs="EPSG:4326") #Set to WGS84 CRS

observations_us_raw_gdf = observations_us_raw_gdf.join(species_details_df, on="species_id", lsuffix="_obs")
observations_us_raw_gdf = observations_us_raw_gdf.drop(columns=["species_id", "species_id_obs"]) #Don't need species ID anymore since focusing on genus

observations_us_raw_gdf = observations_us_raw_gdf[observations_us_raw_gdf["GBIF_kingdom_name"] == "Animalia"]
observations_us_raw_gdf.drop_duplicates(inplace=True)
print("Observations in all of US: ", len(observations_us_raw_gdf))

states_data = geopandas.read_file('USA_States_(Generalized)/USA_States_Generalized.shp')

#Remove states outside of continental US so they don't mess with map
states_data = states_data.loc[states_data["STATE_NAME"] != "Hawaii"]
states_data = states_data.loc[states_data["STATE_NAME"] != "Alaska"]

#Filter to just data in Texas and Massachusetts

texas_data = states_data.loc[states_data["STATE_NAME"] == "Texas"]
points_in_texas = geopandas.sjoin(observations_us_raw_gdf, texas_data)
points_in_texas["state"] = "Texas"
print("Observations in Texas: ", len(points_in_texas))

massachusetts_data = states_data.loc[states_data["STATE_NAME"] == "Massachusetts"]
points_in_massachusetts = geopandas.sjoin(observations_us_raw_gdf, massachusetts_data)
points_in_massachusetts["state"] = "Massachusetts"
print("Observations in Massachusetts: ", len(points_in_massachusetts))

#Randomly sample only some points in TX and MA so we have less data for now
points_in_texas = points_in_texas.sample(frac=0.1)
print("Reduced observations in Texas: ", len(points_in_texas))

points_in_massachusetts = points_in_massachusetts.sample(frac=0.1)
print("Reduced observations in Massachusetts: ", len(points_in_massachusetts))

#Combine TX and MA data and get relevant columns
points = points_in_massachusetts.append(points_in_texas)
points = points[["observation_id", "latitude", "longitude", "GBIF_genus_name", "state"]]
points = points.rename(columns={"GBIF_genus_name": "genus"})
print(points)
print("Number of genus classes: ", len(points["genus"].unique()))

points.to_csv("data/observations.csv")

#Plot data in a map
ax = states_data.plot(color="lightblue", edgecolor="black")
points_in_texas.plot(ax=ax, markersize=5)
points_in_massachusetts.plot(ax=ax, markersize=5)
#plt.show()
plt.savefig("data/observations.png")

counter = 0

#Find and copy relevant patches for the reduced dataset
for index, row in points.iterrows():
    observation_id = str(row["observation_id"])
    first_dir = observation_id[6:]
    second_dir = observation_id[4:6]
    
    patch_output_dir = os.path.join(output_dir, first_dir, second_dir)
    genus = row["genus"]
    
    if not os.path.exists(patch_output_dir):
        
        outer_path = os.path.join(output_dir, first_dir)
        if not os.path.exists(outer_path):
            os.mkdir(outer_path)
        os.mkdir(patch_output_dir)
    
    input_path = os.path.join(us_patches_dir, first_dir, second_dir)
    
    rgb_image_path = os.path.join(input_path, observation_id + "_rgb.jpg")
    near_ir_image_path = os.path.join(input_path, observation_id + "_near_ir.jpg")
    landcover_image_path = os.path.join(input_path, observation_id + "_landcover.tif")
    altitude_image_path = os.path.join(input_path, observation_id + "_altitude.tif")
    
    # patches_zipped.extract(rgb_image_path, patch_output_dir)
    # patches_zipped.extract(near_ir_image_path, patch_output_dir)
    # patches_zipped.extract(landcover_image_path, patch_output_dir)
    # patches_zipped.extract(altitude_image_path, patch_output_dir)
    
    shutil.copy(rgb_image_path, patch_output_dir)
    shutil.copy(near_ir_image_path, patch_output_dir)
    shutil.copy(landcover_image_path, patch_output_dir)
    shutil.copy(altitude_image_path, patch_output_dir)
    
    counter += 1
    print("Copied patches for observation ", counter, "/", len(points))
        