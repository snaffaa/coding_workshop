import pandas as pd 
import csv 
import os
import datetime 
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy
import netCDF4
import csv
#import geopandas as gpd
#import shapely.geometry as gmt
#from numpy import arange, dtype

'''
The first part to calculate Kling Gupta effeciency is to calculate the pearson coeffecient for observations and estimation 
then substract it from 1 and get the square of the results. The formula used to calculate pearson coeffecient is:
r = sum((x-x)* (y-y))/ square_root(sum(x-x')**2) * square_root(sum (y-y)**2). where x, y are observations mean 
and estimation mean. 

'''
inputdir = '/home/naffa002/projects/shoekrancvsfiles/'
outputdir = '/scratch/safaa/output-validation/5arcmin_run/sed_delivery/new_lakes_data/kge_per_station/'
mapdir = '/home/naffa002/projects/shoekranmaps/'


dateparse = lambda x: pd.datetime.strptime(x,'%Y-%m')
#~ sim_obs= pd.read_csv(os.path.join(inputdir,'obs_sim_values_all_stations_withoutlakes.csv'),\
                     #~ parse_dates=[0], date_parser=dateparse, delimiter=";", decimal=",")
sim_obs= pd.read_csv(os.path.join(inputdir,'obs_est_hybam_sed_flux_new_lakes_data.csv'),\
                     parse_dates=[1], date_parser=dateparse, delimiter=",")

print(sim_obs)
#~ sim_obs= pd.read_csv(os.path.join(inputdir,'obs_sim_all_stations_values_sat_30years.csv'),\
                     #~ parse_dates=[0], date_parser=dateparse, delimiter=";", decimal=",")
#sim_obs= pd.read_csv(os.path.join(inputdir,'obs_sim_values_all_stations.csv'),\
                     #parse_dates=[0], date_parser=dateparse, delimiter=";", decimal=",")
sim_obs.rename(columns = {'month-year':'Date' }, inplace = True)
print(sim_obs.head())
sim_obs["year"] = pd.DatetimeIndex(sim_obs['Date']).year
#print("years in the results are", sim_obs["year"])
sim_obs["month"] = pd.DatetimeIndex(sim_obs['Date']).month
#print("months in the results are", sim_obs["month"])
print(sim_obs.head())

cols=[i for i in sim_obs.columns if i not in ['Date','month', 'year']]
print(cols)

# Our dataframe has for each station a column with observations and a column
# with estimations. These columns have the same name. We split the observation
# and estimation columns into different dataframes so we can address the data
# in a simpler way.
data_obs={}
data_est={}
stationnames=[]
for col in cols:
    mycol = pd.to_numeric(sim_obs[col]).replace(0, float('NaN'))
    # delete the last 2 charecters at the end of the estimation's names .1
    if (col.endswith("_y")):
        data_est[col] = mycol
        
    else:
        data_obs[col] = mycol
        stationnames.append(col)
    #fi
#rof
obs = pd.DataFrame.from_dict(data_obs)
obs.drop(obs.filter(regex="Unnamed: 0"),axis=1, inplace=True)
est = pd.DataFrame.from_dict(data_est)

print(obs.head())
print(est.head())
# to set the Date as index 
dateindex=sim_obs["Date"]
obs.set_axis(dateindex, axis='index', inplace=True)
est.set_axis(dateindex, axis='index', inplace=True)

print("Observations: ", obs.head(5))
print("Estimations: ", est.head(5))

# calculate the sum of every column and add put the ones for 
# estimations in a dataframe and the observatioons in another
mean_obs=[]
mean_est=[]
for name in stationnames:
    obsmean = obs[name].mean()
    estmean = est[name].mean()
    mean_obs.append(obsmean)
    mean_est.append(estmean)
#rof
obs_mean=pd.DataFrame(mean_obs, index=stationnames)
obs_mean.shape
obs_mean.index.name= 'stations'
print('the columns of observation means dataframe are', list(obs_mean.columns))
#obs_mean.rename(columns= {" ", "means"}, inplace= True)
#~ obs_mean.plot()
#~ plt.show()
est_mean=pd.DataFrame(mean_est, index=stationnames)
est_mean.index.name= 'stations'
#est_mean.columns.values[0] = 'means'
print("Observation mean: ", obs_mean.head(20))
print("Estimation mean: ", est_mean.head(20))

# determine distance from the mean for all values for each stations (x - x')
data_obs_obsmean={}
data_est_estmean={}
for name in stationnames:
    data_obs_obsmean[name] = (obs[name] - obs_mean.loc[name, 0])
    data_est_estmean[name] = (est[name] - est_mean.loc[name, 0])
#rof
#print('distance from mean observation for ', stationnames[0], data_obs_obsmean[stationnames[0]])
obs_minus_obsmean= pd.DataFrame(data_obs_obsmean)
print('the variation between the observation and their mean for each station is',obs_minus_obsmean.head(5))

est_minus_estmean= pd.DataFrame(data_est_estmean)
print('the variation between the estimations and their mean for each station is',est_minus_estmean.head(5))

# the distance from mean for obs values multiplied by the distnce from mean for all estimations values at all stations
#(x - x') * (y - y')
data_distance= {}
for name in stationnames:
    #print('the observations minus observation mean is', obs_minus_obsmean[name])
    data_distance[name]= (obs_minus_obsmean[name] * est_minus_estmean[name])
#rof
distance_calc= pd.DataFrame(data_distance)
print('the variation of observations multiplied with the one of estimations', distance_calc.head(15))

sums=[]
for name in stationnames:
    sums.append(data_distance[name].sum())
#rof
dfsums = pd.DataFrame(sums, index = stationnames)
print('the sums of the variation products: ', dfsums.head(15))

# determine the square of observation distance and estimation from their mean at all stations  
data_obs_obsmean_sq={}
data_est_estmean_sq={}
for name in stationnames:
    data_obs_obsmean_sq[name] = (obs[name] - obs_mean.loc[name, 0]) **2
    data_est_estmean_sq[name]= (est[name] - est_mean.loc[name, 0])**2
#rof
obs_minus_obsmean_sq= pd.DataFrame(data_obs_obsmean_sq)
obs_minus_obsmean_sq_sum= obs_minus_obsmean_sq.sum()
est_minus_estmean_sq= pd.DataFrame(data_est_estmean_sq)
est_minus_estmean_sq_sum = est_minus_estmean_sq.sum()
print('the difference observation',obs_minus_obsmean_sq.head(5))
print('the sum is of observations is', obs_minus_obsmean_sq_sum.head(5))
print('the difference estimations',est_minus_estmean_sq.head(5))
print('the sum is of the estimations is', est_minus_estmean_sq_sum)
 
# determine the square root of the observation and estimation distance sequared 
data_square_root_obs = []
data_square_root_est = []
data_square_root_obs_est = []
for name in stationnames:
    sqrobs = np.sqrt(obs_minus_obsmean_sq_sum[name])
    sqrest = np.sqrt(est_minus_estmean_sq_sum[name])
    data_square_root_obs.append(sqrobs)
    data_square_root_est.append(sqrest)
    data_square_root_obs_est.append(sqrobs * sqrest)
#rof
obs_square_root = pd.DataFrame(data_square_root_obs, index = stationnames)
est_square_root = pd.DataFrame(data_square_root_est, index = stationnames)
obs_est_sq_root = pd.DataFrame(data_square_root_obs_est, index = stationnames)
print('the square root', obs_est_sq_root.head())
    
data_division = []
data_division_r= []
for name in stationnames:
    data_division_r.append(dfsums.loc[name, 0] / obs_est_sq_root.loc[name, 0])
    data_division.append(((dfsums.loc[name, 0] / obs_est_sq_root.loc[name, 0]) - 1) ** 2)
#rof
pearson_coeffecient = pd.DataFrame(data_division_r, index=stationnames)
pearson_coeffecient_kge = pd.DataFrame(data_division, index=stationnames)
print('the pearson coeffecient for every station is', pearson_coeffecient_kge.head(5))
print('the pearson coeffecient for every station is', pearson_coeffecient.head(5))
pearson_coeffecient_kge.to_csv(os.path.join(outputdir, 'pearson_coeffecient.csv'))



'''
The second part of kling gupta effeciency formula is to calculate the standard deviation for observations
and the standard deviation of estimations and then devide them and substract 1 for the result((sd_est/sd_ob)-1)**2.
To calculate the standard deviation we used the following formula:  
 sd= square_root(1/n *(sum(x - x')**2)
'''
data_obs_sd = []
data_est_sd = []
for name in stationnames:
    data_obs_sd.append(np.sqrt(obs_minus_obsmean_sq[name].mean()))
    data_est_sd.append(np.sqrt(est_minus_estmean_sq[name].mean()))
#rof
standard_deviation_obs = pd.DataFrame(data_obs_sd, index = stationnames)
standard_deviation_est = pd.DataFrame(data_est_sd, index = stationnames)

print('the standard deviation of observations is ', standard_deviation_obs.head())
print('the standard deviation of estimations is ', standard_deviation_est.head())

data_sd = []
for name in stationnames:
     data_sd.append(((standard_deviation_est.loc[name, 0] / standard_deviation_obs.loc[name, 0]) - 1) ** 2)
#rof
sd_est_obs= pd.DataFrame(data_sd, index = stationnames)
print ('the standard deviation component for kGE is',sd_est_obs.head())

'''
The third part of KGE eequation is to divide the average of estimations to the average of observations 
then substract 1 from the divided values and calculte the square of the results ((est_mean/obs_mean)-1)**2 
'''
data_mean = []
for name in stationnames:
    data_mean.append(((est_mean.loc[name, 0] / obs_mean.loc[name, 0]) - 1) ** 2)
#rof
mean_obs_est= pd.DataFrame(data_mean, index = stationnames)
print('results of divided obs and est means are', mean_obs_est.head())

data_kge = []
for name in stationnames: 
    data_kge.append(1 - (np.sqrt(pearson_coeffecient_kge.loc[name, 0] + sd_est_obs.loc[name, 0] + mean_obs_est.loc[name, 0])))
#rof
kling_Gupta_coeffecient= pd.DataFrame(data_kge, index = stationnames)
kling_Gupta_coeffecient.index.name = "station_name"
print("kling_Gupta_coeffecient", kling_Gupta_coeffecient.head(20))
#kling_Gupta_coeffecient.to_csv(os.path.join(outputdir, 'KGE_all stations.csv'))


station_location = pd.read_csv(os.path.join(inputdir,'station_info_5arcmin_new_stations_selected.csv'))
#station_location = pd.read_csv(os.path.join(inputdir,'station_info_30arcmin_new_stations_updated.csv'))
# delete unnamed column froom the dataframe
station_location = station_location.loc[:, ~station_location.columns.str.contains('^Unnamed')]

print(station_location.head())
print(station_location.shape)

# To make a netcdf file from the KGE results and its components for each station, 
# we convert the csv file to netcdf file with three variable names longitude, latitude and the component
# including Kling gupta effeciency (KGE), standard deviation (sd) ,pearson coeffecient (r)

# in this part, the Nash Sutcliffe model Efficiency coefficient is calculated. The formula used for that is 
# Nash= 1-sum(obs-sim)**2/ sum(obs-obs')**2 where obs'is the observation mean
data_nash_components= {}
data_sum_obs_est_square=[]

for name in stationnames:
    data_nash_components[name]=((obs[name] -est[name])**2)
    data_sum_obs_est_square.append(data_nash_components[name].sum())
#rof
nasch_first_component = pd.DataFrame(data_nash_components)
nash_first_part = pd.DataFrame(data_sum_obs_est_square, index= stationnames)

print('the first component of nash calculation is',nasch_first_component)
print('the first part of nash computation is', nash_first_part)

data_nash= []
for name in stationnames:
    print('the observation minus observation mean is',obs_minus_obsmean_sq_sum.head())
    data_nash.append(1-(nash_first_part.loc[name,0] / obs_minus_obsmean_sq_sum[name]))
#rof 
print('the nash data are',data_nash)
nash_model_efficiency = pd.DataFrame(data_nash, index = stationnames)
print('the nash_model_efficiency is', nash_model_efficiency) 

finalresults = station_location.copy()
finalresults = pd.merge(station_location, kling_Gupta_coeffecient, on=['station_name'], how='left')
#finalresults.to_csv(os.path.join(outputdir, 'KGE_location_all stations.csv'))
finalresults.to_csv(os.path.join(outputdir, 'KGE_location_all stations-30years_5arcminuts.csv'))
print('the final results are', finalresults)
#finalresults.to_csv(os.path.join(outputdir, 'KGE_location_all stations-30years_sat.csv'))
#make points for the longitude and latitude of the stations locations 

#station_coords = [gmt.Point(row.longitude, row.latitude) for i, row in finalresults.iterrows()]
#finalresults_gdf = gpd.GeoDataFrame(finalresults, geometry = station_coords)
#finalresults_gdf.columns = ['station','lon','lat','kge','geometry']
# keep only the rows that has kge value
#finalresults_gdf= finalresults_gdf[finalresults_gdf.kge.notnull()]
#print("final results", finalresults_gdf.head(30))

#finalresults_gdf.to_file(os.path.join(mapdir, 'KGE_location_all stations_30years_5arcmin.shp'))
