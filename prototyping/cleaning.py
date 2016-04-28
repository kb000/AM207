import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline

import seaborn as sns
sns.set_style("white")

import time
import timeit

import scipy.stats as stats
import pandas as pd
import pymc as pm

import re
import numpy as np


# Set a flag on whether to load the whole data set, or just a portion.
load_all = False
nb_root = "../../../final/"
# Read ridership data
rides_path = "data/hubway_2011_07_through_2013_11/%shubway_trips.csv"
raw_rides = pd.read_csv(nb_root + rides_path % "") if load_all else pd.read_csv(nb_root + rides_path % "fewer_")
# Read weather data
weather_na = ("unknown", "9999", "-9999")
raw_weather = pd.read_csv(nb_root + "data/ncdc-2013.csv", na_values=weather_na)


do_rebuild = False
ride_pickle_name = 'rides_pickle'
try:
    if do_rebuild: raise 'abort.';
    rides = pd.read_pickle(ride_pickle_name)
except:
    import dateutil.parser as dtp
    rides = raw_rides.copy()
    totalseconds = lambda t: (t.hour*60 + t.minute)*60 + t.second
    for endpoint in ('start', 'end'):
        datetimes = [dtp.parse(d) for d in rides[endpoint + '_date']]
        rides.loc[:,endpoint + '_datetime'] = pd.Series(datetimes, index = rides.index)
        rides.loc[:,endpoint + '_weekday'] = pd.Series([d.weekday() for d in datetimes], index = rides.index)
        rides.loc[:,endpoint + '_seconds'] = pd.Series([totalseconds(d.time()) for d in datetimes], index = rides.index)

    rides.to_pickle(ride_pickle_name)


