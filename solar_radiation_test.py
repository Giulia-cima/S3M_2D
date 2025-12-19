import numpy as np


def solarpos(h, doy, lat, lon):

    """compute azimuth and elevation as a function of hour of day, day of year, and latitude and longitude
    input: hour of day, day of year, latitude Nord, longitude Est
    output: azimuth and elevation"""

    doy = doy + h / 24
    gamma = 2 * np.pi * (doy - 1) / 365  # day angle [rad]
    delta = 0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(
        2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    dt1 = 1 - lon / 15
    h = (h + 12 - dt1) * 15
    h = np.mod(h, 360)
    sinalp = (np.sin(delta) * np.sin(lat * np.pi / 180) + np.cos(delta) * np.cos(lat * np.pi / 180) *
              np.cos(h * np.pi / 180))
    el = 180. / np.pi * np.arcsin(sinalp)
    cosaz = (np.cos(h * np.pi / 180) * np.cos(delta) * np.sin(lat * np.pi / 180) - np.sin(delta) *
             np.cos(lat * np.pi / 180)) / np.cos(el * np.pi / 180)
    az = 180 - (180/ np.pi * np.arccos(cosaz))
    if h > 12:
        az = 360 - az

    return az, el  # azimuth and elevation


def solar_radiation(h, doy, lat, lon):

    """radiation at top of atmosphere as a function of hour of day, day of year, and latitude and longitude
    input: hour of day, day of year, latitude, longitude
    output: radiation at top of atmosphere"""

    solar_constant = 1367  # W/m^2
    az, el = solarpos(h, doy, lat, lon)
    R = np.max([0, solar_constant * np.sin(el / 180 * np.pi)])
    return R

def solarhours(lat, lon, doy):
    """compute sunrise and sunset hours as a function of day of year, latitude and longitude
    input: day of year, latitude, longitude
    output: sunrise and sunset hours hrise and hset"""
    # create an array of hours from 0 to 23
    h = np.arange(0, 24, 1)
    # compute the elevation and azimuth for each hour. az and el are two vector of 24 elements
    az = np.zeros(24)
    el = np.zeros(24)
    # use a string comprehension to compute the azimuth and elevation for each hour. Unpack the two vectors
    for i, j in enumerate(h):
        az[i], el[i] = solarpos(j, doy, lat, lon)
# find the hour of sunrise and sunset
    hrise = np.min(h[el > 0])
    hset = np.max(h[el > 0])
    return hrise, hset

