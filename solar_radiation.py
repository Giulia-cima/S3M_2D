import numpy as np

def solarpos(h, doy, lat, lon):
    """Compute azimuth and elevation as a function of hour of day, day of year, and latitude and longitude.
    Input: hour of day, day of year, latitude (vector), longitude (vector).
    Output: azimuth and elevation (2D arrays)."""

    doy = doy + h / 24
    gamma = 2 * np.pi * (doy - 1) / 365  # day angle [rad]
    delta = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) -
             0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma) -
             0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    dt1 = 1 - lon / 15
    h = (h + 12 - dt1) * 15
    h = np.mod(h, 360)
    sinalp = (np.sin(delta) * np.sin(lat * np.pi / 180) +
              np.cos(delta) * np.cos(lat * np.pi / 180) * np.cos(h * np.pi / 180))
    el = 180. / np.pi * np.arcsin(sinalp)
    cosaz = (np.cos(h * np.pi / 180) * np.cos(delta) * np.sin(lat * np.pi / 180) -
             np.sin(delta) * np.cos(lat * np.pi / 180)) / np.cos(el * np.pi / 180)
    az = 180 - (180 / np.pi * np.arccos(cosaz))
    az = np.where(h > 12, 360 - az, az)
    return az, el  # azimuth and elevation

def solar_radiation(h, doy, lat, lon):
    """Radiation at top of atmosphere as a function of hour of day, day of year, and latitude and longitude.
    Input: hour of day, day of year, latitude (vector), longitude (vector).
    Output: radiation at top of atmosphere (2D array)."""

    solar_constant = 1367  # W/m^2
    az, el = solarpos(h, doy, lat, lon)
    R = np.maximum(0, solar_constant * np.sin(el / 180 * np.pi))
    return R

def solarhours(lat, lon, doy):
    """Compute sunrise and sunset hours as a function of day of year, latitude, and longitude.
    Input: day of year, latitude (vector), longitude (vector).
    Output: sunrise and sunset hours (2D arrays)."""

    h = np.arange(24).reshape(-1, 1, 1) * np.ones((1, lat.size, lon.size))
    # Compute the elevation and azimuth for each hour
    az, el = solarpos(h, doy, lat, lon)
    # Find the hour of sunrise and sunset
    mask = el > 0  # Create a mask where elevation is positive
    # Find the hour of sunrise and sunset for each (lat, lon) point
    # if lat and lon are 1d vector
    if lat.ndim == 1:
        # Compute sunrise and sunset hours as 1D vectors
        hrise = np.min(np.where(mask, h, np.inf))
        hset = np.max(np.where(mask, h, -np.inf))
    else:
        hrise = np.min(np.where(mask, np.arange(24).reshape(-1, 1, 1), np.inf), axis=0)
        hset = np.max(np.where(mask, np.arange(24).reshape(-1, 1, 1), -np.inf), axis=0)  # Sunset hour
    return hrise, hset