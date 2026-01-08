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
    mask = h > 12
    az = np.where(mask, 360 - az, az)
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

    # create an array of hours from 0 to 23
    h = np.arange(0, 24, 1)
    # compute the elevation and azimuth for each hour. az and el are two vector of 24 elements
    az = np.zeros((24, len(lat)))
    el = np.zeros((24, len(lat)))
    # use a string comprehension to compute the azimuth and elevation for each hour. Unpack the two vectors

    for i in range(len(h)):
        az[i, :], el[i, :] = solarpos(h[i], doy, lat, lon)


    hrise = np.full(len(lat), None)
    hset = np.full(len(lat), None)

    for i in range(len(lat)):
        valid_hours = h[el[:, i] > 0]
        if valid_hours.size > 0:
            hrise[i] = np.min(valid_hours)
            hset[i] = np.max(valid_hours)

    return hrise, hset



def solarhours2D(lat, lon, doy):
    """
    Compute sunrise and sunset hours per grid cell.
    lat, lon: 2D arrays (ny, nx)
    doy: scalar
    Returns:
        hrise, hset: 2D arrays (ny, nx)
    """

    h = np.arange(24)
    nlat= lat.shape[0]

    az = np.zeros((24, nlat, nlat))
    el = np.zeros((24, nlat, nlat))

    for i, hour in enumerate(h):
        az[i, :, :], el[i, :, :] = solarpos(hour, doy, lat, lon)


    hrise = np.full(len(lat), None)
    hset = np.full(len(lat), None)

    for i in range(len(lat)):
        valid_hours = h[el[:,i,i] > 0]
        if valid_hours.size > 0:
            hrise[i] = np.min(valid_hours)
            hset[i] = np.max(valid_hours)


    return hrise, hset
