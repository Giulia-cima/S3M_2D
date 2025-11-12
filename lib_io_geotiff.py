import os
import numpy as np
import rioxarray as rxr
import xarray as xr

class GDAL_ImportError(Exception):
    """Custom exception for GDAL import errors."""
    def __init__(self, function_name):
        super().__init__(f"GDAL import error in function: {function_name}")


def read_geotiff(filename: str, out='xarray', stack=True):
    if isinstance(filename, list):
        data_list = [read_geotiff(file, out) for file in filename]
        if stack:
            if out == 'xarray':
                data = xr.concat(data_list, dim='band')
            elif out == 'array':
                data = np.stack(data_list)
            elif out == 'gdal':
                data = data_list
                raise Warning('Cannot stack GDAL datasets')
        else:
            data = data_list

        return data

    if out == 'xarray':
        data = rxr.open_rasterio(filename)

        # ensure that the data has descending latitudes
        y_dim = data.rio.y_dim
        if y_dim is None:
            for dim in data.dims:
                if 'lat' in dim.lower() | 'y' in dim.lower():
                    y_dim = dim
                    break
        if data[y_dim][0] < data[y_dim][-1]:
            data = data.sortby(y_dim, ascending=False)

    elif out == 'gdal':
        try:
            from osgeo import gdal, gdalconst
        except ImportError:
            raise GDAL_ImportError('read_geotiff')
        data = gdal.Open(filename, gdalconst.GA_ReadOnly)

    elif out == 'array':
        da = read_geotiff(filename, 'xarray')
        data = da.values.squeeze()

    else:
        raise ValueError(f'out must be one of ["xarray", "gdal", "array"], got {out}')

    return data


def read_geotiff_asGDAL(filename):
    return read_geotiff(filename, out='gdal')


def read_geotiff_asXarray(filename):
    return read_geotiff(filename, out='xarray')


def read_geotiff_as_array(filename):
    return read_geotiff(filename, out='array')


def read_multiple_geotiffs_asXarray(tiff_file_paths):
    return read_geotiff(tiff_file_paths, out='xarray', stack=True)

""" 
def write_geotiff(data: xr.DataArray | np.ndarray, filename, **kwargs):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if 'metadata' in kwargs:
        metadata = kwargs['metadata']
    else:
        metadata = None

    if 'nodata_value' in kwargs:
        nodata_value = kwargs['nodata_value']
    else:
        nodata_value = None

    if isinstance(data, xr.DataArray):
        if metadata is not None:
            data.attrs = metadata
        if nodata_value is not None:
            data = data.rio.write_nodata(nodata_value)
        data.rio.to_raster(filename, compress='LZW')

    elif isinstance(data, np.ndarray):
        if 'template' not in kwargs:
            raise ValueError('template must be provided when writing a numpy array to a geotiff')
        template = kwargs['template']
        template = read_geotiff(template)
        template = template.squeeze()
        # check if the data has the same data type as the template
        if data.dtype != template.dtype:
            template.values = np.zeros_like(template.values, dtype=data.dtype)
        data = np.squeeze(data)
        data_array = template.copy(data=data)
        write_geotiff(data_array, filename, **kwargs)

    else:
        try:
            from osgeo import gdal
        except ImportError:
            raise GDAL_ImportError('write_geotiff')
        if isinstance(data, gdal.Dataset):
            if metadata is not None:
                data.SetMetadata(metadata)
            if nodata_value is not None:
                data.GetRasterBand(1).SetNoDataValue(nodata_value)
            gdal.Translate(filename, data, creationOptions=['COMPRESS=LZW'])


def write_geotiff_fromXarray(data, filename):
    write_geotiff(data, filename)


def write_geotiff_fromGDAL(ds, filename):
    write_geotiff(ds, filename)


def write_geotiff_singleband(filename, data, template, metadata=None, nodata_value=None):
    write_geotiff(data, filename, template=template, metadata=metadata, nodata_value=nodata_value)
"""