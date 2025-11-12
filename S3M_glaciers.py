import numpy as np


def compute_glacier_delta_h(csn, csnlast, timestep, csndhmax, glacier, glaciere,
                            dem, elevbin, glacierdh, glacierdhdt, glacierdhn,
                            glacierdhdtmax, glacierthickmin, glaciermax):
    """
    Python equivalent of the Fortran S3M_Phys_Snow_Apps_GlacierDeltaH subroutine.

    :param csn: Current snow cover thickness (2D array)
    :param csnlast: Previous snow cover thickness (2D array)
    :param timestep: Time step in seconds
    :param csndhmax: Maximum allowed snow depth change
    :param glacier: Glacier mask (2D array, 1 where glacier is present, 0 otherwise)
    :param glaciere: Glacier elevation mask (2D array, 1 where glacier elevation changes, 0 otherwise)
    :param dem: Digital elevation model (2D array)
    :param elevbin: Elevation bins (1D array)
    :param glacierdh: Glacier thickness change (2D array)
    :param glacierdhdt: Glacier thickness change rate (2D array)
    :param glacierdhn: Temporary glacier thickness change (2D array)
    :param glacierdhdtmax: Maximum allowed glacier thickness change rate
    :param glacierthickmin: Minimum glacier thickness
    :param glaciermax: Maximum glacier thickness
    """
    glacierdhn[:, :] = csn - csnlast
    glacierdhn = np.clip(glacierdhn, -csndhmax, csndhmax)

    glacierdh[:, :] += glacierdhn
    glacierdhdt[:, :] = glacierdhn / (timestep / 86400.0)  # Convert to daily rate

    # Apply constraints on glacier thickness change rate
    glacierdhdt = np.clip(glacierdhdt, -glacierdhdtmax, glacierdhdtmax)

    # Glacier elevation change application
    mask = (glacier == 1) & (glaciere == 1)
    glacierdh[mask] = np.clip(glacierdh[mask], -glaciermax, glaciermax)
    glacierdh[mask] = np.maximum(glacierdh[mask], -dem[mask] + glacierthickmin)

    return glacierdh, glacierdhdt



