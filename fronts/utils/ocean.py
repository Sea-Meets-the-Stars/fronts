""" Oceanography utils """


import numpy as np

def coriolis(lat:float, lon:float):
    """ Coriolis parameter at lat, lon

    Args:
        lat (float): Latitude in degrees
        lon (float): Longitude in degrees

    Returns:
        float: Coriolis parameter in 1/s
    """

    # Convert to radians
    lat_rad = np.deg2rad(lat)

    # Coriolis parameter
    omega = 7.2921e-5  # Earth's rotation rate in rad/s
    f = 2 * omega * np.sin(lat_rad)

    return f