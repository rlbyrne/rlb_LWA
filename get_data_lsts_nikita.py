import re
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u

def parse_filename(filename):
    """
    Parses the filename to extract the date and UTC time.

    Expected filename format: YYYYMMDD_HHMMSS_*.ms
    Example: 20240302_103004_73MHz.ms
    """
    pattern = r'(\d{8})_(\d{6})_.*\.ms'
    match = re.match(pattern, filename)
    if not match:
        raise ValueError("Filename does not match the expected format 'YYYYMMDD_HHMMSS_*.ms'")
    
    date_str, time_str = match.groups()
    # Format the date and time into ISO format
    iso_time = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
    return iso_time

def calculate_lst(filename, x, y, z):
    """
    Calculates the Local Sidereal Time (LST) for the observatory.

    Parameters:
    - filename: str, the name of the file containing date and UTC time.
    - x, y, z: float, ECEF coordinates in meters.

    Returns:
    - lst_hour: float, LST in decimal hours.
    """
    # Parse the filename to get ISO formatted time
    utc_time_iso = parse_filename(filename)
    print(f"Parsed UTC Time: {utc_time_iso}")
    
    # Create an Astropy Time object
    time = Time(utc_time_iso, format='isot', scale='utc')
    
    # Define the observatory's ECEF coordinates
    observatory = EarthLocation.from_geocentric(x * u.m, y * u.m, z * u.m)
    
    # Convert to geodetic coordinates to get longitude
    geodetic = observatory.to_geodetic()
    longitude = geodetic.lon
    
    # Calculate Local Sidereal Time (LST)
    lst = time.sidereal_time('mean', longitude=longitude)
    lst_hour = lst.hour  # Convert to decimal hours
    
    print(f"Local Sidereal Time (LST): {lst.to_string(unit=u.hour, sep=':')}")
    return lst_hour

if __name__ == "__main__":
    # Example usage
    filename = "20240601_050959_82MHz.ms"
    
    # Observatory ECEF coordinates in meters
    x = -2409261.7339418
    y = -4477916.56772157
    z = 3839351.13864434
    
    # Calculate LST
    lst = calculate_lst(filename, x, y, z)
    print(lst)

        
