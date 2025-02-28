import datetime

import pandas
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u

def main():
    mlc = {
        'lat': 32.902 * u.deg,
        'long': -105.530 * u.deg,
        'elev': 2225 * u.m
    }
    ctio = {
        'lat': -30.168 * u.deg,
        'long': -70.805 * u.deg,
        'elev': 2286 * u.m
    }
    mo = {
        'lat': -31.638 * u.deg,
        'long': 116.989 * u.deg,
        'elev': 197 * u.m
    }
    oauj = {
        'lat': 50.054 * u.deg,
        'long': 19.828 * u.deg,
        'elev': 318 * u.m
    }

    # Define Telescope Coordinates
    tele = mlc

    telescope_lat = tele['lat']  # positive for North
    telescope_lon = tele['long']  # positive for East
    telescope_elevation =tele['elev']  # meters

    telescope_location = EarthLocation(
        lat=telescope_lat,
        lon=telescope_lon,
        height=telescope_elevation
    )

    # Define Target Coordinates (RA, Dec)
    gl = pandas.read_csv(r"C:\Users\Dylan\Downloads\S250206dm_7_galaxies_sorted.csv")

    target_coords = {}
    for g in gl.itertuples():
        name1 = ''.join(char for char in g.objname if char.isalnum())
        target_coords[g.objname] = SkyCoord(ra=g.ra*u.deg, dec=g.dec*u.deg, frame='icrs')

    # Observation Time (UTC)
    now = datetime.datetime.now(datetime.UTC)
    observation_time = Time(now)

    # Convert RA/Dec to Alt/Az
    for name, target_coord in target_coords.items():
        altaz_frame = AltAz(obstime=observation_time, location=telescope_location)
        altaz = target_coord.transform_to(altaz_frame)

        # 5. Check if the target is above the horizon
        if altaz.alt.deg > 20.0:
            ra_target = round(float(target_coord.ra.hour), 6)
            dec_target = round(float(target_coord.dec.deg), 6)

            print(f"{name}, ra={ra_target} hr, dec={dec_target} deg")


if __name__ == '__main__':
    main()
