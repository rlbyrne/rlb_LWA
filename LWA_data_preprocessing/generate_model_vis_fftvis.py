# Adapted from Matthew Kolopanis

import sys
import numpy as np
from astropy import units
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.units import Quantity
import time
import pyradiosky
import pyuvdata
import matvis
import fftvis


def run_fftvis_diffuse_sim(map_path, beam_path, input_data_path, output_uvfits_path):

    # Read metadata from file
    uvd = pyuvdata.UVData.from_file(
        input_data_path,
        read_data=False,
        use_future_array_shapes=True,
        file_type="ms",
        ignore_single_chan=False,
    )
    if uvd.telescope_name == "OVRO_MMA":  # Correct telescope location
        uvd.telescope_name = "OVRO-LWA"
        uvd.set_telescope_params(overwrite=True, warn=True)
    uvd.set_uvws_from_antenna_positions(update_vis=False)
    uvd.phase_to_time(np.mean(uvd.time_array))  # Phase data
    uvd.flag_array[:, :, :] = False  # Unflag all
    # Define antenna locations
    antpos, ants = uvd.get_ENU_antpos()
    antpos = {a: pos for (a, pos) in zip(ants, antpos)}
    uvdata_antpos = {
        int(a): tuple(pos) for (a, pos) in zip(ants, uvd.antenna_positions)
    }
    antpairs = np.array([(a1, a2) for a1 in antpos.keys() for a2 in antpos.keys()])
    # Get observation time
    lat, lon, alt = uvd.telescope_location_lat_lon_alt_degrees
    location = EarthLocation.from_geodetic(lat=lat, lon=lon, height=alt)
    obstime = Time(np.mean(uvd.time_array), format="jd", scale="utc", location=location)
    lst = obstime.sidereal_time("apparent")

    # Get beam
    uvb = pyuvdata.UVBeam.from_file(beam_path)
    uvb.peak_normalize()
    if uvb.feed_array[0].lower() == "e":
        pol_mapping = {
            v: k
            for k, v in pyuvdata.utils._x_orientation_rep_dict(
                uvb.x_orientation
            ).items()
        }
        uvb.feed_array = np.array(
            [pol_mapping[feed.lower()] for feed in uvb.feed_array]
        )
    uvb.freq_interp_kind = "linear"

    # Get model
    model = pyradiosky.SkyModel.from_skyh5(map_path)
    model.at_frequencies(Quantity(uvd.freq_array, "Hz"))
    if model.component_type == "healpix":
        model.healpix_to_point()
        ra_new, dec_new = matvis.conversions.equatorial_to_eci_coords(
            model.ra.rad, model.dec.rad, obstime, location, unit="rad", frame="icrs"
        )
    else:
        ra_new = model.ra.rad
        dec_new = model.dec.rad
    use_model = model.at_frequencies(uvd.freq_array * units.Hz, inplace=False)

    # Run simulation
    sim_start_time = time.time()
    vis_full = fftvis.simulate.simulate_vis(
        ants=antpos,
        fluxes=use_model.stokes[0].T.to_value("Jy"),
        ra=ra_new,
        dec=dec_new,
        freqs=uvd.freq_arra,
        lsts=np.array([lst.to_value("rad")]),
        beam=uvb,
        polarized=True,
        precision=2,
        latitude=location.lat.rad,
    )
    sim_duration = time.time() - sim_start_time
    with open("/home/rbyrne/sim_timing.txt", "a") as f:
        f.write(
            f"\n FFTvis, linear beam interpolation: simulation timing {sim_duration/60.0} minutes"
        )

    uvd_out = pyuvdata.UVData.new(
        freq_array=uvd.freq_array,
        polarization_array=np.array([-5, -7, -8, -6]),
        antenna_positions=uvdata_antpos,
        telescope_location=location,
        telescope_name=uvd.telescope_name,
        times=np.array([obstime.jd]),
        antpairs=antpairs,
        data_array=vis_full.reshape(
            (uvd.Nfreqs, 4, len(antpos) * len(antpos))
        ).transpose([2, 0, 1]),
    )
    uvd_out.telescope_location = np.array(uvd_out.telescope_location)

    # Remove duplicate baselines
    uvd_out.conjugate_bls()
    baselines = list(zip(uvd_out.ant_1_array, uvd_out.ant_2_array))
    keep_baselines = []
    for bl_ind, baseline in enumerate(baselines):
        if baseline not in baselines[:bl_ind]:
            keep_baselines.append(bl_ind)
    uvd_out.select(blt_inds=keep_baselines)

    # Assign antenna names
    for ant_ind in range(len(uvd_out.antenna_names)):
        uvd_out.antenna_names[ant_ind] = uvd.antenna_names[
            np.where(uvd.antenna_numbers == uvd_out.antenna_numbers[ant_ind])[0][0]
        ]

    uvd_out.reorder_blts()
    uvd_out.reorder_pols(order="AIPS")
    uvd_out.reorder_freqs(channel_order="freq")
    uvd_out.phase_to_time(np.mean(uvd.time_array))
    uvd_out.check()

    # Save as uvfits
    uvd_out.write_uvfits(output_uvfits_path, fix_autos=True)

    # Save as ms
    uvd_out.reorder_pols(order="CASA")
    uvd_out.write_ms(f"{output_uvfits_path.removesuffix('.uvfits')}.ms", clobber=True)


if __name__ == "__main__":

    args = sys.argv
    map_path = args[1]
    beam_path = args[2]
    input_data_path = args[3]
    output_uvfits_path = args[4]

    run_fftvis_diffuse_sim(map_path, beam_path, input_data_path, output_uvfits_path)
