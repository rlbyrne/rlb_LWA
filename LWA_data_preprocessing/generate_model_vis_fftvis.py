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


def run_fftvis_diffuse_sim(
    map_path=None,
    beam_path=None,
    input_data_path=None,
    output_uvfits_path=None,
    log_path=None,
):

    if log_path is None:
        log_path = f"{output_uvfits_path.removesuffix('.uvfits')}_log.txt"

    f = open(log_path, "w")
    f.write("Starting fftvis simulation.\n")
    f.write(f"Simulation skymodel: {map_path}\n")
    f.write(f"Simulation beam model: {beam_path}\n")
    f.write(f"Simulation input datafile: {input_data_path}\n")
    f.close()

    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    sys.stdout = sys.stderr = log_file_new = open(log_path, "a")

    # Read metadata from file
    print("Reading data...")
    sys.stdout.flush()
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
    antpairs = [(a1, a2) for ant1_ind, a1 in enumerate(ants) for a2 in ants[ant1_ind:]]

    # Get observation time
    lat, lon, alt = uvd.telescope_location_lat_lon_alt_degrees
    location = EarthLocation.from_geodetic(lat=lat, lon=lon, height=alt)
    obstimes = Time(
        sorted(list(set(uvd.time_array))), format="jd", scale="utc", location=location
    )
    lsts = obstimes.sidereal_time("apparent")

    # Get beam
    print("Reading the beam model...")
    sys.stdout.flush()
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
    print("Reading the sky model...")
    sys.stdout.flush()
    model = pyradiosky.SkyModel.from_skyh5(map_path)
    model.at_frequencies(Quantity(uvd.freq_array, "Hz"))
    if model.component_type == "healpix":
        model.healpix_to_point()
    # Perform a coordinate transformation to account for time-dependent precession and nutation
    ra_new, dec_new = matvis.conversions.equatorial_to_eci_coords(
        model.ra.rad,
        model.dec.rad,
        np.mean(obstimes),
        location,
        unit="rad",
        frame="icrs",
    )

    # Run simulation
    print("Starting the simulation...")
    sys.stdout.flush()
    sim_start_time = time.time()
    vis_full = fftvis.simulate.simulate_vis(
        ants=antpos,
        baselines=antpairs,
        fluxes=model.stokes[0].T.to_value("Jy"),
        ra=ra_new,
        dec=dec_new,
        freqs=uvd.freq_array,
        lsts=np.array(lsts.to_value("rad")),
        beam=uvb,
        polarized=True,
        precision=2,
        latitude=location.lat.rad,
    )
    sim_duration = time.time() - sim_start_time
    print(f"Simulation completed. Timing {sim_duration/60.0} minutes.")
    print("Formatting simulation output...")
    sys.stdout.flush()
    formatting_start_time = time.time()

    uvd_out = pyuvdata.UVData.new(
        freq_array=uvd.freq_array,
        polarization_array=[-5, -7, -8, -6],
        antenna_positions=uvdata_antpos,
        telescope_location=location,
        telescope_name=uvd.telescope_name,
        times=np.array(obstimes.jd),
        antpairs=np.array(antpairs),
        time_axis_faster_than_bls=True,
        data_array=vis_full.transpose([4, 1, 0, 2, 3]).reshape(
            (len(antpairs) * uvd.Ntimes, uvd.Nfreqs, uvd.Npols)
        ),
    )
    uvd_out.telescope_location = np.array(uvd_out.telescope_location)

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
    print(
        f"Formatting completed. Timing {(time.time()-formatting_start_time)/60.0} minutes."
    )
    sys.stdout.flush()

    # Save as uvfits
    print(f"Saving simulation output to {output_uvfits_path}")
    sys.stdout.flush()
    try:
        uvd_out.write_uvfits(output_uvfits_path, fix_autos=True)
    finally:
        # Save as ms
        ms_path = f"{output_uvfits_path.removesuffix('.uvfits')}.ms"
        print(f"Saving simulation output to {ms_path}")
        sys.stdout.flush()
        uvd_out.reorder_pols(order="CASA")
        uvd_out.write_ms(ms_path, clobber=True)

    sys.stdout = stdout_orig
    sys.stderr = stderr_orig
    log_file_new.close()


if __name__ == "__main__":

    args = sys.argv
    map_path = args[1]
    beam_path = args[2]
    input_data_path = args[3]
    output_uvfits_path = args[4]

    run_fftvis_diffuse_sim(map_path, beam_path, input_data_path, output_uvfits_path)
