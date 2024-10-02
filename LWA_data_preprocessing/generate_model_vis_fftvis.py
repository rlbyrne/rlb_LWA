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
import socket


def run_fftvis_diffuse_sim(
    map_path=None,
    beam_path=None,
    input_data_path=None,
    output_uvfits_path=None,
    log_path=None,
    offset_timesteps=0,
):

    if log_path is None:
        log_path = f"{output_uvfits_path.removesuffix('.uvfits')}_log.txt"

    with open(log_path, "w") as f:
        f.write("Starting fftvis simulation.\n")
        f.write(f"Running on {socket.gethostname()}")
        f.write(f"Simulation skymodel: {map_path}\n")
        f.write(f"Simulation beam model: {beam_path}\n")
        f.write(f"Simulation input datafile: {input_data_path}\n")

    # stdout_orig = sys.stdout
    # stderr_orig = sys.stderr
    # sys.stdout = sys.stderr = log_file_new = open(log_path, "a")

    # Read metadata from file
    with open(log_path, "a") as f:
        f.write("Reading data...\n")

    if input_data_path.endswith(
        ".ms"
    ):  # Data reading doesn't automatically detect file type
        file_type = "ms"
    else:
        file_type = None
    uvd = pyuvdata.UVData.from_file(
        input_data_path,
        read_data=False,
        use_future_array_shapes=True,
        file_type=file_type,
        ignore_single_chan=False,
    )
    if uvd.telescope_name == "OVRO_MMA":  # Correct telescope location
        uvd.telescope_name = "OVRO-LWA"
        uvd.set_telescope_params(overwrite=True, warn=True)
    if uvd.telescope_name == "HERA":
        uvd.compress_by_redundancy()

    if offset_timesteps:
        uvd.time_array += np.max(uvd.time_array) - np.min(uvd.time_array) + np.mean(uvd.integration_time) / (60.0*60.0*24.0)
        uvd.set_lsts_from_time_array()

    uvd.set_uvws_from_antenna_positions(update_vis=False)
    uvd.phase_to_time(np.mean(uvd.time_array))  # Phase data
    uvd.flag_array = np.zeros(
        (uvd.Nblts, uvd.Nfreqs, uvd.Npols), dtype=bool
    )  # Unflag all

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
    with open(log_path, "a") as f:
        f.write("Reading the beam model...\n")
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
    if np.max(Quantity(uvd.freq_array, "Hz")) > np.max(Quantity(uvb.freq_array, "Hz")):
        print(
            "WARNING: Max data frequency exceeds max beam model frequency. Using nearest neighbor value."
        )
        uvb_max_freq = uvb.select(frequencies=np.max(uvb.freq_array), inplace=False)
        uvb.freq_array = np.append(uvb.freq_array, np.max(uvd.freq_array))
        uvb.Nfreqs += 1
        uvb.data_array = np.append(
            uvb.data_array,
            uvb_max_freq.data_array,
            axis=2,
        )
        uvb.bandpass_array = np.append(uvb.bandpass_array, uvb_max_freq.bandpass_array)
    if np.min(Quantity(uvd.freq_array, "Hz")) < np.min(Quantity(uvb.freq_array, "Hz")):
        print(
            "WARNING: Minimum data frequency is less than minimum beam model frequency. Using nearest neighbor value."
        )
        uvb_min_freq = uvb.select(frequencies=np.min(uvb.freq_array), inplace=False)
        uvb.freq_array = np.append(np.min(uvd.freq_array), uvb.freq_array)
        uvb.Nfreqs += 1
        uvb.data_array = np.append(
            uvb_min_freq.data_array,
            uvb.data_array,
            axis=2,
        )
        uvb.bandpass_array = np.append(uvb_min_freq.bandpass_array, uvb.bandpass_array)
    uvb.freq_interp_kind = "linear"  # Added for simulation speedup

    # Get model
    with open(log_path, "a") as f:
        f.write("Reading the sky model...\n")
    if map_path.endswith(".skyh5"):
        model = pyradiosky.SkyModel.from_skyh5(map_path)
    elif map_path.endswith(".sav"):
        model = pyradiosky.SkyModel.from_fhd_catalog(map_path, expand_extended=True)
    else:
        model = pyradiosky.SkyModel()
        model.read(map_path)
    use_model_freq_array = uvd.freq_array
    if (
        model.spectral_type == "subband"
    ):  # Define frequency extrapolation to use nearest neighbor values
        if np.max(Quantity(uvd.freq_array, "Hz")) > np.max(
            Quantity(model.freq_array, "Hz")
        ):
            print(
                "WARNING: Max data frequency exceeds max sky model frequency. Using nearest neighbor value."
            )
            use_model_freq_array[
                np.where(
                    Quantity(use_model_freq_array, "Hz")
                    > np.max(Quantity(model.freq_array, "Hz"))
                )
            ] = np.max(model.freq_array)
        if np.min(Quantity(uvd.freq_array, "Hz")) < np.min(
            Quantity(model.freq_array, "Hz")
        ):
            print(
                "WARNING: Minimum data frequency is less than minimum sky model frequency. Using nearest neighbor value."
            )
            use_model_freq_array[
                np.where(
                    Quantity(use_model_freq_array, "Hz")
                    < np.min(Quantity(model.freq_array, "Hz"))
                )
            ] = np.min(model.freq_array)
    model.at_frequencies(Quantity(use_model_freq_array, "Hz"))
    if model.component_type == "healpix":
        model.healpix_to_point()
    use_comp_inds = np.where(np.isfinite(np.sum(model.stokes, axis=(0, 1))))[0]
    if len(use_comp_inds) < model.Ncomponents:  # Remove nan-ed sources
        print("Removing nan flux values from input catalog.")
        model.select(component_inds=use_comp_inds)
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
    with open(log_path, "a") as f:
        f.write("Starting the simulation...\n")
    sim_start_time = time.time()

    vis_full = np.zeros((uvd.Nfreqs, uvd.Ntimes, 2, 2, len(antpairs)), dtype=complex)
    for freq_ind in range(uvd.Nfreqs):
        with open(log_path, "a") as f:
            f.write(f"Simulating frequency {freq_ind + 1}/{uvd.Nfreqs}.\n")
        vis_full[freq_ind, :, :, :, :] = fftvis.simulate.simulate_vis(
            ants=antpos,
            baselines=antpairs,
            fluxes=model.stokes[0].T.to_value("Jy"),
            ra=ra_new,
            dec=dec_new,
            freqs=uvd.freq_array[[freq_ind]],
            lsts=np.array(lsts.to_value("rad")),
            beam=uvb,
            polarized=True,
            precision=1,  # Worse precision
            latitude=location.lat.rad,
        )
    sim_duration = time.time() - sim_start_time
    with open(log_path, "a") as f:
        f.write(f"Simulation completed. Timing {sim_duration/60.0} minutes.\n")
        f.write("Formatting simulation output...\n")

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
            (len(antpairs) * uvd.Ntimes, uvd.Nfreqs, 4)
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
    with open(log_path, "a") as f:
        f.write(
            f"Formatting completed. Timing {(time.time()-formatting_start_time)/60.0} minutes.\n"
        )

    # Save as uvfits
    with open(log_path, "a") as f:
        f.write(f"Saving simulation output to {output_uvfits_path}\n")
    try:
        uvd_out.write_uvfits(output_uvfits_path, fix_autos=True)
    except:
        # Save as ms
        ms_path = f"{output_uvfits_path.removesuffix('.uvfits')}.ms"
        with open(log_path, "a") as f:
            f.write(f"Saving simulation output to {ms_path}\n")
        uvd_out.reorder_pols(order="CASA")
        uvd_out.write_ms(ms_path, clobber=True)

    # sys.stdout = stdout_orig
    # sys.stderr = stderr_orig
    # log_file_new.close()


if __name__ == "__main__":

    args = sys.argv
    map_path = args[1]
    beam_path = args[2]
    input_data_path = args[3]
    output_uvfits_path = args[4]
    offset_timesteps = args[5]

    run_fftvis_diffuse_sim(
        map_path=map_path,
        beam_path=beam_path,
        input_data_path=input_data_path,
        output_uvfits_path=output_uvfits_path,
        log_path=None,
        offset_timesteps=offset_timesteps,
    )
