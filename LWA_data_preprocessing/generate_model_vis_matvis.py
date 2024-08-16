# Adapted from Matthew Kolopanis

import sys
import numpy as np
from tqdm import tqdm
from astropy import units
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.units import Quantity
import time
import pyradiosky
import pyuvdata
import matvis
import fftvis


def run_matvis_diffuse_sim(
    map_path=None,
    beam_path=None,
    input_data_path=None,
    output_uvfits_path=None,
):

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
    beams = [uvb]
    beam_ids = np.zeros(len(antpos), dtype=np.uint8)

    # Get model
    model = pyradiosky.SkyModel.from_skyh5(map_path)
    model.at_frequencies(Quantity(uvd.freq_array, "Hz"))
    if model.component_type == "healpix":
        model.healpix_to_point()
        model.stokes += (
            np.abs(model.stokes[0].min()) + 0.01 * units.Jy
        )  # matvis does not support negative sources

    # Run simulation
    vis_full = np.zeros(
        (uvd.freq_array.size, len([lst]), 2, 2, len(antpos), len(antpos)),
        dtype=np.complex64,
    )

    sim_start_time = time.time()
    for components in tqdm(
        np.array_split(np.arange(model.Ncomponents), np.min([1000, model.Ncomponents])),
        position=1,
        desc="Sources",
    ):
        m2 = model.select(component_inds=components, inplace=False)
        ra_new, dec_new = matvis.conversions.equatorial_to_eci_coords(
            m2.ra.rad, m2.dec.rad, obstime, location, unit="rad", frame="icrs"
        )
        assert np.all(m2.stokes[0] > 0), "Found sources with negative flux"

        for freq_inds in tqdm(
            np.array_split(
                np.arange(uvd.freq_array.size), np.min([50, uvd.freq_array.size])
            ),
            position=0,
            desc="Freqs",
            leave=False,
        ):
            freqs = uvd.freq_array[freq_inds]

            m3 = m2.at_frequencies(freqs * units.Hz, inplace=False)

            vis_full[freq_inds] += matvis.simulate_vis(
                ants=antpos,
                fluxes=m3.stokes[0].T.to_value("Jy"),
                ra=ra_new,
                dec=dec_new,
                freqs=freqs,
                lsts=np.array([lst.to_value("rad")]),
                beams=beams,
                beam_idx=beam_ids,
                polarized=True,
                precision=2,
                latitude=location.lat.rad,
                use_gpu=True,
            )
    sim_duration = time.time() - sim_start_time

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


def combine_mmode_and_sources(
    source_simulation=None,
    mmode_simulation=None,
    output_filename=None,
    adjust_mmode_flux_scale=False,
    calibrated_data_filepath=None,  # Required if adjust_mmode_flux_scale is True
):

    sources = pyuvdata.UVData()
    sources.read(source_simulation)
    sources.phase_to_time(np.mean(sources.time_array))
    sources.filename = [""]

    mmode = pyuvdata.UVData()
    mmode.read(mmode_simulation)
    mmode.phase_to_time(np.mean(sources.time_array))
    mmode.filename = [""]

    if adjust_mmode_flux_scale:

        calibrated_data = pyuvdata.UVData()
        calibrated_data.read(calibrated_data_filepath)
        calibrated_data.phase_to_time(np.mean(sources.time_array))
        calibrated_data.filename = [""]
        calibrated_data.sum_vis(
            sources,
            difference=True,
            inplace=True,
            override_params=[
                "antenna_diameters",
                "integration_time",
                "lst_array",
                "uvw_array",
                "phase_center_id_array",
                "phase_center_app_ra",
                "phase_center_app_dec",
                "phase_center_frame_pa",
                "phase_center_catalog",
                "telescope_location",
                "telescope_name",
                "instrument",
                "flag_array",
                "nsample_array",
                "scan_number_array",
                "timesys",
                "dut1",
                "gst0",
                "earth_omega",
                "rdate",
            ],
        )  # Get residual
        calibrated_data.write_ms(
            "/data03/rbyrne/20231222/test_diffuse_normalization/cal46_time11_newcal_deGasperin_cyg_cas_48MHz_residual.ms",
            clobber=True,
        )
        calibrated_data.select(polarizations=[-5, -6])
        mmode_use = mmode.select(polarizations=[-5, -6], inplace=False)

        mmode_use.data_array[np.where(mmode_use.flag_array)] = 0.0
        calibrated_data.data_array[np.where(calibrated_data.flag_array)] = 0.0

        diff_orig = calibrated_data.sum_vis(
            mmode_use,
            difference=True,
            inplace=False,
            override_params=[
                "antenna_diameters",
                "integration_time",
                "lst_array",
                "uvw_array",
                "phase_center_id_array",
                "phase_center_app_ra",
                "phase_center_app_dec",
                "phase_center_frame_pa",
                "phase_center_catalog",
                "telescope_location",
                "telescope_name",
                "instrument",
                "flag_array",
                "nsample_array",
                "scan_number_array",
                "timesys",
                "dut1",
                "gst0",
                "earth_omega",
                "rdate",
            ],
        )
        print(
            f"Total visibility power, before amplitude offset correction: {np.sum(np.abs(diff_orig.data_array) ** 2.0)}"
        )

        mmode_amp_offset = np.sum(
            np.real(np.conj(mmode_use.data_array) * calibrated_data.data_array)
        ) / np.sum(np.abs(mmode_use.data_array) ** 2.0)

        print(f"Calculated amplitude offset: {mmode_amp_offset}")

        mmode_use.data_array *= mmode_amp_offset
        diff_new = calibrated_data.sum_vis(
            mmode_use,
            difference=True,
            inplace=False,
            override_params=[
                "antenna_diameters",
                "integration_time",
                "lst_array",
                "uvw_array",
                "phase_center_id_array",
                "phase_center_app_ra",
                "phase_center_app_dec",
                "phase_center_frame_pa",
                "phase_center_catalog",
                "telescope_location",
                "telescope_name",
                "instrument",
                "flag_array",
                "nsample_array",
                "scan_number_array",
                "timesys",
                "dut1",
                "gst0",
                "earth_omega",
                "rdate",
            ],
        )
        print(
            f"Total visibility power, after amplitude offset correction: {np.sum(np.abs(diff_new.data_array) ** 2.0)}"
        )

        mmode.data_array *= mmode_amp_offset

    mmode.sum_vis(
        sources,
        inplace=True,
        override_params=[
            "antenna_diameters",
            "integration_time",
            "lst_array",
            "uvw_array",
            "phase_center_id_array",
            "phase_center_app_ra",
            "phase_center_app_dec",
            "phase_center_frame_pa",
            "phase_center_catalog",
            "telescope_location",
            "telescope_name",
            "instrument",
            "flag_array",
            "nsample_array",
            "scan_number_array",
            "timesys",
            "dut1",
            "gst0",
            "earth_omega",
            "rdate",
        ],
    )
    mmode.write_uvfits(output_filename)


if __name__ == "__main__":

    args = sys.argv
    map_path = args[1]
    beam_path = args[2]
    input_data_path = args[3]
    output_uvfits_path = args[4]

    run_matvis_diffuse_sim(
        map_path=map_path,
        beam_path=beam_path,
        input_data_path=input_data_path,
        output_uvfits_path=output_uvfits_path,
    )

    if False:
        combine_mmode_and_sources(
            source_simulation="/data03/rbyrne/20231222/test_diffuse_normalization/cal46_time11_conj_deGasperin_cyg_cas_48MHz_sim.uvfits",
            mmode_simulation="/data03/rbyrne/20231222/test_diffuse_normalization/cal46_time11_conj_mmode_46.992MHz_nside512_sim.uvfits",
            output_filename="/data03/rbyrne/20231222/test_diffuse_normalization/cal46_time11_conj_deGasperin_cyg_cas_48MHz_with_normalized_mmode_sim.uvfits",
            adjust_mmode_flux_scale=True,
            calibrated_data_filepath="/data03/rbyrne/20231222/test_diffuse_normalization/cal46_time11_newcal_deGasperin_cyg_cas_48MHz_with_mmode.ms",
        )
