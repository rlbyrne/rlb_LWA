# Adapted from Matthew Kolopanis

import sys
import numpy as np
from tqdm import tqdm
from astropy import units
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.units import Quantity
import pyradiosky
import pyuvdata
import matvis


def run_matvis_diffuse_sim(map_path, beam_path, input_data_path, output_uvfits_path):

    # Read metadata from file
    uvd = pyuvdata.UVData.from_file(
        input_data_path,
        read_data=False,
        use_future_array_shapes=True,
    )
    if uvd.telescope_name == "OVRO_MMA":  # Correct telescope location
        uvd.telescope_name = "OVRO-LWA"
        uvd.set_telescope_params(overwrite=True, warn=True)
    uvd.phase_to_time(np.mean(uvd.time_array))  # Phase data
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
    beams = [uvb]
    beam_ids = np.zeros(len(antpos), dtype=np.uint8)

    # Get model
    model = pyradiosky.SkyModel.from_skyh5(map_path)
    model.at_frequencies(Quantity(uvd.freq_array, "Hz"))
    model.healpix_to_point()
    model.stokes += (
        np.abs(model.stokes[0].min()) + 0.01 * units.Jy
    )  # matvis does not support negative sources

    # Run simulation
    vis_full = np.zeros(
        (uvd.freq_array.size, len([lst]), 2, 2, len(antpos), len(antpos)),
        dtype=np.complex64,
    )

    for components in tqdm(
        np.array_split(np.arange(model.Ncomponents), 5000), position=1, desc="Sources"
    ):
        m2 = model.select(component_inds=components, inplace=False)
        ra_new, dec_new = matvis.conversions.equatorial_to_eci_coords(
            m2.ra.rad, m2.dec.rad, obstime, location, unit="rad", frame="icrs"
        )
        assert np.all(m2.stokes[0] > 0), "Found sources with negative flux"

        for freq_inds in tqdm(
            np.array_split(np.arange(uvd.freq_array.size), 5),
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
                use_gpu=False,
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
    uvd_out.reorder_pols(order="AIPS")
    uvd_out.phase_to_time(np.mean(uvd.time_array))
    uvd_out.check()

    # Save as uvfits
    uvd_out.write_uvfits(output_uvfits_path, fix_autos=True)

    # Save as ms
    uvd_out.reorder_pols(order="CASA")
    uvd_out.write_ms(f"{output_uvfits_path.removesuffix('.uvfits')}.ms", clobber=True)


def reformat_matvis_output(
    orig_filepath, reference_filepath, new_filepath
):  # This function is to bootstrap the output after the fact, should be combined with the above function before the next run

    uvd = pyuvdata.UVData()
    uvd.read(reference_filepath)
    uvd_out = pyuvdata.UVData()
    uvd_out.read(orig_filepath)

    # Remove duplicate baselines
    uvd_out.conjugate_bls()
    baselines = list(zip(uvd_out.ant_1_array, uvd_out.ant_2_array))
    keep_baselines = []
    for bl_ind, baseline in enumerate(baselines):
        if baseline not in baselines[:bl_ind]:
            keep_baselines.append(bl_ind)
    uvd_out.select(blt_inds=keep_baselines)

    # assign antenna names
    for ant_ind in range(len(uvd_out.antenna_names)):
        uvd_out.antenna_names[ant_ind] = uvd.antenna_names[
            np.where(uvd.antenna_numbers == uvd_out.antenna_numbers[ant_ind])[0][0]
        ]

    uvd_out.reorder_blts()
    uvd_out.reorder_pols(order="AIPS")
    uvd_out.reorder_freqs(channel_order="freq")
    uvd_out.write_ms(new_filepath, clobber=True)


def combine_mmode_and_sources(
    source_simulation="/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_cyg_cas_sim.ms",
    mmode_maps=[
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_mmode_sim.ms",
        "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_matvis_sim_reformatted.ms",
        "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_matvis_sim_nside512_reformatted.ms",
    ],
    output_filenames=[
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_mmode_with_cyg_cas_pyuvsim_nside128_sim.ms",
        "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_with_cyg_cas_matvis_nside128_sim.ms",
        "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_with_cyg_cas_matvis_nside512_sim.ms",
    ],
):

    sources = pyuvdata.UVData()
    sources.read(source_simulation)
    sources.phase_to_time(np.mean(sources.time_array))
    sources.filename = [""]

    for mmode_map_ind in range(len(mmode_maps)):
        mmode = pyuvdata.UVData()
        mmode.read(mmode_maps[mmode_map_ind])
        mmode.phase_to_time(np.mean(sources.time_array))
        mmode.filename = [""]
        mmode.sum_vis(
            sources,
            inplace=True,
            override_params=[
                "antenna_diameters",
                "integration_time",
                "lst_array",
                "phase_center_id_array",
                "phase_center_app_ra",
                "phase_center_app_dec",
                "phase_center_frame_pa",
                "phase_center_catalog",
                "telescope_location",
                "telescope_name",
                "instrument",
                "filename",
                "uvw_array",
            ],
        )
        mmode.write_ms(output_filenames[mmode_map_ind], clobber=True)


if __name__ == "__main__":

    if False:
        args = sys.argv
        map_path = args[1]
        beam_path = args[2]
        input_data_path = args[3]
        output_uvfits_path = args[4]

    if False:
        map_path = "/fast/rbyrne/skymodels/ovro_lwa_sky_map_46.992MHz_nside512.skyh5"
        beam_path = "/home/rbyrne/rlb_LWA/LWAbeam_2015.fits"
        input_data_path = (
            "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj.ms"
        )
        output_uvfits_path = "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_matvis_sim_nside512.uvfits"
        run_matvis_diffuse_sim(map_path, beam_path, input_data_path, output_uvfits_path)

    # reformat_matvis_output(
    #    "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_matvis_sim.ms",
    #    "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_cyg_cas_sim.ms",
    #    "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_matvis_sim_reformatted.ms",
    # )

    reformat_matvis_output(
        "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_matvis_sim_nside512.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_cyg_cas_sim.ms",
        "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_matvis_sim_nside512_reformatted.ms",
    )

    combine_mmode_and_sources()
