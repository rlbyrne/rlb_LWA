import pyuvsim
import pyuvdata
import pyradiosky
import numpy as np
#import matplotlib.pyplot as plt
import sys
from pyuvsim.telescope import BeamList
from astropy.units import Quantity


def create_random_array(
    uvdata_template,
    uv_density_wavelengths,
    uv_extent_wavelengths=100.0,
    frequency_mhz=182.0,
    min_antenna_spacing_m=14.0,
    c=3e8,
    plot=False,
):

    # Calculate random baseline locations in the uv plane
    coord_extent = (
        np.floor(uv_extent_wavelengths / uv_density_wavelengths)
        * uv_density_wavelengths
        + uv_density_wavelengths / 2
    )
    u_coords = np.linspace(
        -coord_extent,
        coord_extent,
        num=2 * int(np.floor(uv_extent_wavelengths / uv_density_wavelengths)) + 2,
    )
    v_coords = np.linspace(
        -uv_density_wavelengths / 2,
        coord_extent,
        num=int(np.floor(uv_extent_wavelengths / uv_density_wavelengths)) + 2,
    )
    min_antenna_spacing_wavelengths = min_antenna_spacing_m / c * frequency_mhz * 1e6

    baseline_locs_u = []
    baseline_locs_v = []
    for u_ind in range(len(u_coords) - 1):
        for v_ind in range(len(v_coords) - 1):
            if u_coords[u_ind] > 0 or v_coords[v_ind] > 0:
                u_loc = np.random.uniform(u_coords[u_ind], u_coords[u_ind + 1])
                v_loc = np.random.uniform(v_coords[v_ind], v_coords[v_ind + 1])
                bl_length = np.sqrt(u_loc**2 + v_loc**2)

                if (
                    bl_length < uv_extent_wavelengths
                    and bl_length > min_antenna_spacing_wavelengths
                ):
                    baseline_locs_u.append(u_loc)
                    baseline_locs_v.append(v_loc)

    #if plot:
    #    fig = plt.figure()
    #    plt.plot(baseline_locs_u, baseline_locs_v, ".")
    #    ax = plt.gca()
    #    ax.set_aspect("equal")
    #    ax.set_xticks(ticks=u_coords, minor=True)
    #    ax.set_yticks(ticks=v_coords, minor=True)
    #    ax.grid(which="minor")
    #    plt.show()

    # Calculate the corresponding antenna locations
    Nbls = len(baseline_locs_u)
    Nants = Nbls * 2  # Each antenna contributes to only one baseline
    ant_1_array = np.arange(0, Nants, 2)
    ant_2_array = np.arange(1, Nants, 2)
    antenna_locs = np.full((Nants, 2), np.nan, dtype=float)
    for bl_ind in range(Nbls):
        ant_1_u = -baseline_locs_u[bl_ind] / 2.0
        ant_1_v = -baseline_locs_v[bl_ind] / 2.0
        ant_2_u = baseline_locs_u[bl_ind] / 2.0
        ant_2_v = baseline_locs_v[bl_ind] / 2.0

        # if False:
        if bl_ind > 0:
            antenna_1_spacings = np.sqrt(
                (antenna_locs[:, 0] - ant_1_u) ** 2.0
                + (antenna_locs[:, 1] - ant_1_v) ** 2.0
            )
            antenna_2_spacings = np.sqrt(
                (antenna_locs[:, 0] - ant_2_u) ** 2.0
                + (antenna_locs[:, 1] - ant_2_v) ** 2.0
            )
            while (
                np.nanmin(antenna_1_spacings) < min_antenna_spacing_wavelengths
                or np.nanmin(antenna_2_spacings) < min_antenna_spacing_wavelengths
            ):
                # Random walk
                ang = np.random.uniform(0, 2 * np.pi)
                ant_1_u += min_antenna_spacing_wavelengths * np.cos(ang)
                ant_2_u += min_antenna_spacing_wavelengths * np.cos(ang)
                ant_1_v += min_antenna_spacing_wavelengths * np.sin(ang)
                ant_2_v += min_antenna_spacing_wavelengths * np.sin(ang)
                # Recalculate spacings
                antenna_1_spacings = np.sqrt(
                    (antenna_locs[:, 0] - ant_1_u) ** 2.0
                    + (antenna_locs[:, 1] - ant_1_v) ** 2.0
                )
                antenna_2_spacings = np.sqrt(
                    (antenna_locs[:, 0] - ant_2_u) ** 2.0
                    + (antenna_locs[:, 1] - ant_2_v) ** 2.0
                )

        antenna_locs[ant_1_array[bl_ind], 0] = ant_1_u
        antenna_locs[ant_2_array[bl_ind], 0] = ant_2_u
        antenna_locs[ant_1_array[bl_ind], 1] = ant_1_v
        antenna_locs[ant_2_array[bl_ind], 1] = ant_2_v
    # Convert to m
    antenna_locs *= c / frequency_mhz / 1e6

    #if plot:
    #    fig = plt.figure()
    #    plt.plot(antenna_locs[:, 0], antenna_locs[:, 1], ".")
    #    ax = plt.gca()
    #    ax.set_aspect("equal")
    #    plt.show()

    # Generate uvdata object
    antenna_locs_ENU = np.zeros((Nants, 3))
    antenna_locs_ENU[:, 0] = antenna_locs[:, 0]
    antenna_locs_ENU[:, 1] = antenna_locs[:, 1]

    uv.Nants_data = Nants
    uv.Nants_telescope = Nants
    uv.Nbls = Nbls
    uv.Nblts = Nbls * uv.Ntimes
    uv.antenna_numbers = np.arange(Nants)
    uv.antenna_names = np.array([str(ind) for ind in np.arange(Nants)])
    uv.ant_1_array = np.tile(ant_1_array, uv.Ntimes)
    uv.ant_2_array = np.tile(ant_2_array, uv.Ntimes)
    baseline_array = 2048 * (ant_1_array + 1) + ant_2_array + 1 + 2**16
    uv.baseline_array = np.tile(baseline_array, uv.Ntimes)
    old_time_array = np.copy(uv.time_array)
    uv.time_array = np.repeat(np.array(list(set(old_time_array))), uv.Nbls)
    uv.lst_array = np.array(
        [uv.lst_array[np.where(old_time_array == time)[0][0]] for time in uv.time_array]
    )
    uv.nsample_array = np.full(
        (uv.Nblts, uv.Nspws, uv.Nfreqs, uv.Npols), 1.0, dtype=float
    )
    uv.integration_time = np.full((uv.Nblts), np.mean(uv.integration_time))
    # Unflag all
    uv.flag_array = np.full(
        (uv.Nblts, uv.Nspws, uv.Nfreqs, uv.Npols), False, dtype=bool
    )
    # Calculate UVWs
    antenna_locs_ECEF = pyuvdata.utils.ECEF_from_ENU(
        antenna_locs_ENU, *uv.telescope_location_lat_lon_alt
    )
    uv.antenna_positions = antenna_locs_ECEF - uv.telescope_location
    # Add dummy data
    uv.data_array = np.zeros(
        (uv.Nblts, uv.Nspws, uv.Nfreqs, uv.Npols),
        dtype=complex,
    )
    uv.uvw_array = np.zeros((uv.Nblts, 3), dtype=float)
    uv.unphase_to_drift()
    uv.set_uvws_from_antenna_positions(
        allow_phasing=True, orig_phase_frame="gcrs", output_phase_frame="icrs"
    )
    uv.phase_to_time(np.mean(uv.time_array))
    uv.check()

    return uv


def get_airy_beam(diameter_m=14.0):

    airy_beam = pyuvsim.AnalyticBeam("airy", diameter=diameter_m)
    airy_beam.peak_normalize()

    return airy_beam


if __name__ == "__main__":

    for uv_density in [10, 5, 1, 0.5]:

        uv = pyuvdata.UVData()
        uv.read_uvfits("/safepool/rbyrne/mwa_data/1061316296.uvfits")
        uv = create_random_array(uv, 20.0)
        uv.write_uvfits(
            f"/safepool/rbyrne/uv_density_simulations/antenna_layout_uv_density_{uv_density}.uvfits"
        )

        airy_beam = get_airy_beam()

        ### Run healpix sim
        healpix_map_path = "/safepool/rbyrne/diffuse_map.skyh5"
        diffuse_map = pyradiosky.SkyModel()
        print("Reading map")
        diffuse_map.read_skyh5(healpix_map_path)

        # Reformat the map with a spectral index
        diffuse_map.spectral_type = "spectral_index"
        diffuse_map.spectral_index = np.full(diffuse_map.Ncomponents, -0.8)
        diffuse_map.reference_frequency = Quantity(
            np.full(diffuse_map.Ncomponents, diffuse_map.freq_array[0].value), "Hz"
        )
        diffuse_map.freq_array = None
        if not diffuse_map.check():
            print("WARNING: Diffuse map fails check.")

        diffuse_map_pyuvsim_formatted = pyuvsim.simsetup.SkyModelData(diffuse_map)
        # The formatted map has the reference frequency stripped; correct this
        diffuse_map_pyuvsim_formatted.reference_frequency = diffuse_map.reference_frequency.value

        print("Starting diffuse simulation")
        diffuse_sim_uv = pyuvsim.uvsim.run_uvdata_uvsim(
            input_uv=uv,
            beam_list=BeamList(beam_list=[airy_beam]),
            beam_dict=None,  # Same beam for all ants
            catalog=diffuse_map_pyuvsim_formatted,
            quiet=False,
        )

        ### Run catalog sim
        catalog_path = "/home/rbyrne/FHD/catalog_data/GLEAM_v2_plus_rlb2019.sav"
        catalog = pyradiosky.SkyModel()
        print("Reading catalog")
        catalog.read_fhd_catalog(catalog_path)
        print("Starting catalog simulation")
        catalog_sim_uv = pyuvsim.uvsim.run_uvdata_uvsim(
            input_uv=uv,
            beam_list=BeamList(beam_list=[airy_beam]),
            beam_dict=None,  # Same beam for all ants
            catalog=pyuvsim.simsetup.SkyModelData(catalog),
            quiet=False,
        )

        catalog_sim_uv.sum_vis(diffuse_sim_uv, inplace=True)
        catalog_sim_uv.write_uvfits(
            f"/safepool/rbyrne/uv_density_simulations/sim_output_uv_density_{uv_density}.uvfits"
        )
