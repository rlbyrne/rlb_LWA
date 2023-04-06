import pyuvsim
import pyuvdata
import pyradiosky
import numpy as np

# import matplotlib.pyplot as plt
import sys
from pyuvsim.telescope import BeamList
from astropy.units import Quantity


def create_random_array(
    uvdata_template,
    uv_density_wavelengths,
    uv_extent_wavelengths=50.0,
    freq_min_mhz=150.0,
    freq_max_mhz=200.0,
    min_antenna_spacing_m=14.0,
    c=3e8,
    plot=False,
    include_autocorrelation=True,
    Ntimes=1,
):

    frequency_mhz = np.mean([freq_min_mhz, freq_max_mhz])

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

    if plot:
        fig = plt.figure()
        plt.plot(baseline_locs_u, baseline_locs_v, ".")
        ax = plt.gca()
        ax.set_aspect("equal")
        ax.set_xticks(ticks=u_coords, minor=True)
        ax.set_yticks(ticks=v_coords, minor=True)
        ax.grid(which="minor")
        plt.show()

    # Calculate the corresponding antenna locations
    Nbls = len(baseline_locs_u)
    Nants = Nbls * 2  # Each antenna contributes to only one baseline
    ant_1_array = np.arange(0, Nants, 2)
    ant_2_array = np.arange(1, Nants, 2)
    antenna_locs = np.full((Nants, 2), np.nan, dtype=float)
    baseline_center_dist = 0.0
    for bl_ind in range(Nbls):
        print(f"Placing baseline {bl_ind+1}/{Nbls}")
        ang = np.random.uniform(0, 2 * np.pi)
        ant_1_u = -baseline_locs_u[bl_ind] / 2.0 + (baseline_center_dist * np.cos(ang))
        ant_1_v = -baseline_locs_v[bl_ind] / 2.0 + (baseline_center_dist * np.sin(ang))
        ant_2_u = baseline_locs_u[bl_ind] / 2.0 + (baseline_center_dist * np.cos(ang))
        ant_2_v = baseline_locs_v[bl_ind] / 2.0 + (baseline_center_dist * np.sin(ang))

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
            baseline_center_dist = np.sqrt(
                np.mean([ant_1_u, ant_2_u]) ** 2.0 + np.mean([ant_1_v, ant_2_v]) ** 2.0
            )

        antenna_locs[ant_1_array[bl_ind], 0] = ant_1_u
        antenna_locs[ant_2_array[bl_ind], 0] = ant_2_u
        antenna_locs[ant_1_array[bl_ind], 1] = ant_1_v
        antenna_locs[ant_2_array[bl_ind], 1] = ant_2_v
    # Convert to m
    antenna_locs *= c / frequency_mhz / 1e6

    if plot:
        fig = plt.figure()
        plt.plot(antenna_locs[:, 0], antenna_locs[:, 1], ".")
        ax = plt.gca()
        ax.set_aspect("equal")
        plt.show()

    # Add an autocorrelation value
    if include_autocorrelation:
        ant_1_array = np.append(ant_1_array, 0)
        ant_2_array = np.append(ant_2_array, 0)
        Nbls += 1

    # Generate uvdata object
    print("Generating uvdata object")
    antenna_locs_ENU = np.zeros((Nants, 3))
    antenna_locs_ENU[:, 0] = antenna_locs[:, 0]
    antenna_locs_ENU[:, 1] = antenna_locs[:, 1]

    times = np.unique(uvdata_template.time_array)
    uv = uvdata_template.select(times=times[0:Ntimes], inplace=False)

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
    uv.freq_array = np.arange(freq_min_mhz * 1e6, freq_max_mhz * 1e6, uv.channel_width)[
        np.newaxis, :
    ]
    uv.Nfreqs = np.shape(uv.freq_array)[1]
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


if __name__ == "__main__":

    for uv_density in [10, 5, 1, 0.5]:

        uv = pyuvdata.UVData()
        uv.read_uvfits("/safepool/rbyrne/mwa_data/1061316296.uvfits")
        uv = create_random_array(uv, uv_density)
        print("Writing uvfits")
        uv.write_uvfits(
            f"/safepool/rbyrne/uv_density_simulations_Apr2023/antenna_layout_uv_density_{uv_density}.uvfits"
        )
