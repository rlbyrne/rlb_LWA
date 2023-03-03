import numpy as np
import pyuvdata
import matplotlib.pyplot as plt
from matplotlib import cm


c = 3e8


def get_antpos(antpos_filepath="/Users/ruby/Astro/20210226W.cfg"):

    f = open(antpos_filepath, "r")
    antpos_data = f.readlines()
    f.close()

    Nants = len(antpos_data) - 2
    antpos = np.zeros((Nants, 2))
    for ant_ind, ant in enumerate(antpos_data[2:]):
        line = ant.split(" ")
        antpos[ant_ind, 0] = line[0]
        antpos[ant_ind, 1] = line[1]

    return antpos


def get_baselines(antpos):

    Nants = np.shape(antpos)[0]
    Nbls = int((Nants**2 - Nants) / 2)
    baselines_m = np.zeros((Nbls, 2))
    bl_ind = 0
    for ant1 in range(Nants):
        for ant2 in range(ant1 + 1, Nants):
            bl_coords = antpos[ant1, :] - antpos[ant2, :]
            if bl_coords[1] < 0:
                bl_coords *= -1
            baselines_m[bl_ind, :] = bl_coords
            bl_ind += 1

    return baselines_m


def airy_beam_vals(
    uv_offset_wl,
    freq_hz,
    antenna_diameter_m,
):

    wavelength = c / freq_hz
    ant_diameter_wl = antenna_diameter_m / wavelength
    uv_offset_dist = np.sqrt(
        uv_offset_wl[:, :, 0] ** 2.0 + uv_offset_wl[:, :, 1] ** 2.0
    )
    beam_vals = np.zeros_like(uv_offset_wl[:, :, 0])
    nonzero_locs = np.where(uv_offset_dist <= ant_diameter_wl)
    beam_vals[nonzero_locs] = np.real(
        8
        / (np.pi**2.0 * ant_diameter_wl**4.0)
        * (
            ant_diameter_wl**2.0
            * np.arccos(uv_offset_dist[nonzero_locs] / ant_diameter_wl)
            - uv_offset_dist[nonzero_locs]
            * np.sqrt(ant_diameter_wl**2.0 - uv_offset_dist[nonzero_locs] ** 2.0)
        )
    )

    return beam_vals


def create_var_matrix(
    baselines_m,
    freq_hz=None,
    antenna_diameter_m=None,
    uv_extent=None,
    uv_spacing=None,
):

    wavelength = c / freq_hz
    baselines_wl = baselines_m / wavelength
    ant_diameter_wl = antenna_diameter_m / wavelength

    u_coords_wl = np.arange(0, uv_extent, uv_spacing)
    u_coords_wl = np.append(-np.flip(u_coords_wl[1:]), u_coords_wl)
    v_coords_wl = np.copy(u_coords_wl)
    u_mesh, v_mesh = np.meshgrid(u_coords_wl, v_coords_wl)
    u_pixels = len(u_coords_wl)

    use_baselines = baselines_wl[
        np.where(
            np.sqrt(baselines_wl[:, 0] ** 2 + baselines_wl[:, 1] ** 2)
            < np.sqrt(2) * uv_extent
        )[0],
        :,
    ]
    use_Nbls = np.shape(use_baselines)[0]

    weights_mat = np.zeros((u_pixels, u_pixels))
    weights_squared_mat = np.zeros((u_pixels, u_pixels))

    for bl_ind in range(use_Nbls):
        for bl_coords in [use_baselines[bl_ind, :], -use_baselines[bl_ind, :]]:
            uv_offset = np.zeros((u_pixels, u_pixels, 2))
            uv_offset[:, :, 0] = u_mesh - bl_coords[0]
            uv_offset[:, :, 1] = v_mesh - bl_coords[1]
            beam_vals = airy_beam_vals(uv_offset, freq_hz, antenna_diameter_m)
            weights_mat += beam_vals
            weights_squared_mat += beam_vals**2.0

    return u_coords_wl, v_coords_wl, weights_mat, weights_squared_mat


def get_visibility_stddev(
    freq_hz=None,
    tsys_k=None,
    aperture_efficiency=None,
    antenna_diameter_m=None,
    freq_resolution_hz=None,
    int_time_s=None,
):

    wavelength_m = c / freq_hz
    eff_collecting_area = (
        np.pi * antenna_diameter_m**2.0 / 4 * aperture_efficiency
    )  # Uses the single antenna aperture efficiency. Is this right?
    visibility_rms = (
        wavelength_m**2.0
        * tsys_k
        / (eff_collecting_area * np.sqrt(freq_resolution_hz * int_time_s))
    )
    visibility_stddev = visibility_rms / np.sqrt(2)

    return visibility_stddev


def generate_uvf_variance():

    # Set variables
    field_of_view_deg2 = 10.6
    min_freq_hz = 0.7e9
    max_freq_hz = c / 0.21
    antenna_diameter_m = 5
    freq_resolution_hz = 162.5e3
    int_time_s = 1
    aperture_efficiency = 0.62
    tsys_k = 25
    uv_extent = 1000

    field_of_view_diameter = 2 * np.sqrt(field_of_view_deg2 / np.pi)
    uv_spacing = 0.5 * 180 / field_of_view_diameter  # Nyquist sample the FoV
    freq_array_hz = np.arange(min_freq_hz, max_freq_hz, freq_resolution_hz)

    antpos = get_antpos()
    baselines_m = get_baselines(antpos)

    for freq_ind, freq_hz in enumerate(freq_array_hz):
        print(f"On frequency channel {freq_ind+1} of {len(freq_array_hz)}")
        u_coords_wl, v_coords_wl, weights_mat, weights_squared_mat = create_var_matrix(
            baselines_m,
            freq_hz=freq_hz,
            antenna_diameter_m=antenna_diameter_m,
            uv_extent=uv_extent,
            uv_spacing=uv_spacing,
        )
        visibility_stddev = get_visibility_stddev(
            freq_hz=freq_hz,
            tsys_k=tsys_k,
            aperture_efficiency=aperture_efficiency,
            antenna_diameter_m=antenna_diameter_m,
            freq_resolution_hz=freq_resolution_hz,
            int_time_s=int_time_s,
        )
        uv_plane_variance = visibility_stddev**2.0 * weights_squared_mat / weights_mat

        if freq_ind == 0:
            uv_plane_variance_arr = np.full(
                (
                    np.shape(uv_plane_variance)[0],
                    np.shape(uv_plane_variance)[1],
                    len(freq_array_hz),
                ),
                np.nan,
            )
        uv_plane_variance_arr[:, :, freq_ind] = uv_plane_variance

    np.save("/Users/ruby/Astro/dsa2000_variance", uv_plane_variance_arr)


def generate_uvn_variance_simple_ft(uv_plane_variance_arr):

    where_finite = np.isfinite(uv_plane_variance_arr)
    uvn_variance = np.nansum(uv_plane_variance_arr, axis=2) / np.sum(
        where_finite, axis=2
    )
    uvn_variance = np.repeat(
        uvn_variance[:, :, np.newaxis], np.shape(uv_plane_variance_arr)[2], axis=2
    )  # Assume all delays have equal variance
    return uvn_variance


def uvf_to_cosmology_axis_transform(
    u_coords_wl,
    v_coords_wl,
    freq_array_hz,
    freq_resolution_hz,
):
    # See "Cosmological Parameters and Conversions Memo"

    # Set cosmological parameters
    hubble_dist = 3000  # Units Mpc/h
    hubble_const = c / hubble_dist
    rest_frame_wl = 0.21  # Units m
    omega_M = 0.27
    omega_k = 0
    omega_Lambda = 0.73

    avg_freq_hz = np.mean(freq_array_hz)
    avg_wl = c / avg_freq_hz
    z = avg_wl / rest_frame_wl - 1

    # Line-of-sight conversion
    delay_array_s = np.fft.fftfreq(len(freq_array_hz), d=freq_resolution_hz)
    e_func = np.sqrt(omega_M * (1 + z) ** 3.0 + omega_k * (1 + z) ** 2.0 + omega_Lambda)
    kz = (
        (2 * np.pi * hubble_const * e_func)
        / ((1 + z) ** 2.0 * rest_frame_wl)
        * delay_array_s
    )  # units h/Mpc

    # Perpendicular to line-of-sight conversion
    dist_comoving_func = lambda z, omega_M, omega_k, omega_Lambda: 1 / np.sqrt(
        omega_M * (1 + z) ** 3.0 + omega_k * (1 + z) ** 2.0 + omega_Lambda
    )
    dist_comoving_int, err = scipy.integrate.quad(
        dist_comoving_func,
        0,
        7,
        args=(
            omega_M,
            omega_k,
            omega_Lambda,
        ),
    )
    dist_comoving = hubble_dist * dist_comoving_int
    kx = 2 * np.pi * u_coords_wl / dist_comoving  # units h/Mpc
    ky = 2 * np.pi * v_coords_wl / dist_comoving  # units h/Mpc

    return kx, ky, kz


if __name__ == "__main__":

    print("")
