import numpy as np
import pyuvdata
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy


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
    v_coords_wl = np.arange(0, uv_extent, uv_spacing)
    v_mesh, u_mesh = np.meshgrid(v_coords_wl, u_coords_wl)
    u_pixels = len(u_coords_wl)
    v_pixels = len(v_coords_wl)

    # use_baselines = baselines_wl[
    #    np.where(
    #        np.sqrt(baselines_wl[:, 0] ** 2 + baselines_wl[:, 1] ** 2)
    #        < np.sqrt(2) * uv_extent
    #    )[0],
    #    :,
    # ]
    use_baselines_bool = (
        np.abs(baselines_wl[:, 0]) < (uv_extent + antenna_diameter_m * c / freq_hz)
    ) & (np.abs(baselines_wl[:, 1]) < (uv_extent + antenna_diameter_m * c / freq_hz))
    use_baselines = baselines_wl[np.where(use_baselines_bool)[0], :]
    use_Nbls = np.shape(use_baselines)[0]

    weights_mat = np.zeros((u_pixels, v_pixels))
    weights_squared_mat = np.zeros((u_pixels, v_pixels))

    for bl_ind in range(use_Nbls):
        for bl_coords in [use_baselines[bl_ind, :], -use_baselines[bl_ind, :]]:
            uv_offset = np.zeros((u_pixels, v_pixels, 2))
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
    )  # Assumes a circular aperture. Uses the single antenna aperture efficiency. Is this right?
    visibility_rms = (
        wavelength_m**2.0
        * tsys_k
        / (eff_collecting_area * np.sqrt(freq_resolution_hz * int_time_s))
    )
    visibility_stddev_k = visibility_rms / np.sqrt(2)
    visibility_stddev_mk = 1e3 * visibility_stddev_k  # Convert from K to mK

    return visibility_stddev_mk


def generate_uvf_variance(
    save_filepath=None,
    field_of_view_deg2=None,
    min_freq_hz=None,
    max_freq_hz=None,
    antenna_diameter_m=None,
    freq_resolution_hz=None,
    uv_extent=None,
    tsys_k=None,
    aperture_efficiency=None,
    int_time_s=None,
):

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

        visibility_stddev_mk = get_visibility_stddev(
            freq_hz=freq_hz,
            tsys_k=tsys_k,
            aperture_efficiency=aperture_efficiency,
            antenna_diameter_m=antenna_diameter_m,
            freq_resolution_hz=freq_resolution_hz,
            int_time_s=int_time_s,
        )

        print(visibility_stddev_mk)

        uv_plane_variance = (
            visibility_stddev_mk**2.0 * weights_squared_mat / weights_mat**2.0
        )

        if freq_ind == 0:
            uvf_variance = np.full(
                (
                    np.shape(uv_plane_variance)[0],
                    np.shape(uv_plane_variance)[1],
                    len(freq_array_hz),
                ),
                np.nan,
            )
        uvf_variance[:, :, freq_ind] = uv_plane_variance

    if save_filepath is not None:
        np.save(save_filepath, uvf_variance)

    return u_coords_wl, v_coords_wl, freq_array_hz, uvf_variance


def generate_uvn_variance_simple_ft(freq_array_hz, freq_resolution_hz, uvf_variance):

    # where_finite = np.isfinite(uvf_variance)
    # uvn_variance = np.nansum(uvf_variance, axis=2) / np.sum(where_finite, axis=2) ** 2.0
    uvn_variance = np.nansum(uvf_variance, axis=2)
    uvn_variance[np.where(~np.isfinite(np.nanmax(uvf_variance, axis=2)))] = np.nan
    uvn_variance = np.repeat(
        uvn_variance[:, :, np.newaxis], np.shape(uvf_variance)[2], axis=2
    )  # All delays have equal variance

    delay_array_s = np.fft.fftshift(
        np.fft.fftfreq(len(freq_array_hz), d=freq_resolution_hz)
    )
    return delay_array_s, uvn_variance


def mask_foregrounds(
    delay_array_s,
    uvn_variance,
    max_delay=2e-9,
    wedge_slope=0,  # Doesn't support wedge masking yet
    inplace=False,
):
    # Todo: edit this function to mask any modes that _contain_ foregrounds

    mask_modes = np.where(np.abs(delay_array_s) < max_delay)

    if inplace:
        uvn_variance[:, :, mask_modes[0]] = np.nan
    else:
        uvn_variace_masked = np.copy(uvn_variance)
        uvn_variace_masked[:, :, mask_modes[0]] = np.nan
        return uvn_variace_masked


def get_cosmological_parameters():

    cosmological_parameter_dict = {
        "D_H": 3000,  # Hubble distance, units Mpc/h
        "HI rest frame wavelength": 0.21,  # 21 cm wavelength, units m
        "omega_M": 0.27,  # Matter density
        "omega_k": 0,  # Curvature
        "omega_Lambda": 0.73,  # Cosmological constant
        "omega_B": 0.04,  # Baryon density
        "mass_frac_HI": 0.015,  # Mass fraction of neutral hydrogen
        "bias": 0.75,  # Bias between matter PS and HI PS
        "h": 0.71,
    }
    # Derived quantities
    cosmological_parameter_dict["H_0"] = (
        c / cosmological_parameter_dict["D_H"]
    )  # Hubble constant, units h*s/(m Mpc)

    return cosmological_parameter_dict


def uvf_to_cosmology_axis_transform(
    u_coords_wl,
    v_coords_wl,
    freq_array_hz,
    freq_resolution_hz,
):
    # See "Cosmological Parameters and Conversions Memo"

    cosmological_parameter_dict = get_cosmological_parameters()
    hubble_dist = cosmological_parameter_dict["D_H"]
    hubble_const = cosmological_parameter_dict["H_0"]
    rest_frame_wl = cosmological_parameter_dict["HI rest frame wavelength"]
    omega_M = cosmological_parameter_dict["omega_M"]
    omega_k = cosmological_parameter_dict["omega_k"]
    omega_Lambda = cosmological_parameter_dict["omega_Lambda"]

    avg_freq_hz = np.mean(freq_array_hz)
    avg_wl = c / avg_freq_hz
    z = avg_wl / rest_frame_wl - 1

    # Line-of-sight conversion
    delay_array_s = np.fft.fftshift(
        np.fft.fftfreq(len(freq_array_hz), d=freq_resolution_hz)
    )
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


def get_binned_kcube_variance(
    kx,
    ky,
    kz,
    uvn_variance,
    min_k=None,
    max_k=None,
    n_kbins=10,
):

    power_spectrum_cube_inv_variance = 1 / (4.0 * uvn_variance**2.0)

    distance_mat = np.sqrt(
        kx[:, np.newaxis, np.newaxis] ** 2.0
        + ky[np.newaxis, :, np.newaxis] ** 2.0
        + kz[np.newaxis, np.newaxis, :] ** 2.0
    )
    if min_k is None:
        min_k = 0
    if max_k is None:
        max_k = np.max(distance_mat)
    bin_size = (max_k - min_k) / n_kbins
    bin_centers = np.linspace(min_k + bin_size / 2, max_k - bin_size / 2, num=n_kbins)
    binned_ps_variance = np.full(n_kbins, np.nan, dtype=float)
    for bin in range(n_kbins):
        use_values = np.where(np.abs(distance_mat - bin_centers[bin]) <= bin_size / 2)
        if len(use_values[0]) > 0:
            binned_ps_variance[bin] = 1 / np.nansum(
                power_spectrum_cube_inv_variance[use_values]
            )  # Need to correct this so it doesn't return zero when all nans

    return bin_centers, binned_ps_variance


def matter_ps_to_21cm_ps_conversion(
    k_axis,  # Units h/Mpc
    matter_ps,  # Units (Mpc/h)^3
    z,  # redshift
):
    # See Pober et al. 2013
    # Produces a slight difference from the paper results. Why?

    cosmological_parameter_dict = get_cosmological_parameters()
    omega_M = cosmological_parameter_dict["omega_M"]
    omega_Lambda = cosmological_parameter_dict["omega_Lambda"]
    omega_B = cosmological_parameter_dict["omega_B"]
    mass_frac_HI = cosmological_parameter_dict["mass_frac_HI"]
    bias = cosmological_parameter_dict["bias"]
    h = cosmological_parameter_dict["h"]

    brightness_temp = (
        0.084
        * h
        * (1 + z) ** 2.0
        * (omega_M * ((1 + z) ** 3.0) + omega_Lambda) ** -0.5
        * (omega_B / 0.044)
        * (mass_frac_HI / 0.01)
    )  # Units mK
    ps = brightness_temp**2.0 * bias**2.0 * matter_ps  # Units mK^2(Mpc/h)^3

    # Convert to a dimensionless PS
    ps *= k_axis**3.0 / (2 * np.pi**2.0)

    return ps


def get_sample_variance(
    ps_model,  # Units mK^2
    model_k_axis,  # Units h/Mpc
    uv_extent=None,
    field_of_view_deg2=None,
    min_freq_hz=None,
    max_freq_hz=None,
    freq_resolution_hz=None,
    kx=None,
    ky=None,
    kz=None,
    k_bin_size=None,
    k_bin_centers=None,
    min_k=None,
    max_k=None,
    n_kbins=None,
):

    if kx is None or ky is None or kz is None:  # Recalculate axes
        field_of_view_diameter = 2 * np.sqrt(field_of_view_deg2 / np.pi)
        uv_spacing = 0.5 * 180 / field_of_view_diameter  # Nyquist sample the FoV
        freq_array_hz = np.arange(min_freq_hz, max_freq_hz, freq_resolution_hz)

        u_coords_wl = np.arange(0, uv_extent, uv_spacing)
        u_coords_wl = np.append(-np.flip(u_coords_wl[1:]), u_coords_wl)
        v_coords_wl = np.arange(0, uv_extent, uv_spacing)

        kx, ky, kz = uvf_to_cosmology_axis_transform(
            u_coords_wl,
            v_coords_wl,
            freq_array_hz,
            freq_resolution_hz,
        )

    k_dist = np.sqrt(
        kx[:, np.newaxis, np.newaxis] ** 2.0
        + ky[np.newaxis, :, np.newaxis] ** 2.0
        + kz[np.newaxis, np.newaxis, :] ** 2.0
    )
    sample_variance_cube = np.interp(k_dist, model_k_axis, ps_model)

    distance_mat = np.sqrt(
        kx[:, np.newaxis, np.newaxis] ** 2.0
        + ky[np.newaxis, :, np.newaxis] ** 2.0
        + kz[np.newaxis, np.newaxis, :] ** 2.0
    )

    if k_bin_size is None or k_bin_centers is None:  # Recalculate k bins
        if min_k is None:
            min_k = 0
        if max_k is None:
            max_k = np.max(distance_mat)
        k_bin_size = (max_k - min_k) / n_kbins
        k_bin_centers = np.linspace(min_k + k_bin_size / 2, max_k - k_bin_size / 2, num=n_kbins)
    else:
        n_kbins = len(k_bin_centers)

    binned_ps_variance = np.full(n_kbins, np.nan, dtype=float)
    for bin in range(n_kbins):
        use_values = np.where(np.abs(distance_mat - k_bin_centers[bin]) <= k_bin_size / 2)
        if len(use_values[0]) > 0:
            binned_ps_variance[bin] = (
                np.nansum(sample_variance_cube[use_values]) / len(use_values[0]) ** 2.0
            )

    return sample_variance_cube, k_bin_centers, binned_ps_variance


def get_ps_model_Paul2023(z=0.32):
    # Returns the model values from the Paul et al. 2023 MeerKAT analysis

    if z == 0.32:
        ps_model = np.array(
            [
                37.49011751,
                6.45822706,
                2.1502804,
                0.92349697,
                0.40971913,
                0.23231663,
                0.14804924,
            ]
        )
        model_k_axis = np.array([0.43, 0.74, 1.25, 2.04, 3.30, 5.19, 7.96])
    elif z == 0.44:
        ps_model = np.array(
            [
                31.6840213,
                6.93438621,
                3.19883961,
                1.79542739,
                0.9431229,
                0.62953508,
                0.43315279,
            ]
        )
        model_k_axis = np.array([0.34, 0.61, 1.01, 1.68, 2.81, 4.31, 7.04])
    else:
        print("Error: z options are 0.32 and 0.44.")

    # Convert k from Mpc^-1 to h/Mpc
    cosmological_parameter_dict = get_cosmological_parameters()
    h = cosmological_parameter_dict["h"]
    model_k_axis /= h

    # Convert PS to units mK^2
    ps_model *= model_k_axis**3.0 / (2.0 * np.pi**2.0)

    return ps_model, model_k_axis


def delay_ps_sensitivity_analysis(zenith_angle=0):

    min_freq_hz = 0.7e9
    max_freq_hz = c / 0.21
    freq_hz = np.mean([min_freq_hz, max_freq_hz])
    tsys_k = 25
    aperture_efficiency = 0.62
    antenna_diameter_m = 5
    freq_resolution_hz = 162.5e3
    # int_time_s = 1
    int_time_s = 15.0 * 60  # 15 minutes in each survey field
    min_k = 0
    max_k = None
    n_kbins = 50
    max_bl_m = 1000

    visibility_stddev_mk = get_visibility_stddev(
        freq_hz=freq_hz,
        tsys_k=tsys_k,
        aperture_efficiency=aperture_efficiency,
        antenna_diameter_m=antenna_diameter_m,
        freq_resolution_hz=freq_resolution_hz,
        int_time_s=int_time_s,
    )
    freq_array_hz = np.arange(min_freq_hz, max_freq_hz, freq_resolution_hz)
    delay_visibility_variance = visibility_stddev_mk**2.0 * len(freq_array_hz)
    ps_variance = 4.0 * delay_visibility_variance**2.0

    antpos = get_antpos()
    baselines_m = get_baselines(antpos)
    baselines_m = baselines_m[
        np.where(np.sqrt(np.sum(baselines_m**2.0, axis=1)) < max_bl_m)[0], :
    ]
    if zenith_angle != 0:
        baselines_m[:, 0] *= np.cos(np.radians(zenith_angle))
    wavelength = c / freq_hz
    baselines_wl = baselines_m / wavelength

    kx, ky, kz = uvf_to_cosmology_axis_transform(
        baselines_wl[:, 0],
        baselines_wl[:, 1],
        freq_array_hz,
        freq_resolution_hz,
    )

    # 2d binning
    kperp_dist = np.sqrt(kx**2.0 + ky**2.0)
    min_kperp = min_kpar = 0
    max_kperp = np.max(kperp_dist)
    max_kpar = np.max(kz)
    bin_size_kperp = (max_kperp - min_kperp) / n_kbins
    bin_size_kpar = (max_kpar - min_kpar) / n_kbins
    bin_centers_kperp = np.linspace(
        min_kperp + bin_size_kperp / 2, max_kperp - bin_size_kperp / 2, num=n_kbins
    )
    bin_centers_kpar = np.linspace(
        min_kpar + bin_size_kpar / 2, max_kpar - bin_size_kpar / 2, num=n_kbins
    )

    nsamples_kpar = np.zeros(n_kbins, dtype=int)
    for kpar_bin in range(n_kbins):
        use_values_kpar = np.where(
            np.abs(kz - bin_centers_kpar[kpar_bin]) <= bin_size_kpar / 2
        )
        nsamples_kpar[kpar_bin] = len(use_values_kpar[0])
    nsamples_kperp = np.zeros(n_kbins, dtype=int)
    for kperp_bin in range(n_kbins):
        use_values_kperp = np.where(
            np.abs(kperp_dist - bin_centers_kperp[kperp_bin]) <= bin_size_kperp / 2
        )
        nsamples_kperp[kperp_bin] = len(use_values_kperp[0])
    nsamples_2d = np.outer(nsamples_kperp, nsamples_kpar)

    binned_ps_variance_2d = ps_variance / nsamples_2d

    # 1d binning
    kz = np.sort(kz)[
        10:
    ]  # Remove first bins for foreground masking; need to validate which bins to remove

    nsamples = np.zeros(n_kbins, dtype=int)
    distance_mat = np.sqrt(
        kx[:, np.newaxis] ** 2.0 + ky[:, np.newaxis] ** 2.0 + kz[np.newaxis, :] ** 2.0
    )
    if min_k is None:
        min_k = 0
    if max_k is None:
        max_k = np.max(distance_mat)
    bin_size = (max_k - min_k) / n_kbins
    bin_centers = np.linspace(min_k + bin_size / 2, max_k - bin_size / 2, num=n_kbins)
    binned_ps_variance = np.full(n_kbins, np.nan, dtype=float)
    for bin in range(n_kbins):
        use_values = np.where(np.abs(distance_mat - bin_centers[bin]) <= bin_size / 2)
        nsamples[bin] = len(use_values[0])

    binned_ps_variance = ps_variance / nsamples

    return (
        nsamples,
        bin_centers,
        binned_ps_variance,
        nsamples_2d,
        bin_centers_kperp,
        bin_centers_kpar,
        binned_ps_variance_2d,
    )


if __name__ == "__main__":

    int_time_s = 1
    aperture_efficiency = 0.62
    tsys_k = 25
    save_filepath = "/Users/ruby/Astro/dsa2000_variance"
    field_of_view_deg2 = 10.6
    min_freq_hz = 0.7e9
    max_freq_hz = c / 0.21
    antenna_diameter_m = 5
    freq_resolution_hz = 162.5e3
    uv_extent = 1e4

    generate_uvf_variance(
        save_filepath=save_filepath,
        field_of_view_deg2=field_of_view_deg2,
        min_freq_hz=min_freq_hz,
        max_freq_hz=max_freq_hz,
        antenna_diameter_m=antenna_diameter_m,
        freq_resolution_hz=freq_resolution_hz,
        uv_extent=uv_extent,
        tsys_k=tsys_k,
        aperture_efficiency=aperture_efficiency,
        int_time_s=int_time_s,
    )
