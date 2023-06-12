import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyuvdata
import csv


def make_polar_contour_plot(
    ax,
    plot_vals,
    az_vals,
    za_vals,
    vmin=-1,
    vmax=1,
    cyclic_colorbar=False,
    ncontours=11,
):

    if cyclic_colorbar:
        use_cmap = matplotlib.cm.get_cmap("twilight_shifted").copy()
    else:
        if vmin >= 0:
            use_cmap = matplotlib.cm.get_cmap("inferno").copy()
        else:
            use_cmap = matplotlib.cm.get_cmap("Spectral").copy()
    use_cmap.set_bad(color="whitesmoke")

    # Fill in plotting gap by copying az=0 values to az=2Pi
    az_zeros = np.where(az_vals == 0.0)
    az_vals = np.concatenate(
        (az_vals, (az_vals[az_zeros[0], az_zeros[1]] + 2 * np.pi)[np.newaxis, :]),
        axis=0,
    )
    za_vals = np.concatenate(
        (za_vals, (za_vals[az_zeros[0], az_zeros[1]])[np.newaxis, :]), axis=0
    )
    plot_vals = np.concatenate(
        (plot_vals, (plot_vals[az_zeros[0], az_zeros[1]])[np.newaxis, :]), axis=0
    )

    # Set contour levels
    levels = np.linspace(vmin, vmax, num=ncontours)

    contourplot = ax.contourf(
        az_vals,
        za_vals,
        plot_vals,
        levels,
        vmin=vmin,
        vmax=vmax,
        cmap=use_cmap,
    )
    contourplot.set_clim(vmin=vmin, vmax=vmax)
    return contourplot


def make_polar_scatter_plot(
    ax,
    plot_vals,
    az_vals,
    za_vals,
    vmin=-1,
    vmax=1,
    cyclic_colorbar=False,
):

    if cyclic_colorbar:
        use_cmap = matplotlib.cm.get_cmap("twilight_shifted").copy()
    else:
        use_cmap = matplotlib.cm.get_cmap("Spectral").copy()
    use_cmap.set_bad(color="whitesmoke")

    # Fill in plotting gap by copying az=0 values to az=2Pi
    az_zeros = np.where(az_vals == 0.0)
    az_vals = np.concatenate(
        (az_vals, (az_vals[az_zeros[0], az_zeros[1]] + 2 * np.pi)[np.newaxis, :]),
        axis=0,
    )
    za_vals = np.concatenate(
        (za_vals, (za_vals[az_zeros[0], az_zeros[1]])[np.newaxis, :]), axis=0
    )
    plot_vals = np.concatenate(
        (plot_vals, (plot_vals[az_zeros[0], az_zeros[1]])[np.newaxis, :]), axis=0
    )

    scatterplot = ax.scatter(
        az_vals,
        za_vals,
        c=plot_vals,
        cmap=use_cmap,
        s=5,
        edgecolors="black",
        linewidth=0.1,
    )
    ax.set_axisbelow(True)
    scatterplot.set_clim(vmin=vmin, vmax=vmax)
    return scatterplot


def simple_polar_plot(
    plot_vals,
    az_vals,
    za_vals,
    vmin=-1,
    vmax=1,
    contour_plot=True,
    title="",
    cyclic_colorbar=False,
):

    if contour_plot:
        plot_function = make_polar_contour_plot
    else:
        plot_function = make_polar_scatter_plot

    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(9, 9))
    contourplot = plot_function(
        ax,
        plot_vals,
        az_vals,
        za_vals,
        vmin=vmin,
        vmax=vmax,
        cyclic_colorbar=cyclic_colorbar,
    )
    fig.colorbar(contourplot, ax=ax)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_beam(
    beam,  # pyuvdata beam object
    plot_freq=50.0,  # frequency in MHz, must be included in the beam obj
    real_part=True,
    plot_amplitude=False,
    plot_pols=[0, 1],
    vmin=-1,
    vmax=1,
    contour_plot=True,
    savepath=None,
):

    use_beam = beam.select(frequencies=[plot_freq], inplace=False)
    az_axis = np.degrees(beam.axis1_array)
    za_axis = np.degrees(beam.axis2_array)

    if plot_amplitude:
        plot_jones_vals = np.sqrt(
            np.abs(use_beam.data_array[0, :, :, :, :, :]) ** 2.0
            + np.abs(use_beam.data_array[1, :, :, :, :, :]) ** 2.0
        )
        # Normalize
        plot_jones_vals /= np.max(plot_jones_vals)
        title = f"Beam Amplitude, {plot_freq/1e6} MHz"
    elif real_part:
        plot_jones_vals = np.real(use_beam.data_array)
        title = f"Jones Matrix Components at {plot_freq/1e6} MHz, Real Part"
    else:
        plot_jones_vals = np.imag(use_beam.data_array)
        title = f"Jones Matrix Components at {plot_freq/1e6} MHz, Imaginary Part"

    if contour_plot:
        plot_function = make_polar_contour_plot
    else:
        plot_function = make_polar_scatter_plot

    use_cmap = matplotlib.cm.get_cmap("Spectral").copy()
    use_cmap.set_bad(color="whitesmoke")
    za_vals, az_vals = np.meshgrid(za_axis, az_axis)

    if plot_amplitude:
        fig, ax = plt.subplots(
            nrows=1, ncols=2, subplot_kw=dict(projection="polar"), figsize=(9, 6)
        )
        pol_names = ["A", "B"]
        for pol in plot_pols:
            contourplot = plot_function(
                ax[pol],
                (plot_jones_vals[0, pol, 0, :, :]).T,
                np.radians(az_vals),
                za_vals,
                vmin=vmin,
                vmax=vmax,
            )
            fig.colorbar(contourplot, ax=ax[pol])
            ax[pol].set_title(f"Pol {pol_names[pol]}")
    else:
        fig, ax = plt.subplots(
            nrows=2, ncols=2, subplot_kw=dict(projection="polar"), figsize=(9, 9)
        )
        for pol1 in plot_pols:
            for pol2 in [0, 1]:
                contourplot = plot_function(
                    ax[pol1, pol2],
                    (plot_jones_vals[pol1, 0, pol2, 0, :, :]).T,
                    np.radians(az_vals),
                    za_vals,
                    vmin=vmin,
                    vmax=vmax,
                )
                fig.colorbar(contourplot, ax=ax[pol1, pol2])
                ax[pol1, pol2].set_title(f"J[{pol1},{pol2}]")
    fig.suptitle(title)
    fig.tight_layout()
    if savepath is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(savepath, dpi=300)
        plt.close()


def plot_mueller_matrix(
    mueller_mat,  # pyuvdata beam object
    az_axis,
    za_axis,
    freq_axis,
    plot_freq=50.0,  # frequency in MHz, must be included in the beam obj
    real_part=True,
    vmin=-1,
    vmax=1,
    contour_plot=True,
    stokes=False,
):

    freq_ind = np.where(freq_axis == plot_freq)[0][0]
    use_mueller = mueller_mat[:, 0, :, freq_ind, :, :]

    if real_part:
        use_mueller = np.real(use_mueller)
        title = f"Mueller Matrix Components at {plot_freq} MHz, Real Part"
    else:
        use_mueller = np.imag(use_mueller)
        title = f"Mueller Matrix Components at {plot_freq} MHz, Imaginary Part"

    if contour_plot:
        plot_function = make_polar_contour_plot
    else:
        plot_function = make_polar_scatter_plot

    if stokes:
        pol_names = ["I", "Q", "U", "V"]
    else:
        pol_names = ["RA-RA", "Dec-Dec", "RA-Dec", "Dec-RA"]
    instr_pol_names = ["XX", "YY", "XY", "YX"]

    use_cmap = matplotlib.cm.get_cmap("Spectral").copy()
    use_cmap.set_bad(color="whitesmoke")
    za_vals, az_vals = np.meshgrid(za_axis, az_axis)

    fig, ax = plt.subplots(
        nrows=4, ncols=4, subplot_kw=dict(projection="polar"), figsize=(9, 9)
    )
    for pol1 in range(4):
        for pol2 in range(4):
            contourplot = plot_function(
                ax[pol1, pol2],
                (use_mueller[pol1, pol2, :, :]).T,
                np.radians(az_vals),
                za_vals,
                vmin=vmin,
                vmax=vmax,
            )
            fig.colorbar(contourplot, ax=ax[pol1, pol2])
            ax[pol1, pol2].set_title(
                f"M[{pol1},{pol2}], {instr_pol_names[pol2]}->{pol_names[pol1]}"
            )
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def coordinate_transfrom_azza_to_radec(az_vals, za_vals, latitude, hour_angle=0.0):

    ra_vals = hour_angle - np.arctan2(
        -np.sin(za_vals) * np.sin(az_vals - np.pi / 2),
        -np.sin(np.radians(latitude)) * np.sin(za_vals) * np.cos(az_vals - np.pi / 2)
        + np.cos(np.radians(latitude)) * np.cos(za_vals),
    )  # Signs of all terms are determined empirically
    dec_vals = np.pi / 2 - np.arccos(
        np.cos(np.radians(latitude)) * np.sin(za_vals) * np.cos(az_vals - np.pi / 2)
        + np.sin(np.radians(latitude)) * np.cos(za_vals)
    )

    return ra_vals, dec_vals


def get_parallactic_angle(az_vals, za_vals, latitude=37.23):

    ra_vals, dec_vals = coordinate_transfrom_azza_to_radec(az_vals, za_vals, latitude)

    parallactic_angle = np.arctan2(
        -np.sin(ra_vals),
        np.cos(dec_vals) * np.tan(np.radians(latitude))
        - np.sin(dec_vals) * np.cos(ra_vals),
    )

    return ra_vals, dec_vals, parallactic_angle


def pol_basis_transform_azza_to_radec(
    beam, latitude=37.23, inplace=False, reverse=False
):

    za_vals, az_vals = np.meshgrid(beam.axis2_array, beam.axis1_array)
    ra_vals, dec_vals, parallactic_angle = get_parallactic_angle(
        az_vals, za_vals, latitude=latitude
    )

    if reverse:
        parallactic_angle *= -1

    rot_matrix = np.zeros((2, 2, beam.Naxes1, beam.Naxes2), dtype=float)
    rot_matrix[0, 0, :, :] = np.sin(parallactic_angle)
    rot_matrix[1, 0, :, :] = -np.cos(parallactic_angle)
    rot_matrix[0, 1, :, :] = -np.cos(parallactic_angle)
    rot_matrix[1, 1, :, :] = -np.sin(parallactic_angle)

    new_jones_vals = np.einsum("jion,jklmno->iklmno", rot_matrix, beam.data_array)
    new_basis_array = np.einsum("ijon,jlno->ilno", rot_matrix, beam.basis_vector_array)

    if inplace:
        beam.data_array = new_jones_vals
        beam.basis_vector_array = new_basis_array
    else:
        beam_new = beam.copy()
        beam_new.data_array = new_jones_vals
        beam_new.basis_vector_array = new_basis_array
        return beam_new


def convert_jones_to_mueller(beam):

    mueller_mat = np.full(
        (4, 1, 4, beam.Nfreqs, beam.Naxes2, beam.Naxes1), np.nan, dtype=complex
    )

    # XX pol
    mueller_mat[0, 0, 0, :, :, :] = (
        np.abs(beam.data_array[0, 0, 0, :, :, :]) ** 2.0
    )  # RA-RA
    mueller_mat[1, 0, 0, :, :, :] = (
        np.abs(beam.data_array[1, 0, 0, :, :, :]) ** 2.0
    )  # Dec-Dec
    mueller_mat[2, 0, 0, :, :, :] = beam.data_array[0, 0, 0, :, :, :] * np.conj(
        beam.data_array[1, 0, 0, :, :, :]
    )  # RA-Dec
    mueller_mat[3, 0, 0, :, :, :] = beam.data_array[1, 0, 0, :, :, :] * np.conj(
        beam.data_array[0, 0, 0, :, :, :]
    )  # Dec-RA

    # YY pol
    mueller_mat[0, 0, 1, :, :, :] = (
        np.abs(beam.data_array[0, 0, 1, :, :, :]) ** 2.0
    )  # RA-RA
    mueller_mat[1, 0, 1, :, :, :] = (
        np.abs(beam.data_array[1, 0, 1, :, :, :]) ** 2.0
    )  # Dec-Dec
    mueller_mat[2, 0, 1, :, :, :] = beam.data_array[0, 0, 1, :, :, :] * np.conj(
        beam.data_array[1, 0, 1, :, :, :]
    )  # RA-Dec
    mueller_mat[3, 0, 1, :, :, :] = beam.data_array[1, 0, 1, :, :, :] * np.conj(
        beam.data_array[0, 0, 1, :, :, :]
    )  # Dec-RA

    # XY pol
    mueller_mat[0, 0, 2, :, :, :] = beam.data_array[0, 0, 0, :, :, :] * np.conj(
        beam.data_array[0, 0, 1, :, :, :]
    )  # RA-RA
    mueller_mat[1, 0, 2, :, :, :] = beam.data_array[1, 0, 0, :, :, :] * np.conj(
        beam.data_array[1, 0, 1, :, :, :]
    )  # Dec-Dec
    mueller_mat[2, 0, 2, :, :, :] = beam.data_array[0, 0, 0, :, :, :] * np.conj(
        beam.data_array[1, 0, 1, :, :, :]
    )  # RA-Dec
    mueller_mat[3, 0, 2, :, :, :] = beam.data_array[1, 0, 0, :, :, :] * np.conj(
        beam.data_array[0, 0, 1, :, :, :]
    )  # Dec-RA

    # YX pol
    mueller_mat[0, 0, 3, :, :, :] = beam.data_array[0, 0, 1, :, :, :] * np.conj(
        beam.data_array[0, 0, 0, :, :, :]
    )  # RA-RA
    mueller_mat[1, 0, 3, :, :, :] = beam.data_array[1, 0, 1, :, :, :] * np.conj(
        beam.data_array[1, 0, 0, :, :, :]
    )  # Dec-Dec
    mueller_mat[2, 0, 3, :, :, :] = beam.data_array[0, 0, 1, :, :, :] * np.conj(
        beam.data_array[1, 0, 0, :, :, :]
    )  # RA-Dec
    mueller_mat[3, 0, 3, :, :, :] = beam.data_array[1, 0, 1, :, :, :] * np.conj(
        beam.data_array[0, 0, 0, :, :, :]
    )  # Dec-RA

    return mueller_mat


def pol_basis_transform_radec_to_stokes(mueller_mat, inplace=False):

    mueller_mat_new = np.full_like(mueller_mat, np.nan)
    # Stokes I
    mueller_mat_new[0, 0, :, :, :, :] = (
        mueller_mat[0, 0, :, :, :, :] + mueller_mat[1, 0, :, :, :, :]
    )
    # Stokes Q
    mueller_mat_new[1, 0, :, :, :, :] = (
        mueller_mat[0, 0, :, :, :, :] - mueller_mat[1, 0, :, :, :, :]
    )
    # Stokes U
    mueller_mat_new[2, 0, :, :, :, :] = (
        mueller_mat[2, 0, :, :, :, :] + mueller_mat[3, 0, :, :, :, :]
    )
    # Stokes V
    mueller_mat_new[3, 0, :, :, :, :] = (
        1j * mueller_mat[2, 0, :, :, :, :] - 1j * mueller_mat[3, 0, :, :, :, :]
    )

    if inplace:
        mueller_mat = mueller_mat_new
    else:
        return mueller_mat_new


def read_beam_txt_file(path, header_line=6):

    with open(path, "r") as f:
        lines = f.readlines()
    f.close()

    header = lines[header_line]
    npoints = len(lines) - (header_line + 1)

    za_deg = np.zeros(npoints, dtype=float)
    az_deg = np.zeros(npoints, dtype=float)
    freq_mhz = np.zeros(npoints, dtype=float)
    jones_theta = np.zeros(npoints, dtype=complex)
    jones_phi = np.zeros(npoints, dtype=complex)

    for point in range(npoints):
        line_data = [float(value) for value in lines[point + 7].split()]
        za_deg[point] = line_data[0]
        az_deg[point] = line_data[1]
        freq_mhz[point] = line_data[2]
        jones_theta[point] = line_data[7] + 1j * line_data[8]
        jones_phi[point] = line_data[9] + 1j * line_data[10]

    za_axis = np.unique(za_deg)
    freq_axis = np.unique(freq_mhz)
    az_axis = np.unique(az_deg)
    # Fill in other quadrants
    az_axis = np.concatenate(
        (az_axis, az_axis + 90.0, az_axis + 180.0, az_axis + 270.0)
    )
    az_axis[np.where(az_axis < 0.0)] += 360.0
    az_axis[np.where(az_axis >= 360.0)] -= 360.0
    az_axis = np.unique(az_axis)

    jones = np.full(
        (2, 2, len(freq_axis), len(za_axis), len(az_axis)),
        np.nan + 1j * np.nan,
        dtype=complex,
    )

    # Flip the Jones matrix in particular quadrants
    multiply_factors = np.ones(
        (2, 2, 4), dtype=float
    )  # Shape (nfeeds, npols, nquadrants)
    multiply_factors[0, 0, 0:2] = -1.0
    multiply_factors[0, 1, 1:3] = -1.0
    multiply_factors[1, 0, 0] = -1
    multiply_factors[1, 0, 3] = -1
    multiply_factors[1, 1, 0:2] = -1

    for point in range(npoints):
        za_ind = np.where(za_axis == za_deg[point])
        freq_ind = np.where(freq_axis == freq_mhz[point])
        for pol in [0, 1]:
            use_az = az_deg[point]
            if pol == 0:
                use_az = 90.0 - use_az  # Q pol is rotated 90 degrees

            az_ind_quad_1 = np.where(az_axis == use_az)
            jones[0, pol, freq_ind, za_ind, az_ind_quad_1] = (
                multiply_factors[0, pol, 0] * jones_theta[point]
            )
            jones[1, pol, freq_ind, za_ind, az_ind_quad_1] = (
                multiply_factors[1, pol, 0] * jones_phi[point]
            )

            az_ind_quad_2 = np.where(az_axis == 180.0 - use_az)
            jones[0, pol, freq_ind, za_ind, az_ind_quad_2] = (
                multiply_factors[0, pol, 1] * jones_theta[point]
            )
            jones[1, pol, freq_ind, za_ind, az_ind_quad_2] = (
                multiply_factors[1, pol, 1] * jones_phi[point]
            )

            az_ind_quad_3 = np.where(az_axis == 180.0 + use_az)
            jones[0, pol, freq_ind, za_ind, az_ind_quad_3] = (
                multiply_factors[0, pol, 2] * jones_theta[point]
            )
            jones[1, pol, freq_ind, za_ind, az_ind_quad_3] = (
                multiply_factors[1, pol, 2] * jones_phi[point]
            )

            az_ind_quad_4 = np.where(az_axis == 360.0 - use_az)
            jones[0, pol, freq_ind, za_ind, az_ind_quad_4] = (
                multiply_factors[0, pol, 3] * jones_theta[point]
            )
            jones[1, pol, freq_ind, za_ind, az_ind_quad_4] = (
                multiply_factors[1, pol, 3] * jones_phi[point]
            )

    # jones = np.transpose(jones, axes=(1, 0, 2, 3, 4))
    jones = np.flip(jones, axis=1)

    # Polarization mode at zenith is undefined
    # Insert values that will make the conversion to RA/Dec work properly
    # This discards any imaginary component. Is that ok?
    zenith_points = np.where(za_axis == 0)[0]
    jones[1, 0, :, zenith_points, :] = np.sqrt(
        jones[0, 0, :, zenith_points, :] ** 2.0
        + jones[1, 0, :, zenith_points, :] ** 2.0
    )
    jones[0, 1, :, zenith_points, :] = -np.sqrt(
        jones[0, 1, :, zenith_points, :] ** 2.0
        + jones[1, 1, :, zenith_points, :] ** 2.0
    )
    jones[0, 0, :, zenith_points, :] = 0.0
    jones[1, 1, :, zenith_points, :] = 0.0

    beam_obj = pyuvdata.UVBeam()
    beam_obj.Naxes_vec = 2
    beam_obj.Nfreqs = len(freq_axis)
    beam_obj.Nspws = 1
    beam_obj.antenna_type = "simple"
    beam_obj.bandpass_array = np.full((1, len(freq_axis)), 1.0)
    beam_obj.beam_type = "efield"
    beam_obj.data_array = np.copy(jones[:, np.newaxis, :, :, :, :])
    beam_obj.data_normalization = "physical"
    beam_obj.feed_name = ""
    beam_obj.feed_version = ""
    beam_obj.freq_array = (
        np.copy(freq_axis[np.newaxis, :]) * 1e6
    )  # Convert from MHz to Hz
    beam_obj.history = ""
    beam_obj.model_name = ""
    beam_obj.model_version = ""
    beam_obj.pixel_coordinate_system = "az_za"
    beam_obj.spw_array = [0]
    beam_obj.telescope_name = "LWA"
    beam_obj.Naxes1 = len(az_axis)
    beam_obj.Naxes2 = len(za_axis)
    beam_obj.Ncomponents_vec = 2
    beam_obj.Nfeeds = 2
    beam_obj.Npols = 2
    beam_obj.axis1_array = np.radians(az_axis)
    beam_obj.axis2_array = np.radians(za_axis)
    beam_obj.basis_vector_array = np.repeat(
        (
            np.repeat(
                (np.identity(2, dtype=float))[:, :, np.newaxis], len(za_axis), axis=2
            )
        )[:, :, :, np.newaxis],
        len(az_axis),
        axis=3,
    )
    beam_obj.feed_array = ["E", "N"]
    beam_obj.x_orientation = "east"
    beam_obj.peak_normalize()  # Ends up nan-ing the entire beam. Why???
    beam_obj.check()

    return beam_obj


def write_mueller_to_csv(
    mueller_mat,
    az_axis,
    za_axis,
    freq_axis,
    output_path,
    stokes=True,
):

    if stokes:
        pol_names = ["I", "Q", "U", "V"]
    else:
        pol_names = ["RA-RA", "Dec-Dec", "RA-Dec", "Dec-RA"]
    instr_pol_names = ["XX", "YY", "XY", "YX"]

    mueller_dict = []
    for sky_pol_ind, sky_pol_name in enumerate(pol_names):
        for instr_pol_ind, instr_pol_name in enumerate(instr_pol_names):
            for freq_ind, freq in enumerate(freq_axis):
                for az_ind, az in enumerate(az_axis):
                    for za_ind, za in enumerate(za_axis):
                        mueller_val = mueller_mat[
                            sky_pol_ind,
                            0,
                            instr_pol_ind,
                            freq_ind,
                            za_ind,
                            az_ind,
                        ]
                        mueller_dict.append(
                            {
                                "Instr Pol": instr_pol_name,
                                "Sky Pol": sky_pol_name,
                                "Freq (Hz)": freq,
                                "Az": az,
                                "ZA": za,
                                "Value (Real Part)": np.real(mueller_val),
                                "Value (Imag Part)": np.imag(mueller_val),
                            }
                        )

    fieldnames = [
        "Instr Pol",
        "Sky Pol",
        "Freq (Hz)",
        "Az",
        "ZA",
        "Value (Real Part)",
        "Value (Imag Part)",
    ]
    with open(output_path, "w") as file:
        file_contents = csv.DictWriter(file, fieldnames=fieldnames)
        file_contents.writeheader()
        file_contents.writerows(mueller_dict)
    file.close()


def invert_mueller_matrix(mueller_mat, inplace=False):

    mueller_inv = mueller_mat.copy()
    mueller_inv = np.transpose(mueller_inv, axes=(1, 3, 4, 5, 2, 0))
    mueller_inv = np.linalg.inv(mueller_inv)
    mueller_inv = np.transpose(mueller_inv, axes=(5, 0, 4, 1, 2, 3))
    if inplace:
        mueller_mat = mueller_inv
    else:
        return mueller_mat
