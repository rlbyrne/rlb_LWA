import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyuvdata


def polar_plot(
    plot_vals,
    az_vals,
    za_vals,
    vmin=-1,
    vmax=1,
):

    use_cmap = matplotlib.cm.get_cmap("Spectral").copy()
    use_cmap.set_bad(color="whitesmoke")

    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(9, 9))

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

    contourplot = ax.contourf(
        az_vals,
        za_vals,
        plot_vals,
        vmin=vmin,
        vmax=vmax,
        cmap=use_cmap,
    )
    contourplot.set_clim(vmin=vmin, vmax=vmax)
    fig.colorbar(contourplot, ax=ax)
    fig.tight_layout()
    plt.show()


def plot_beam(
    beam,  # pyuvdata beam object
    plot_freq=50.0,  # frequency in MHz, must be included in the beam obj
    real_part=True,
    vmin=-1,
    vmax=1,
):

    use_beam = beam.select(frequencies=[plot_freq * 1e6], inplace=False)
    az_axis = np.degrees(beam.axis1_array)
    za_axis = np.degrees(beam.axis2_array)

    if real_part:
        plot_jones_vals = np.real(beam.data_array)
        title = f"Jones Matrix Components at {plot_freq} MHz, Real Part"
    else:
        plot_jones_vals = np.imag(beam.data_array)
        title = f"Jones Matrix Components at {plot_freq} MHz, Imaginary Part"

    use_cmap = matplotlib.cm.get_cmap("Spectral").copy()
    use_cmap.set_bad(color="whitesmoke")
    za_vals, az_vals = np.meshgrid(za_axis, az_axis)

    fig, ax = plt.subplots(
        nrows=2, ncols=2, subplot_kw=dict(projection="polar"), figsize=(9, 9)
    )
    for pol1 in [0, 1]:
        for pol2 in [0, 1]:
            contourplot = ax[pol1, pol2].contourf(
                np.radians(az_vals),
                za_vals,
                (plot_jones_vals[pol1, 0, pol2, 0, :, :]).T,
                vmin=vmin,
                vmax=vmax,
                cmap=use_cmap,
            )
            contourplot.set_clim(vmin=vmin, vmax=vmax)
            fig.colorbar(contourplot, ax=ax[pol1, pol2])
            ax[pol1, pol2].set_title(f"J[{pol1},{pol2}]")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_coordinate_transform():

    az_axis = np.degrees(beam.axis1_array)
    za_axis = np.degrees(beam.axis2_array)

    if real_part:
        plot_jones_vals = np.real(beam.data_array)
        title = f"Jones Matrix Components at {plot_freq} MHz, Real Part"
    else:
        plot_jones_vals = np.imag(beam.data_array)
        title = f"Jones Matrix Components at {plot_freq} MHz, Imaginary Part"

    use_cmap = matplotlib.cm.get_cmap("Spectral").copy()
    use_cmap.set_bad(color="whitesmoke")
    r_plot, theta_plot = np.meshgrid(za_axis, az_axis)

    fig, ax = plt.subplots(
        nrows=2, ncols=2, subplot_kw=dict(projection="polar"), figsize=(9, 9)
    )
    for pol1 in [0, 1]:
        for pol2 in [0, 1]:
            contourplot = ax[pol1, pol2].contourf(
                np.radians(theta_plot),
                r_plot,
                (plot_jones_vals[pol1, 0, pol2, 0, :, :]).T,
                vmin=vmin,
                vmax=vmax,
                cmap=use_cmap,
            )
            contourplot.set_clim(vmin=-1, vmax=1)
            fig.colorbar(contourplot, ax=ax[pol1, pol2])
            ax[pol1, pol2].set_title(f"J[{pol1},{pol2}]")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def parallactic_angle(az_vals, za_vals, latitude=37.23):

    ra_vals = za_vals * np.cos(az_vals)
    dec_vals = za_vals * np.sin(az_vals) + np.radians(latitude)

    parallactic_angle = np.arctan2(
        -np.sin(ra_vals),
        np.cos(dec_vals) * np.tan(np.radians(latitude))
        - np.sin(dec_vals) * np.cos(ra_vals),
    )

    return ra_vals, dec_vals, parallactic_angle


def rotate_azza_to_radec(beam, latitude=37.23):

    za_vals, az_vals = np.meshgrid(beam.axis2_array, beam.axis1_array)
    ra_vals, dec_vals, parallactic_angle = parallactic_angle(az_vals, za_vals, latitude=latitude)

    rot_matrix = np.zeros((2, 2, beam.Naxes1, beam.Naxes2), dtype=float)
    rot_matrix[0, 0, :, :] = np.sin(parallactic_angle)
    rot_matrix[1, 0, :, :] = -np.cos(parallactic_angle)
    rot_matrix[0, 1, :, :] = -np.cos(parallactic_angle)
    rot_matrix[1, 1, :, :] = -np.sin(parallactic_angle)


if __name__ == "__main__":

    beam = pyuvdata.UVBeam()
    beam.read_beamfits("/Users/ruby/Astro/rlb_LWA/LWAbeam_2015.fits")
    # plot_beam(beam)
    # plot_beam(beam, real_part=False)

    # Test coordinate transformation
    za_vals, az_vals = np.meshgrid(beam.axis2_array, beam.axis1_array)
    ra_vals, dec_vals, parallactic_angle = parallactic_angle(az_vals, za_vals)
    polar_plot(
        parallactic_angle,
        az_vals,
        za_vals,
        vmin=-2*np.pi,
        vmax=2*np.pi,
    )
