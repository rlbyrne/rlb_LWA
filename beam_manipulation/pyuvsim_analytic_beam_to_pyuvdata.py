import pyuvsim
import pyuvdata
import numpy as np
import jones_to_mueller


def pyuvsim_analytic_to_pyuvdata(
    analytic_beam,
    az_resolution_rad=0.01,
    za_resolution_rad=0.01,
    freq_resolution_hz=10000,
    min_freq_hz=162.0 * 1e6,
    max_freq_hz=202.0 * 1e6,
):

    az_axis = np.arange(0.0, 2 * np.pi, az_resolution_rad, dtype=float)
    za_axis = np.arange(0.0, np.pi / 2.0, za_resolution_rad, dtype=float)
    freq_axis = np.arange(min_freq_hz, max_freq_hz, freq_resolution_hz, dtype=float)

    az_values, za_values = np.meshgrid(az_axis, za_axis)
    beam_values_interp = analytic_beam.interp(
        az_values.flatten(), za_values.flatten(), freq_axis
    )
    beam_values_interp = beam_values_interp[0].reshape(
        2, 1, 2, len(freq_axis), len(za_axis), len(az_axis)
    )
    beam_values_interp = np.flip(
        beam_values_interp, axis=0
    )  # Axis must be flipped for some reason

    beam_obj = pyuvdata.UVBeam()
    beam_obj.Naxes_vec = 2
    beam_obj.Nfreqs = len(freq_axis)
    beam_obj.Nspws = 1
    beam_obj.antenna_type = "simple"
    beam_obj.bandpass_array = np.full((1, len(freq_axis)), 1.0)
    beam_obj.beam_type = "efield"
    beam_obj.data_array = beam_values_interp + 0 * 1j  # Convert to complex type
    beam_obj.data_normalization = "physical"
    beam_obj.feed_name = ""
    beam_obj.feed_version = ""
    beam_obj.freq_array = freq_axis[np.newaxis, :]
    beam_obj.history = "14 m Airy beam"
    beam_obj.model_name = ""
    beam_obj.model_version = ""
    beam_obj.pixel_coordinate_system = "az_za"
    beam_obj.spw_array = [0]
    beam_obj.telescope_name = "airy_sim"
    beam_obj.Naxes1 = len(az_axis)
    beam_obj.Naxes2 = len(za_axis)
    beam_obj.Ncomponents_vec = 2
    beam_obj.Nfeeds = 2
    beam_obj.Npols = 2
    beam_obj.axis1_array = az_axis
    beam_obj.axis2_array = za_axis
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

    jones_to_mueller.pol_basis_transform_azza_to_radec(
        beam_obj, latitude=-26.7, inplace=True, reverse=True
    )

    beam_obj.peak_normalize()

    if not beam_obj.check():
        print("ERROR: Beam object fails check.")

    return beam_obj


if __name__ == "__main__":

    analytic_beam = pyuvsim.AnalyticBeam("airy", diameter=14.0)
    analytic_beam.peak_normalize()
    discrete_beam = pyuvsim_analytic_to_pyuvdata(
        analytic_beam,
        az_resolution_rad=0.01,
        za_resolution_rad=0.01,
        freq_resolution_hz=10000,
        min_freq_hz=162.0 * 1e6,
        max_freq_hz=202.0 * 1e6,
    )
    discrete_beam.write_beamfits("/home/rbyrne/airy_14m.beamfits")
