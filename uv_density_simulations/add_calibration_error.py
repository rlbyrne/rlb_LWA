import numpy as np
import pyuvdata


def add_cal_error():
    sim_output_path = "/safepool/rbyrne/uv_density_simulations"
    output_path = "/safepool/rbyrne/uv_density_simulations/calibration_error_sim"
    obsids = [
        "sim_uv_spacing_10_short_bls",
        "sim_uv_spacing_5_short_bls",
        "sim_uv_spacing_1_short_bls",
        "sim_uv_spacing_0.5_short_bls",
    ]
    gain_error_stddev = 0.002

    for obs in obsids:
        uv = pyuvdata.UVData()
        uv.read(f"{sim_output_path}/{obs}.uvfits")
        gain_ant_1 = np.random.normal(
            loc=1.0, scale=gain_error_stddev, size=(uv.Nblts, 1, uv.Nfreqs, uv.Npols)
        ) + 1j * np.random.normal(
            loc=0.0, scale=gain_error_stddev, size=(uv.Nblts, 1, uv.Nfreqs, uv.Npols)
        )
        gain_ant_2 = np.random.normal(
            loc=1.0, scale=gain_error_stddev, size=(uv.Nblts, 1, uv.Nfreqs, uv.Npols)
        ) + 1j * np.random.normal(
            loc=0.0, scale=gain_error_stddev, size=(uv.Nblts, 1, uv.Nfreqs, uv.Npols)
        )
        uv.data_array *= gain_ant_1 * gain_ant_2
        uv.write_uvfits(f"{output_path}/{obs}_cal_error.uvfits")


def add_cal_amp_error():
    sim_output_path = "/safepool/rbyrne/uv_density_simulations"
    output_path = "/safepool/rbyrne/uv_density_simulations/calibration_error_sim"
    obsids = [
        "sim_uv_spacing_10_short_bls",
        "sim_uv_spacing_5_short_bls",
        "sim_uv_spacing_1_short_bls",
        "sim_uv_spacing_0.5_short_bls",
    ]
    gain_error_stddev = 0.002

    for obs in obsids:
        uv = pyuvdata.UVData()
        uv.read(f"{sim_output_path}/{obs}.uvfits")
        gain_ant_1 = np.random.normal(
            loc=1.0, scale=gain_error_stddev, size=(uv.Nblts, 1, uv.Nfreqs, uv.Npols)
        )
        gain_ant_2 = np.random.normal(
            loc=1.0, scale=gain_error_stddev, size=(uv.Nblts, 1, uv.Nfreqs, uv.Npols)
        )
        uv.data_array *= gain_ant_1 * gain_ant_2
        uv.write_uvfits(f"{output_path}/{obs}_cal_amp_error.uvfits")


if __name__ == "__main__":
    add_cal_amp_error()
