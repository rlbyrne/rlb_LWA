import numpy as np
import pyuvdata
from newcal import calibration_wrappers, calibration_optimization, calibration_qa


def calibrate_Aug4():

    use_dates = ["20230801", "20230802", "20230803", "20230804"]
    use_dates = use_dates[1:]  # First date already processed

    for date_stamp in use_dates:
        data = pyuvdata.UVData()
        data.read(f"/data03/rbyrne/{date_stamp}_091100-091600_73MHz.uvfits")
        data.downsample_in_time(n_times_to_avg=data.Ntimes)  # Added to match model
        model = pyuvdata.UVData()
        model.read(f"/data03/rbyrne/{date_stamp}_091100-091600_73MHz_model.uvfits")

        (
            gains_init,
            Nants,
            Nbls,
            Ntimes,
            Nfreqs,
            N_feed_pols,
            model_visibilities,
            data_visibilities,
            visibility_weights,
            gains_exp_mat_1,
            gains_exp_mat_2,
            antenna_names,
        ) = calibration_wrappers.uvdata_calibration_setup(
            data,
            model,
            gain_init_calfile=None,
            gain_init_stddev=0.0,
            N_feed_pols=2,
            min_cal_baseline=30,
        )

        gains_fit = calibration_wrappers.calibration_per_pol(
            gains_init,
            Nants,
            Nbls,
            Nfreqs,
            N_feed_pols,
            model_visibilities,
            data_visibilities,
            visibility_weights,
            gains_exp_mat_1,
            gains_exp_mat_2,
            xtol=1e-3,
            parallel=True,
            verbose=True,
            log_file_path=f"/data03/rbyrne/{date_stamp}_091100-091600_73MHz_cal_log.txt",
        )

        cal = calibration_optimization.create_uvcal_obj(
            data, antenna_names, gains=gains_fit
        )
        cal.write_calfits(
            f"/data03/rbyrne/{date_stamp}_091100-091600_73MHz.calfits",
            clobber=True,
        )

        # Reload data to undo time averaging
        data = pyuvdata.UVData()
        data.read(f"/data03/rbyrne/{date_stamp}_091100-091600_73MHz.uvfits")

        pyuvdata.utils.uvcalibrate(data, cal, inplace=True, time_check=False)
        data.write_uvfits(
            f"/data03/rbyrne/{date_stamp}_091100-091600_73MHz_calibrated.uvfits"
        )


def debug_Aug21():

    date_stamp = "20230801"

    data = pyuvdata.UVData()
    data.read(f"/data03/rbyrne/{date_stamp}_091100-091600_73MHz.uvfits")
    data.downsample_in_time(n_times_to_avg=data.Ntimes)  # Added to match model
    model = pyuvdata.UVData()
    model.read(f"/data03/rbyrne/{date_stamp}_091100-091600_73MHz_model.uvfits")

    for use_pol in [-5, -6]:

        if use_pol == -5:
            pol_name = "X"
        else:
            pol_name = "Y"

        model_use = model.select(polarizations=[use_pol], inplace=False)
        data_use = data.select(polarizations=[use_pol], inplace=False)

        (
            gains_init,
            Nants,
            Nbls,
            Ntimes,
            Nfreqs,
            N_feed_pols,
            model_visibilities,
            data_visibilities,
            visibility_weights,
            gains_exp_mat_1,
            gains_exp_mat_2,
            antenna_names,
        ) = calibration_wrappers.uvdata_calibration_setup(
            data_use,
            model_use,
            gain_init_calfile=None,
            gain_init_stddev=0.0,
            N_feed_pols=1,
            min_cal_baseline=30,
        )

        gains_fit = calibration_wrappers.calibration_per_pol(
            gains_init,
            Nants,
            Nbls,
            Nfreqs,
            N_feed_pols,
            model_visibilities,
            data_visibilities,
            visibility_weights,
            gains_exp_mat_1,
            gains_exp_mat_2,
            xtol=1e-3,
            parallel=True,
            verbose=True,
            log_file_path=f"/data03/rbyrne/{date_stamp}_091100-091600_73MHz_{pol_name}_cal_log.txt",
        )

        cal = calibration_optimization.create_uvcal_obj(
            data, antenna_names, gains=gains_fit
        )
        cal.write_calfits(
            f"/data03/rbyrne/{date_stamp}_091100-091600_73MHz_{pol_name}.calfits",
            clobber=True,
        )


def antenna_dropout_testing_Sep15():

    # Plot gains:
    cal = pyuvdata.UVCal()
    cal.read_calfits("/data03/rbyrne/20230801_091100-091600_73MHz.calfits")
    calibration_optimization.plot_gains(
        cal,
        "/data03/rbyrne/antenna_dropout_testing",
        plot_prefix="20230801_091100-091600_73MHz_iter1",
    )

    data = pyuvdata.UVData()
    data.read("/data03/rbyrne/20230801_091100-091600_73MHz_calibrated.uvfits")
    data.downsample_in_time(n_times_to_avg=data.Ntimes)  # Added to match model
    model = pyuvdata.UVData()
    model.read("/data03/rbyrne/20230801_091100-091600_73MHz_model.uvfits")

    (
        gains_init,
        Nants,
        Nbls,
        Ntimes,
        Nfreqs,
        N_feed_pols,
        model_visibilities,
        data_visibilities,
        visibility_weights,
        gains_exp_mat_1,
        gains_exp_mat_2,
        antenna_names,
    ) = calibration_wrappers.uvdata_calibration_setup(
        data,
        model,
        gain_init_calfile=None,
        gain_init_to_vis_ratio=False,
        gain_init_stddev=0.0,
        N_feed_pols=2,
        min_cal_baseline=30,
    )
    per_ant_cost = calibration_qa.calculate_per_antenna_cost(
        gains_init,
        Nants,
        Nbls,
        Nfreqs,
        N_feed_pols,
        model_visibilities,
        data_visibilities,
        visibility_weights,
        gains_exp_mat_1,
        gains_exp_mat_2,
    )
    f = open(
        "/data03/rbyrne/antenna_dropout_testing/20230801_091100-091600_73MHz_iter1_per_ant_cost.npy",
        "wb",
    )
    np.save(f, per_ant_cost)
    f.close()
    f = open(
        "/data03/rbyrne/antenna_dropout_testing/20230801_091100-091600_73MHz_iter1_ant_names.npy",
        "wb",
    )
    np.save(f, antenna_names)
    f.close()

    calibration_qa.plot_per_ant_cost(
        per_ant_cost,
        antenna_names,
        "/data03/rbyrne/antenna_dropout_testing",
        plot_prefix="20230801_091100-091600_73MHz_iter1",
    )


if __name__ == "__main__":
    antenna_dropout_testing_Sep15()
