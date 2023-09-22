import numpy as np
import pyuvdata
import LWA_preprocessing
from newcal import calibration_wrappers, calibration_optimization, cost_function_calculations, calibration_qa


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
    #cal = pyuvdata.UVCal()
    #cal.read_calfits("/data03/rbyrne/20230801_091100-091600_73MHz.calfits")
    #calibration_optimization.plot_gains(
    #    cal,
    #    "/data03/rbyrne/antenna_dropout_testing",
    #    plot_prefix="20230801_091100-091600_73MHz_iter1",
    #)

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


def apply_antenna_flagging_and_recalibrate_Sept20():

    with open("/data03/rbyrne/antenna_dropout_testing/20230801_091100-091600_73MHz_iter1_per_ant_cost.npy", "rb") as f:
        per_ant_cost = np.load(f)
    f.close()
    with open("/data03/rbyrne/antenna_dropout_testing/20230801_091100-091600_73MHz_iter1_ant_names.npy", "rb") as f:
        antenna_names_per_ant_cost = np.load(f)
    f.close()

    data = pyuvdata.UVData()
    data.read("/data03/rbyrne/20230801_091100-091600_73MHz.uvfits")
    data.downsample_in_time(n_times_to_avg=data.Ntimes)
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
        gain_init_calfile="/data03/rbyrne/20230801_091100-091600_73MHz.calfits",
        gain_init_stddev=0.0,
        N_feed_pols=2,
        min_cal_baseline=30,
    )

    # Make sure antenna ordering matches
    antenna_inds = np.array([list(antenna_names_per_ant_cost).index(name) for name in antenna_names])
    per_ant_cost = per_ant_cost[antenna_inds, :]

    flag_antenna_list, visibility_weights = calibration_qa.get_antenna_flags_from_per_ant_cost(
        per_ant_cost,
        antenna_names,
        flagging_threshold=2.5,
        visibility_weights=visibility_weights,
        gains_exp_mat_1=gains_exp_mat_1,
        gains_exp_mat_2=gains_exp_mat_2,
    )

    print(gains_init[:, 0, 0])
    print(antenna_names[np.where(~np.isfinite(gains_init[:,0]))])
    gains_init[~np.isfinite(gains_init)] = 0.0
    example_cost = cost_function_calculations.cost_function_single_pol(
        gains_init[:, 0, 0],
        model_visibilities[:, :, 0, 0],
        data_visibilities[:, :, 0, 0],
        visibility_weights[:, :, 0, 0],
        gains_exp_mat_1,
        gains_exp_mat_2,
        0.0,
    )
    print(example_cost)
    hess = cost_function_calculations.hessian_single_pol(
        gains_init[:, 0, 0],
        Nants,
        Nbls,
        model_visibilities[:, :, 0, 0],
        data_visibilities[:, :, 0, 0],
        visibility_weights[:, :, 0, 0],
        gains_exp_mat_1,
        gains_exp_mat_2,
        0.0,
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
        log_file_path="/data03/rbyrne/20230801_091100-091600_73MHz_cal_iter2_log.txt",
    )

    cal = calibration_optimization.create_uvcal_obj(
        data, antenna_names, gains=gains_fit
    )
    cal.write_calfits(
        "/data03/rbyrne/20230801_091100-091600_73MHz_iter2.calfits",
        clobber=True,
    )

    # Reload data to undo time averaging
    data = pyuvdata.UVData()
    data.read("/data03/rbyrne/20230801_091100-091600_73MHz.uvfits")

    # Apply new antenna flagging
    LWA_preprocessing.flag_antennas(
        data,
        antenna_names=flag_antenna_list[:, 0],
        flag_pol="X",
        inplace=True,
    )
    LWA_preprocessing.flag_antennas(
        data,
        antenna_names=flag_antenna_list[:, 1],
        flag_pol="Y",
        inplace=True,
    )

    pyuvdata.utils.uvcalibrate(data, cal, inplace=True, time_check=False)
    data.write_uvfits(
        "/data03/rbyrne/20230801_091100-091600_73MHz_calibrated_iter2.uvfits"
    )


if __name__ == "__main__":
    apply_antenna_flagging_and_recalibrate_Sept20()
