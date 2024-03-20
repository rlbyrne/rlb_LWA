import numpy as np
import pyuvdata
import LWA_preprocessing
from newcal import (
    calibration_wrappers,
    calibration_optimization,
    cost_function_calculations,
    calibration_qa,
)


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
    # cal = pyuvdata.UVCal()
    # cal.read_calfits("/data03/rbyrne/20230801_091100-091600_73MHz.calfits")
    # calibration_optimization.plot_gains(
    #    cal,
    #    "/data03/rbyrne/antenna_dropout_testing",
    #    plot_prefix="20230801_091100-091600_73MHz_iter1",
    # )

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

    with open(
        "/data03/rbyrne/antenna_dropout_testing/20230801_091100-091600_73MHz_iter1_per_ant_cost.npy",
        "rb",
    ) as f:
        per_ant_cost = np.load(f)
    f.close()
    with open(
        "/data03/rbyrne/antenna_dropout_testing/20230801_091100-091600_73MHz_iter1_ant_names.npy",
        "rb",
    ) as f:
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
        min_cal_baseline=30,
    )

    # Make sure antenna ordering matches
    antenna_inds = np.array(
        [list(antenna_names_per_ant_cost).index(name) for name in antenna_names]
    )
    per_ant_cost = per_ant_cost[antenna_inds, :]

    (
        flag_antenna_list,
        visibility_weights,
    ) = calibration_qa.get_antenna_flags_from_per_ant_cost(
        per_ant_cost,
        antenna_names,
        flagging_threshold=2.5,
        visibility_weights=visibility_weights,
        gains_exp_mat_1=gains_exp_mat_1,
        gains_exp_mat_2=gains_exp_mat_2,
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


def casa_calibration_comparison_Feb23():

    data = pyuvdata.UVData()
    data.read_ms("/data03/rbyrne/20231222/cal46.ms", data_column="DATA")
    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/cal46.ms", data_column="MODEL_DATA")

    caldata_obj = calibration_wrappers.CalData()
    caldata_obj.load_data(data, model)
    calibration_wrappers.calibration_per_pol(
        caldata_obj,
        verbose=True,
        log_file_path="/data03/rbyrne/20231222/newcal_calibration/calibration_log.txt",
    )
    uvcal = caldata_obj.convert_to_uvcal()
    uvcal.write_calfits(
        "/data03/rbyrne/20231222/newcal_calibration/cal46.calfits",
        clobber=True,
    )


def apply_calibration_Mar2():

    cal = pyuvdata.UVCal()
    cal.read("/data03/rbyrne/20231222/newcal_calibration/cal46.calfits")
    filenames = [
        "46time2",
        "46time3",
        "46time4",
        "46time5",
        "46time6",
    ]
    for file in filenames:
        uv = pyuvdata.UVData()
        uv.read_ms(f"/data03/rbyrne/20231222/{file}.ms")
        pyuvdata.utils.uvcalibrate(uv, cal, inplace=True, time_check=False)
        uv.reorder_pols(order="CASA", run_check=False)
        uv.write_ms(
            f"/data03/rbyrne/20231222/newcal_calibration/{file}_newcal.ms",
            flip_conj=True,
            run_check=False,
        )


def casa_calibration_comparison_Mar5():

    data = pyuvdata.UVData()
    data.read_ms("/data03/rbyrne/20231222/cal46.ms", data_column="DATA")
    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/cal46.ms", data_column="MODEL_DATA")

    caldata_obj = calibration_wrappers.CalData()
    caldata_obj.load_data(data, model, min_cal_baseline_lambda=15)
    calibration_wrappers.calibration_per_pol(
        caldata_obj,
        verbose=True,
        log_file_path="/data03/rbyrne/20231222/newcal_calibration/calibration_log_Mar5.txt",
    )
    uvcal = caldata_obj.convert_to_uvcal()
    uvcal.write_calfits(
        "/data03/rbyrne/20231222/newcal_calibration/cal46_min_bl_15.calfits",
        clobber=True,
    )


def apply_calibration_Mar18():

    cal = pyuvdata.UVCal()
    cal.read("/data03/rbyrne/20231222/newcal_calibration/cal46_min_bl_15.calfits")
    filenames = [
        "46time2",
        "46time3",
        "46time4",
        "46time5",
        "46time6",
    ]
    for file in filenames:
        uv = pyuvdata.UVData()
        uv.read_ms(f"/data03/rbyrne/20231222/{file}.ms")
        pyuvdata.utils.uvcalibrate(uv, cal, inplace=True, time_check=False)
        uv.reorder_pols(order="CASA", run_check=False)
        uv.write_ms(
            f"/data03/rbyrne/20231222/newcal_calibration/{file}_newcal_min_bl_15.ms",
            flip_conj=True,
            run_check=False,
        )


def test_single_time_Mar19():

    data = pyuvdata.UVData()
    data.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_data.ms")
    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_model.ms")

    caldata_obj = calibration_wrappers.CalData()
    caldata_obj.load_data(data, model, min_cal_baseline_lambda=15)
    calibration_wrappers.calibration_per_pol(
        caldata_obj,
        verbose=True,
        log_file_path="/data03/rbyrne/20231222/newcal_single_time/calibration_logs/newcal_Mar19.txt",
    )
    uvcal = caldata_obj.convert_to_uvcal()
    uvcal.write_calfits(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small.calfits",
        clobber=True,
    )

    # Apply calibration
    data = pyuvdata.UVData()
    data.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_data.ms")
    pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)
    data.reorder_pols(order="CASA", run_check=False)
    data.write_ms(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_newcal_calibrated.ms",
        flip_conj=True,
        run_check=False,
    )


def test_single_time_recalibrate_Mar19():

    data = pyuvdata.UVData()
    data.read_ms(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_casa_calibrated.ms"
    )
    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_model.ms")

    caldata_obj = calibration_wrappers.CalData()
    caldata_obj.load_data(data, model, min_cal_baseline_lambda=15)
    calibration_wrappers.calibration_per_pol(
        caldata_obj,
        verbose=True,
        log_file_path="/data03/rbyrne/20231222/newcal_single_time/calibration_logs/newcal_recalibrate_Mar19.txt",
    )
    uvcal = caldata_obj.convert_to_uvcal()
    uvcal.write_calfits(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_recalibrated_small.calfits",
        clobber=True,
    )

    # Apply calibration
    data = pyuvdata.UVData()
    data.read_ms(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_casa_calibrated.ms"
    )
    pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)
    data.reorder_pols(order="CASA", run_check=False)
    data.write_ms(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_newcal_recalibrated.ms",
        flip_conj=True,
        run_check=False,
    )


def test_single_time_recalibrate_perturbed_Mar19():

    data = pyuvdata.UVData()
    data.read_ms(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_casa_calibrated.ms"
    )
    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_model.ms")

    caldata_obj = calibration_wrappers.CalData()
    caldata_obj.load_data(data, model, min_cal_baseline_lambda=15, gain_init_stddev=0.1)
    calibration_wrappers.calibration_per_pol(
        caldata_obj,
        verbose=True,
        log_file_path="/data03/rbyrne/20231222/newcal_single_time/calibration_logs/newcal_recalibrate_perturbed_Mar19.txt",
    )
    uvcal = caldata_obj.convert_to_uvcal()
    uvcal.write_calfits(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_recalibrated_perturbed_small.calfits",
        clobber=True,
    )

    # Apply calibration
    data = pyuvdata.UVData()
    data.read_ms(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_casa_calibrated.ms"
    )
    pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)
    data.reorder_pols(order="CASA", run_check=False)
    data.write_ms(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_newcal_recalibrated_perturbed.ms",
        flip_conj=True,
        run_check=False,
    )


def test_single_time_lambda0_Mar19():

    data = pyuvdata.UVData()
    data.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_data.ms")
    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_model.ms")

    caldata_obj = calibration_wrappers.CalData()
    caldata_obj.load_data(data, model, min_cal_baseline_lambda=15, lambda_val=0.0)
    calibration_wrappers.calibration_per_pol(
        caldata_obj,
        verbose=True,
        log_file_path="/data03/rbyrne/20231222/newcal_single_time/calibration_logs/newcal_lambda0_Mar19.txt",
    )
    uvcal = caldata_obj.convert_to_uvcal()
    uvcal.write_calfits(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_lambda0.calfits",
        clobber=True,
    )

    # Apply calibration
    data = pyuvdata.UVData()
    data.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_data.ms")
    pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)
    data.reorder_pols(order="CASA", run_check=False)
    data.write_ms(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_newcal_calibrated_lambda0.ms",
        flip_conj=True,
        run_check=False,
    )


def compare_cost_values_Mar20():

    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_model.ms")
    model.select(frequency=47851562.5)
    data_recalibrated = pyuvdata.UVData()
    data_recalibrated.read_ms(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_newcal_recalibrated.ms"
    )
    data_recalibrated.select(frequency=47851562.5)
    data_calibrated = pyuvdata.UVData()
    data_calibrated.read_ms(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_newcal_calibrated.ms"
    )
    data_calibrated.select(frequency=47851562.5)
    use_pol = 0

    caldata_recalibrated = calibration_wrappers.CalData()
    caldata_recalibrated.load_data(
        data_recalibrated,
        model,
        gain_init_to_vis_ratio=False,
        min_cal_baseline_lambda=15,
        lambda_val=0.0,
    )
    recalibrated_cost = cost_function_calculations.cost_function_single_pol(
        caldata_recalibrated.gains[:, 0, use_pol],
        caldata_recalibrated.model_visibilities[0, :, 0, use_pol],
        caldata_recalibrated.data_visibilities[0, :, 0, use_pol],
        caldata_recalibrated.visibility_weights[0, :, 0, use_pol],
        caldata_recalibrated.gains_exp_mat_1,
        caldata_recalibrated.gains_exp_mat_2,
        caldata_recalibrated.lambda_val,
    )
    print(f"Recalibration run cost: {recalibrated_cost}")

    caldata_calibrated = calibration_wrappers.CalData()
    caldata_calibrated.load_data(
        data_calibrated,
        model,
        gain_init_to_vis_ratio=False,
        min_cal_baseline_lambda=15,
        lambda_val=0.0,
    )
    calibrated_cost = cost_function_calculations.cost_function_single_pol(
        caldata_calibrated.gains[:, 0, use_pol],
        caldata_calibrated.model_visibilities[0, :, 0, use_pol],
        caldata_calibrated.data_visibilities[0, :, 0, use_pol],
        caldata_calibrated.visibility_weights[0, :, 0, use_pol],
        caldata_calibrated.gains_exp_mat_1,
        caldata_calibrated.gains_exp_mat_2,
        caldata_calibrated.lambda_val,
    )
    print(f"Calibration run cost: {calibrated_cost}")


if __name__ == "__main__":
    compare_cost_values_Mar20()
