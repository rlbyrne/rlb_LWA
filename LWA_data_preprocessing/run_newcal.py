import numpy as np
import pyuvdata

# import LWA_preprocessing
import os
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
    model.select(frequencies=47851562.5)
    data_recalibrated = pyuvdata.UVData()
    data_recalibrated.read_ms(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_casa_calibrated.ms"
    )
    data_recalibrated.select(frequencies=47851562.5)
    data_calibrated = pyuvdata.UVData()
    data_calibrated.read_ms(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_newcal_calibrated.ms"
    )
    data_calibrated.select(frequencies=47851562.5)
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


def debug_recalibration_Mar20():

    data = pyuvdata.UVData()
    data.read_ms(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_newcal_calibrated.ms"
    )
    data.select(frequencies=47851562.5, polarizations=-5)
    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_model.ms")
    model.select(frequencies=47851562.5, polarizations=-5)

    for recalibration_iter in [1, 2]:

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(
            data,
            model,
            min_cal_baseline_lambda=15,
            gain_init_to_vis_ratio=False,
        )

        iter = 1
        while iter < 20:
            calibration_wrappers.calibration_per_pol(
                caldata_obj,
                verbose=False,
                maxiter=1,
                parallel=False,
            )
            calibrated_cost = cost_function_calculations.cost_function_single_pol(
                caldata_obj.gains[:, 0, 0],
                caldata_obj.model_visibilities[0, :, 0, 0],
                caldata_obj.data_visibilities[0, :, 0, 0],
                caldata_obj.visibility_weights[0, :, 0, 0],
                caldata_obj.gains_exp_mat_1,
                caldata_obj.gains_exp_mat_2,
                caldata_obj.lambda_val,
            )
            print(
                f"Recalibration iteration {recalibration_iter}, optimization iteration {iter}, cost: {calibrated_cost}"
            )
            iter += 1

        uvcal = caldata_obj.convert_to_uvcal()

        # Apply calibration
        data = pyuvdata.UVData()
        data.read_ms(
            "/data03/rbyrne/20231222/newcal_single_time/cal46_small_newcal_calibrated.ms"
        )
        data.select(frequencies=47851562.5, polarizations=-5)
        pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)


def test_calibration_convergence():

    data = pyuvdata.UVData()
    data.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_data.ms")
    data.select(frequencies=47851562.5, polarizations=-5)
    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_model.ms")
    model.select(frequencies=47851562.5, polarizations=-5)

    caldata_obj = calibration_wrappers.CalData()
    caldata_obj.load_data(
        data,
        model,
        min_cal_baseline_lambda=15,
        gain_init_to_vis_ratio=True,
        lambda_val=0,
    )

    # Calibrate with newcal
    calibration_wrappers.calibration_per_pol(
        caldata_obj,
        verbose=False,
        parallel=False,
    )

    # Print resulting cost
    calibrated_cost = cost_function_calculations.cost_function_single_pol(
        caldata_obj.gains[:, 0, 0],
        caldata_obj.model_visibilities[0, :, 0, 0],
        caldata_obj.data_visibilities[0, :, 0, 0],
        caldata_obj.visibility_weights[0, :, 0, 0],
        caldata_obj.gains_exp_mat_1,
        caldata_obj.gains_exp_mat_2,
        caldata_obj.lambda_val,
    )
    print(f"Newcal cost: {calibrated_cost}")
    print(np.shape(caldata_obj.gains))

    # Compare to recalibrated result:
    caldata_obj_recalibrated = calibration_wrappers.CalData()
    caldata_obj_recalibrated.load_data(
        data,
        model,
        min_cal_baseline_lambda=15,
        gain_init_calfile="/data03/rbyrne/20231222/newcal_single_time/cal46_recalibrated_small.calfits",
        lambda_val=0,
    )
    print(np.shape(caldata_obj_recalibrated.gains))
    caldata_obj.gains = caldata_obj_recalibrated.gains
    recalibrated_cost = cost_function_calculations.cost_function_single_pol(
        caldata_obj.gains[:, 0, 0],
        caldata_obj.model_visibilities[0, :, 0, 0],
        caldata_obj.data_visibilities[0, :, 0, 0],
        caldata_obj.visibility_weights[0, :, 0, 0],
        caldata_obj.gains_exp_mat_1,
        caldata_obj.gains_exp_mat_2,
        caldata_obj.lambda_val,
    )
    print(f"Cost with recalibrated gains: {recalibrated_cost}")


def test_calibration_application():

    data = pyuvdata.UVData()
    data.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_data.ms")
    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_model.ms")

    caldata_obj = calibration_wrappers.CalData()
    caldata_obj.load_data(
        data,
        model,
        min_cal_baseline_lambda=15,
        gain_init_to_vis_ratio=True,
        lambda_val=0,
    )

    # Calibrate with newcal
    calibration_wrappers.calibration_per_pol(
        caldata_obj,
        verbose=True,
        parallel=False,
        log_file_path=None,
        get_crosspol_phase=False,
    )

    print("Calibration completed")
    print("Evaluating cost")

    # Print resulting cost
    test_freq_channel = 96
    test_pol_ind = 0
    calibrated_cost = cost_function_calculations.cost_function_single_pol(
        caldata_obj.gains[:, test_freq_channel, test_pol_ind],
        caldata_obj.model_visibilities[0, :, test_freq_channel, test_pol_ind],
        caldata_obj.data_visibilities[0, :, test_freq_channel, test_pol_ind],
        caldata_obj.visibility_weights[0, :, test_freq_channel, test_pol_ind],
        caldata_obj.gains_exp_mat_1,
        caldata_obj.gains_exp_mat_2,
        caldata_obj.lambda_val,
    )
    print(f"Newcal cost: {calibrated_cost}")

    uvcal = caldata_obj.convert_to_uvcal()
    uvcal.write_calfits(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_debug.calfits",
        clobber=True,
    )

    # Apply calibration
    data = pyuvdata.UVData()
    data.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_data.ms")
    pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)
    # data.reorder_pols(order="CASA", run_check=False)
    # data.write_ms(
    #    "/data03/rbyrne/20231222/newcal_single_time/cal46_small_newcal_calibrated_debug.ms",
    #    flip_conj=True,
    #    run_check=False,
    # )

    # Reread calibrated data
    # data = pyuvdata.UVData()
    # data.read_ms(
    #    "/data03/rbyrne/20231222/newcal_single_time/cal46_small_newcal_calibrated_debug.ms"
    # )
    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_model.ms")

    caldata_obj = calibration_wrappers.CalData()
    caldata_obj.load_data(
        data,
        model,
        min_cal_baseline_lambda=15,
        gain_init_to_vis_ratio=True,
        lambda_val=0,
    )
    calibrated_cost = cost_function_calculations.cost_function_single_pol(
        caldata_obj.gains[:, test_freq_channel, test_pol_ind],
        caldata_obj.model_visibilities[0, :, test_freq_channel, test_pol_ind],
        caldata_obj.data_visibilities[0, :, test_freq_channel, test_pol_ind],
        caldata_obj.visibility_weights[0, :, test_freq_channel, test_pol_ind],
        caldata_obj.gains_exp_mat_1,
        caldata_obj.gains_exp_mat_2,
        caldata_obj.lambda_val,
    )
    print(f"Newcal cost, reread: {calibrated_cost}")


def test_calibration_application_2():

    data = pyuvdata.UVData()
    data.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_data.ms")
    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_model.ms")

    caldata_obj = calibration_wrappers.CalData()
    caldata_obj.load_data(
        data,
        model,
        min_cal_baseline_lambda=15,
        gain_init_to_vis_ratio=True,
        lambda_val=0,
    )

    # Calibrate with newcal
    calibration_wrappers.calibration_per_pol(
        caldata_obj,
        verbose=True,
        parallel=False,
        log_file_path=None,
        get_crosspol_phase=False,
    )

    print("Calibration completed")
    print("Evaluating cost")

    # Print resulting cost
    test_freq_channel = 96
    test_pol_ind = 0
    calibrated_cost = cost_function_calculations.cost_function_single_pol(
        caldata_obj.gains[:, test_freq_channel, test_pol_ind],
        caldata_obj.model_visibilities[0, :, test_freq_channel, test_pol_ind],
        caldata_obj.data_visibilities[0, :, test_freq_channel, test_pol_ind],
        caldata_obj.visibility_weights[0, :, test_freq_channel, test_pol_ind],
        caldata_obj.gains_exp_mat_1,
        caldata_obj.gains_exp_mat_2,
        caldata_obj.lambda_val,
    )
    print(f"Newcal cost: {calibrated_cost}")

    # Apply calibration
    gains_expanded = np.matmul(
        caldata_obj.gains_exp_mat_1,
        caldata_obj.gains[:, test_freq_channel, test_pol_ind],
    ) * np.matmul(
        caldata_obj.gains_exp_mat_2,
        np.conj(caldata_obj.gains[:, test_freq_channel, test_pol_ind]),
    )
    caldata_obj.data_visibilities[
        0, :, test_freq_channel, test_pol_ind
    ] *= gains_expanded
    print(np.sum(caldata_obj.data_visibilities[0, :, test_freq_channel, test_pol_ind]))

    uvcal = caldata_obj.convert_to_uvcal()
    uvcal.write_calfits(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_debug.calfits",
        clobber=True,
    )

    # Apply calibration
    data = pyuvdata.UVData()
    data.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_data.ms")
    pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)
    data.reorder_pols(order="CASA", run_check=False)
    data.write_ms(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_newcal_calibrated_debug.ms",
        flip_conj=False,
        run_check=False,
        clobber=True,
    )

    # Reread calibrated data
    data = pyuvdata.UVData()
    data.read_ms(
        "/data03/rbyrne/20231222/newcal_single_time/cal46_small_newcal_calibrated_debug.ms"
    )
    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_model.ms")

    caldata_obj = calibration_wrappers.CalData()
    caldata_obj.load_data(
        data,
        model,
        min_cal_baseline_lambda=15,
        gain_init_to_vis_ratio=False,
        lambda_val=0,
    )
    calibrated_cost = cost_function_calculations.cost_function_single_pol(
        caldata_obj.gains[:, test_freq_channel, test_pol_ind],
        caldata_obj.model_visibilities[0, :, test_freq_channel, test_pol_ind],
        caldata_obj.data_visibilities[0, :, test_freq_channel, test_pol_ind],
        caldata_obj.visibility_weights[0, :, test_freq_channel, test_pol_ind],
        caldata_obj.gains_exp_mat_1,
        caldata_obj.gains_exp_mat_2,
        caldata_obj.lambda_val,
    )
    print(f"Newcal cost, reread: {calibrated_cost}")

    print(np.sum(caldata_obj.data_visibilities[0, :, test_freq_channel, test_pol_ind]))


def test_calibration_full_band_Mar27():

    data = pyuvdata.UVData()
    data.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_data.ms")
    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/newcal_single_time/cal46_small_model.ms")

    caldata_obj = calibration_wrappers.CalData()
    caldata_obj.load_data(
        data,
        model,
        min_cal_baseline_lambda=15,
        gain_init_to_vis_ratio=True,
        lambda_val=100,
    )

    # Calibrate with newcal
    calibration_wrappers.calibration_per_pol(
        caldata_obj,
        verbose=True,
        parallel=True,
        log_file_path="/data03/rbyrne/20231222/newcal_single_time/newcal_log_Mar27.txt",
        get_crosspol_phase=True,
    )

    print("Calibration completed")
    print("Evaluating cost")

    # Print resulting cost
    test_freq_channel = 96
    test_pol_ind = 0
    calibrated_cost = cost_function_calculations.cost_function_single_pol(
        caldata_obj.gains[:, test_freq_channel, test_pol_ind],
        caldata_obj.model_visibilities[0, :, test_freq_channel, test_pol_ind],
        caldata_obj.data_visibilities[0, :, test_freq_channel, test_pol_ind],
        caldata_obj.visibility_weights[0, :, test_freq_channel, test_pol_ind],
        caldata_obj.gains_exp_mat_1,
        caldata_obj.gains_exp_mat_2,
        caldata_obj.lambda_val,
    )
    print(f"Newcal cost: {calibrated_cost}")

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
        flip_conj=False,
        run_check=False,
        clobber=True,
    )


def casa_calibration_comparison_Mar28():

    data = pyuvdata.UVData()
    data.read_ms("/data03/rbyrne/20231222/cal46.ms", data_column="DATA")
    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/cal46.ms", data_column="MODEL_DATA")

    caldata_obj = calibration_wrappers.CalData()
    caldata_obj.load_data(data, model, min_cal_baseline_lambda=15)
    calibration_wrappers.calibration_per_pol(
        caldata_obj,
        verbose=True,
        log_file_path="/data03/rbyrne/20231222/newcal_calibration/calibration_log_Mar28.txt",
    )
    uvcal = caldata_obj.convert_to_uvcal()
    uvcal.write_calfits(
        "/data03/rbyrne/20231222/newcal_calibration/cal46_min_bl_15.calfits",
        clobber=True,
    )


def apply_calibration_Mar29():

    cal = pyuvdata.UVCal()
    cal.read("/data03/rbyrne/20231222/newcal_calibration/cal46_min_bl_15.calfits")
    filenames = [
        "46time1",
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
            flip_conj=False,
            run_check=False,
            clobber=True,
        )


def test_new_wrapper_Apr8():

    uvcal = calibration_wrappers.calibration_per_pol(
        "/data03/rbyrne/20231222/cal46.ms",
        "/data03/rbyrne/20231222/cal46.ms",
        min_cal_baseline_lambda=15,
        verbose=True,
        log_file_path="/data03/rbyrne/20231222/newcal_calibration/calibration_log_Apr8.txt",
    )
    uvcal.write_calfits(
        "/data03/rbyrne/20231222/newcal_calibration/cal46_new_wrapper_test_Apr8.calfits",
        clobber=True,
    )


def test_uvcal_conversion():

    data = pyuvdata.UVData()
    data.read_ms("/data03/rbyrne/20231222/cal46.ms", data_column="DATA")
    model = pyuvdata.UVData()
    model.read_ms("/data03/rbyrne/20231222/cal46.ms", data_column="MODEL_DATA")

    caldata_obj = calibration_wrappers.CalData()
    caldata_obj.load_data(data, model, min_cal_baseline_lambda=15)
    uvcal = caldata_obj.convert_to_uvcal()


def test_crosspol_phase_flip_Apr9():

    uvcal = calibration_wrappers.calibration_per_pol(
        "/data03/rbyrne/20231222/cal46.ms",
        "/data03/rbyrne/20231222/cal46.ms",
        min_cal_baseline_lambda=15,
        verbose=True,
        log_file_path="/data03/rbyrne/20231222/newcal_calibration/calibration_log_Apr9.txt",
    )
    uvcal.write_calfits(
        "/data03/rbyrne/20231222/newcal_calibration/cal46_flipped_phase_test_Apr9.calfits",
        clobber=True,
    )


def apply_phase_flip_calibration_Apr9():

    cal = pyuvdata.UVCal()
    cal.read(
        "/data03/rbyrne/20231222/newcal_calibration/cal46_flipped_phase_test_Apr9.calfits"
    )
    filenames = [
        "46time1",
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
            f"/data03/rbyrne/20231222/newcal_calibration/{file}_newcal_phase_flip.ms",
            flip_conj=False,
            run_check=False,
            clobber=True,
        )


def flip_model_conjugation_Apr16():

    skymodel_names = [
        "cyg_cas_sim",
        "deGasperin_cyg_cas_sim",
        "deGasperin_cyg_cas_sim_NMbeam",
        "deGasperin_sources_sim",
        "VLSS_sim",
        "mmode_with_cyg_cas_sim",
        "mmode_with_deGasperin_cyg_cas_sim",
    ]

    for use_skymodel in skymodel_names:

        model = pyuvdata.UVData()
        model.read_uvfits(
            f"/data03/rbyrne/20231222/simulation_outputs/cal46_time11_{use_skymodel}.uvfits"
        )
        model.data_array = np.conj(model.data_array)
        model.reorder_pols(order="CASA", run_check=False)
        model.write_ms(
            f"/data03/rbyrne/20231222/simulation_outputs/cal46_time11_{use_skymodel}.ms",
            flip_conj=True,
            run_check=False,
            clobber=True,
        )


def test_skymodels_Apr15():

    skymodel_names = [
        "cyg_cas_sim",
        "deGasperin_cyg_cas_sim",
        "deGasperin_cyg_cas_sim_NMbeam",
        "deGasperin_sources_sim",
        "VLSS_sim",
        "mmode_with_cyg_cas_sim",
        "mmode_with_deGasperin_cyg_cas_sim",
    ]
    data_file = "/data03/rbyrne/20231222/simulation_outputs/cal46_time11.ms"

    for use_skymodel in skymodel_names:

        uvcal = calibration_wrappers.calibration_per_pol(
            data_file,
            f"/data03/rbyrne/20231222/simulation_outputs/cal46_time11_{use_skymodel}.ms",
            data_use_column="DATA",
            model_use_column="DATA",
            min_cal_baseline_lambda=10,
            verbose=True,
            log_file_path=f"/data03/rbyrne/20231222/skymodel_testing/calibration_log_{use_skymodel}_Apr16.txt",
        )
        uvcal.write_calfits(
            f"/data03/rbyrne/20231222/skymodel_testing/newcal_{use_skymodel}.calfits",
            clobber=True,
        )
        data = pyuvdata.UVData()
        data.read_ms(data_file, data_column="DATA")
        pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)
        data.write_uvfits(
            f"/data03/rbyrne/20231222/skymodel_testing/cal46_time11_newcal_{use_skymodel}.uvfits"
        )


def test_orig_skymodel():

    data_file = "/data03/rbyrne/20231222/simulation_outputs/cal46_time11.ms"

    uvcal = calibration_wrappers.calibration_per_pol(
        data_file,
        data_file,
        data_use_column="DATA",
        model_use_column="MODEL_DATA",
        min_cal_baseline_lambda=10,
        verbose=True,
        log_file_path=f"/data03/rbyrne/20231222/skymodel_testing/calibration_log_orig_skymodel_Apr17.txt",
    )
    uvcal.write_calfits(
        f"/data03/rbyrne/20231222/skymodel_testing/newcal_orig_skymodel.calfits",
        clobber=True,
    )
    data = pyuvdata.UVData()
    data.read_ms(data_file, data_column="DATA")
    pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)
    data.write_uvfits(
        f"/data03/rbyrne/20231222/skymodel_testing/cal46_time11_newcal_orig_skymodel.uvfits"
    )


def image_calibrated_data():
    data_dir = "/data03/rbyrne/20231222/skymodel_testing"
    skymodel_names = [
        "cyg_cas_sim",
        "deGasperin_cyg_cas_sim",
        "deGasperin_cyg_cas_sim_NMbeam",
        "deGasperin_sources_sim",
        "VLSS_sim",
        "mmode_with_cyg_cas_sim",
        "mmode_with_deGasperin_cyg_cas_sim",
        "orig_skymodel",
    ]
    for skymodel in skymodel_names:
        data = pyuvdata.UVData()
        data.read_uvfits(f"{data_dir}/cal46_time11_newcal_{skymodel}.uvfits")
        data.reorder_pols(order="CASA", run_check=False)
        data.data_array = np.conj(data.data_array)  # Force conjugation
        data.write_ms(
            f"{data_dir}/cal46_time11_newcal_{skymodel}.ms",
            flip_conj=False,
            run_check=False,
            clobber=True,
        )
        os.system(
            f"/opt/bin/wsclean -pol IV -multiscale -multiscale-scale-bias 0.8 -size 4096 4096 -scale 0.03125 -niter 100 -taper-inner-tukey 30 -mgain 0.85 -weight briggs 0 -no-update-model-required -mem 10 -no-reorder -name {data_dir}/wsclean_images/cal46_time11_newcal_{skymodel} {data_dir}/cal46_time11_newcal_{skymodel}.ms"
        )


def test_pyuvsim_calibration_Apr18():

    uvcal = calibration_wrappers.calibration_per_pol(
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_cyg_cas_sim.ms",
        data_use_column="DATA",
        model_use_column="DATA",
        min_cal_baseline_lambda=10,
        verbose=True,
        log_file_path=f"/data03/rbyrne/20231222/test_pyuvsim_modeling/calibration_log_Apr18.txt",
    )
    uvcal.write_calfits(
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_newcal_cyg_cas.calfits",
        clobber=True,
    )
    data = pyuvdata.UVData()
    data.read_ms(
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj.ms",
        data_column="DATA",
    )
    pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)
    data.reorder_pols(order="CASA")
    data.write_ms(
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_newcal_cyg_cas.ms"
    )


def convert_uvfits_to_ms_Apr19():

    filedir = "/data03/rbyrne/20231222/test_pyuvsim_modeling"
    filenames = [
        "cal46_time11_conj_deGasperin_cyg_cas_sim_NMbeam",
        "cal46_time11_conj_deGasperin_cyg_cas_sim",
        "cal46_time11_conj_deGasperin_sources_sim",
        "cal46_time11_conj_mmode_sim",
        "cal46_time11_conj_VLSS_sim",
    ]
    for use_file in filenames:
        # uv = pyuvdata.UVData()
        # uv.read_uvfits(f"{filedir}/{use_file}.uvfits")
        # uv.reorder_pols(order="CASA")
        # uv.write_ms(f"{filedir}/{use_file}.ms")
        # Image with WSClean
        os.system(
            f"/opt/bin/wsclean -pol IV -multiscale -multiscale-scale-bias 0.8 -size 4096 4096 -scale 0.03125 -niter 0 -taper-inner-tukey 30 -mgain 0.85 -weight briggs 0 -no-update-model-required -mem 10 -no-reorder -name {filedir}/{use_file} {filedir}/{use_file}.ms"
        )


def test_sky_models_May9():

    datafile = "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj.ms"
    model_files = [
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/orig_model.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_VLSS_sim.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_deGasperin_sources_sim.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_deGasperin_cyg_cas_sim.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_deGasperin_cyg_cas_sim_NMbeam.ms",
    ]
    model_names = [
        "orig",
        "VLSS",
        "deGasperin_sources",
        "deGasperin_cyg_cas",
        "deGasperin_cyg_cas_NMbeam",
    ]

    for model_ind, model_file in enumerate(model_files):
        uvcal = calibration_wrappers.calibration_per_pol(
            datafile,
            model_file,
            data_use_column="DATA",
            model_use_column="DATA",
            min_cal_baseline_lambda=10,
            verbose=True,
            log_file_path=f"/data03/rbyrne/20231222/test_pyuvsim_modeling/calibration_logs/cal_log_{model_names[model_ind]}_May9.txt",
        )
        uvcal.write_calfits(
            f"/data03/rbyrne/20231222/test_pyuvsim_modeling/calfits_files/cal46_time11_newcal_{model_names[model_ind]}.calfits",
            clobber=True,
        )
        data = pyuvdata.UVData()
        data.read_ms(
            datafile,
            data_column="DATA",
        )
        pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)
        data.reorder_pols(order="CASA")
        data.write_ms(
            f"/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_newcal_{model_names[model_ind]}.ms",
            fix_autos=True,
        )


def test_mmode_models_May9():

    datafile = "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj.ms"
    source_simulation = (
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_cyg_cas_sim.ms"
    )
    model_files = [
        # "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_mmode_sim.ms",
        "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_matvis_sim.ms",
        "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_matvis_sim_nside512.ms",
    ]
    model_names = [
        # "mmode_with_cyg_cas_pyuvsim_nside128",
        "mmode_with_cyg_cas_matvis_nside128",
        "mmode_with_cyg_cas_matvis_nside512",
    ]

    for model_ind, model_file in enumerate(model_files):

        # Combine mmode map with Cyg and Cas models
        map_path = "/".join(model_file.split("/")[:-1])
        combined_map_name = f"{map_path}/cal46_time11_conj_{model_names}_sim.ms"
        mmode = pyuvdata.UVData()
        mmode.read(model_file)
        sources = pyuvdata.UVData()
        sources.read(source_simulation)
        mmode.filename = [""]
        sources.filename = [""]
        mmode.phase_to_time(np.mean(mmode.time_array))
        sources.phase_to_time(np.mean(mmode.time_array))
        mmode.reorder_blts()
        sources.reorder_blts()
        mmode.reorder_pols(order="AIPS")
        sources.reorder_pols(order="AIPS")
        mmode.reorder_freqs(channel_order="freq")
        sources.reorder_freqs(channel_order="freq")
        mmode.filename = [""]
        sources.filename = [""]
        mmode.sum_vis(
            sources,
            inplace=True,
            override_params=[
                "phase_center_frame_pa",
                "scan_number_array",
                "antenna_names",
                "telescope_location",
                "phase_center_app_ra",
                "phase_center_app_dec",
                "instrument",
                "time_array",
                "phase_center_id_array",
            ],
        )
        mmode.reorder_pols(order="CASA")
        mmode.write_ms(combined_map_name, fix_autos=True, clobber=True)

        uvcal = calibration_wrappers.calibration_per_pol(
            datafile,
            combined_map_name,
            data_use_column="DATA",
            model_use_column="DATA",
            min_cal_baseline_lambda=10,
            verbose=True,
            log_file_path=f"{map_path}/calibration_logs/cal_log_{model_names[model_ind]}_May9.txt",
        )
        uvcal.write_calfits(
            f"{map_path}/calfits_files/cal46_time11_newcal_{model_names[model_ind]}.calfits",
            clobber=True,
        )
        data = pyuvdata.UVData()
        data.read_ms(
            datafile,
            data_column="DATA",
        )
        pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)
        data.reorder_pols(order="CASA")
        data.write_ms(
            f"{map_path}/cal46_time11_newcal_{model_names[model_ind]}.ms",
            fix_autos=True,
            clobber=True,
        )


if __name__ == "__main__":
    test_mmode_models_May9()
