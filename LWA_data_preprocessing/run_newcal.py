import numpy as np
import pyuvdata
import sys

import LWA_preprocessing
import os
from calico import (
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
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_mmode_sim.ms",
        "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_matvis_sim.ms",
        "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_matvis_sim_nside512.ms",
    ]
    model_names = [
        "mmode_with_cyg_cas_pyuvsim_nside128",
        "mmode_with_cyg_cas_matvis_nside128",
        "mmode_with_cyg_cas_matvis_nside512",
    ]

    for model_ind, model_file in enumerate(model_files):

        # Combine mmode map with Cyg and Cas models
        map_path = "/".join(model_file.split("/")[:-1])
        combined_map_name = (
            f"{map_path}/cal46_time11_conj_{model_names[model_ind]}_sim.ms"
        )
        mmode = pyuvdata.UVData()
        mmode.read(model_file)
        sources = pyuvdata.UVData()
        sources.read(source_simulation)
        mmode.filename = [""]
        sources.filename = [""]
        mmode.phase_to_time(np.mean(mmode.time_array))
        sources.phase_to_time(np.mean(mmode.time_array))

        mmode.conjugate_bls()
        sources.conjugate_bls()

        # mmode map contains duplicate baselines
        mmode_baselines = list(zip(mmode.ant_1_array, mmode.ant_2_array))
        keep_baselines = []
        for bl_ind, baseline in enumerate(mmode_baselines):
            if baseline not in mmode_baselines[:bl_ind]:
                keep_baselines.append(bl_ind)
        mmode.select(blt_inds=keep_baselines)

        # assign antenna names
        for ant_ind in range(len(mmode.antenna_names)):
            mmode.antenna_names[ant_ind] = sources.antenna_names[
                np.where(sources.antenna_numbers == mmode.antenna_numbers[ant_ind])[0][
                    0
                ]
            ]

        mmode_baselines = list(set(list(zip(mmode.ant_1_array, mmode.ant_2_array))))
        sources_baselines = list(
            set(list(zip(sources.ant_1_array, sources.ant_2_array)))
        )
        use_baselines = [
            baseline
            for baseline in mmode_baselines
            if (baseline in sources_baselines) or (baseline[::-1] in sources_baselines)
        ]

        mmode.select(bls=use_baselines)
        sources.select(bls=use_baselines)
        mmode.reorder_blts()
        sources.reorder_blts()
        mmode.reorder_pols(order="AIPS")
        sources.reorder_pols(order="AIPS")
        mmode.reorder_freqs(channel_order="freq")
        sources.reorder_freqs(channel_order="freq")
        mmode.sum_vis(
            sources,
            inplace=True,
            override_params=[
                "antenna_diameters",
                "integration_time",
                "lst_array",
                "uvw_array",
                "phase_center_id_array",
                "phase_center_app_ra",
                "phase_center_app_dec",
                "phase_center_frame_pa",
                "phase_center_catalog",
                "telescope_location",
                "telescope_name",
                "instrument",
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


def test_mmode_models_May15():

    datafile = "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj.ms"
    file_directories = [
        "/data03/rbyrne/20231222/test_pyuvsim_modeling",
        "/data03/rbyrne/20231222/matvis_modeling",
        "/data03/rbyrne/20231222/matvis_modeling",
    ]
    model_files = [
        "cal46_time11_conj_mmode_with_cyg_cas_pyuvsim_nside128_sim.ms",
        "cal46_time11_conj_mmode_with_cyg_cas_matvis_nside128_sim.ms",
        "cal46_time11_conj_mmode_with_cyg_cas_matvis_nside512_sim.ms",
    ]
    model_names = [
        "mmode_with_cyg_cas_pyuvsim_nside128",
        "mmode_with_cyg_cas_matvis_nside128",
        "mmode_with_cyg_cas_matvis_nside512",
    ]

    for model_ind, model_file in enumerate(model_files):
        uvcal = calibration_wrappers.calibration_per_pol(
            datafile,
            f"{file_directories[model_ind]}/{model_file}",
            data_use_column="DATA",
            model_use_column="DATA",
            min_cal_baseline_lambda=10,
            verbose=True,
            log_file_path=f"{file_directories[model_ind]}/calibration_logs/cal_log_{model_names[model_ind]}_May9.txt",
        )
        uvcal.write_calfits(
            f"{file_directories[model_ind]}/calfits_files/cal46_time11_newcal_{model_names[model_ind]}.calfits",
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
            f"{file_directories[model_ind]}/cal46_time11_newcal_{model_names[model_ind]}.ms",
            fix_autos=True,
            clobber=True,
        )


def test_mmode_models_May16():

    datafile = "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj.ms"
    file_directories = [
        "/data03/rbyrne/20231222/test_pyuvsim_modeling",
        "/data03/rbyrne/20231222/matvis_modeling",
        "/data03/rbyrne/20231222/matvis_modeling",
    ]
    model_files = [
        "cal46_time11_conj_mmode_sim.ms",
        "cal46_time11_conj_mmode_matvis_sim_reformatted.ms",
        "cal46_time11_conj_mmode_matvis_sim_nside512_reformatted.ms",
    ]
    model_names = [
        "mmode_pyuvsim_nside128",
        "mmode_matvis_nside128",
        "mmode_matvis_nside512",
    ]

    for model_ind, model_file in enumerate(model_files):
        uvcal = calibration_wrappers.calibration_per_pol(
            datafile,
            f"{file_directories[model_ind]}/{model_file}",
            data_use_column="DATA",
            model_use_column="DATA",
            min_cal_baseline_lambda=10,
            verbose=True,
            log_file_path=f"{file_directories[model_ind]}/calibration_logs/cal_log_{model_names[model_ind]}_May16.txt",
        )
        uvcal.write_calfits(
            f"{file_directories[model_ind]}/calfits_files/cal46_time11_newcal_{model_names[model_ind]}.calfits",
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
            f"{file_directories[model_ind]}/cal46_time11_newcal_{model_names[model_ind]}.ms",
            fix_autos=True,
            clobber=True,
        )


def calculate_mmode_flux_offset():

    calibrated_data_path = (
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_newcal_cyg_cas.ms"
    )
    mmode_sim_path = "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_matvis_sim_nside512_reformatted.ms"

    data = pyuvdata.UVData()
    data.read(calibrated_data_path)
    model = pyuvdata.UVData()
    model.read(mmode_sim_path)

    model_baselines = list(set(list(zip(model.ant_1_array, model.ant_2_array))))
    data_baselines = list(set(list(zip(data.ant_1_array, data.ant_2_array))))
    use_baselines = [
        baseline
        for baseline in model_baselines
        if (baseline in data_baselines) or (baseline[::-1] in data_baselines)
    ]
    data.select(bls=use_baselines, polarizations=[-5, -6])
    model.select(bls=use_baselines, polarizations=[-5, -6])
    data.phase_to_time(np.mean(data.time_array))
    model.phase_to_time(np.mean(data.time_array))
    # Ensure ordering matches
    data.reorder_blts()
    model.reorder_blts()
    data.reorder_pols(order="AIPS")
    model.reorder_pols(order="AIPS")
    data.reorder_freqs(channel_order="freq")
    model.reorder_freqs(channel_order="freq")
    data.filename = [""]
    model.filename = [""]

    model.data_array[np.where(model.flag_array)] = 0.0
    data.data_array[np.where(data.flag_array)] = 0.0

    diff = data.sum_vis(
        model,
        difference=True,
        inplace=False,
        override_params=[
            "antenna_diameters",
            "integration_time",
            "lst_array",
            "uvw_array",
            "phase_center_id_array",
            "phase_center_app_ra",
            "phase_center_app_dec",
            "phase_center_frame_pa",
            "phase_center_catalog",
            "telescope_location",
            "telescope_name",
            "instrument",
            "flag_array",
            "nsample_array",
            "scan_number_array",
        ],
    )
    print(np.sum(np.abs(diff.data_array) ** 2.0))

    offset_value = np.sum(
        np.real(np.conj(model.data_array) * data.data_array)
    ) / np.sum(np.abs(model.data_array) ** 2.0)
    print(offset_value)

    model.data_array *= offset_value
    diff = data.sum_vis(
        model,
        difference=True,
        inplace=False,
        override_params=[
            "antenna_diameters",
            "integration_time",
            "lst_array",
            "uvw_array",
            "phase_center_id_array",
            "phase_center_app_ra",
            "phase_center_app_dec",
            "phase_center_frame_pa",
            "phase_center_catalog",
            "telescope_location",
            "telescope_name",
            "instrument",
            "flag_array",
            "nsample_array",
            "scan_number_array",
        ],
    )
    print(np.sum(np.abs(diff.data_array) ** 2.0))


def run_newcal_with_mmode_amp_offset_May17():

    datafile = "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj.ms"
    file_directories = [
        "/data03/rbyrne/20231222/test_pyuvsim_modeling",
        "/data03/rbyrne/20231222/matvis_modeling",
        "/data03/rbyrne/20231222/matvis_modeling",
    ]
    model_files = [
        "cal46_time11_conj_mmode_amp_offset_with_cyg_cas_pyuvsim_nside128_sim.ms",
        "cal46_time11_conj_mmode_amp_offset_with_cyg_cas_matvis_nside128_sim.ms",
        "cal46_time11_conj_mmode_amp_offset_with_cyg_cas_matvis_nside512_sim.ms",
    ]
    model_names = [
        "mmode_amp_offset_with_cyg_cas_pyuvsim_nside128",
        "mmode_amp_offset_with_cyg_cas_matvis_nside128",
        "mmode_amp_offset_with_cyg_cas_matvis_nside512",
    ]

    for model_ind, model_file in enumerate(model_files):
        uvcal = calibration_wrappers.calibration_per_pol(
            datafile,
            f"{file_directories[model_ind]}/{model_file}",
            data_use_column="DATA",
            model_use_column="DATA",
            min_cal_baseline_lambda=10,
            verbose=True,
            log_file_path=f"{file_directories[model_ind]}/calibration_logs/cal_log_{model_names[model_ind]}_May17.txt",
        )
        uvcal.write_calfits(
            f"{file_directories[model_ind]}/calfits_files/cal46_time11_newcal_{model_names[model_ind]}.calfits",
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
            f"{file_directories[model_ind]}/cal46_time11_newcal_{model_names[model_ind]}.ms",
            fix_autos=True,
            clobber=True,
        )


def run_newcal_Jun3():

    datafile = "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj.ms"
    model_files = [
        # "/data03/rbyrne/20231222/test_diffuse_normalization/cal46_time11_conj_deGasperin_cyg_cas_48MHz_sim.uvfits",
        "/data03/rbyrne/20231222/test_diffuse_normalization/cal46_time11_conj_deGasperin_cyg_cas_48MHz_with_mmode_sim.uvfits",
    ]
    model_names = [
        # "deGasperin_cyg_cas_48MHz",
        "deGasperin_cyg_cas_48MHz_with_mmode",
    ]
    output_directory = "/data03/rbyrne/20231222/test_diffuse_normalization"

    for model_ind, model_file in enumerate(model_files):
        uvcal = calibration_wrappers.calibration_per_pol(
            datafile,
            model_file,
            data_use_column="DATA",
            min_cal_baseline_lambda=10,
            verbose=True,
            log_file_path=f"{output_directory}/calibration_logs/cal_log_{model_names[model_ind]}_Jun4.txt",
        )
        uvcal.write_calfits(
            f"{output_directory}/calfits_files/cal46_time11_newcal_{model_names[model_ind]}.calfits",
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
            f"{output_directory}/cal46_time11_newcal_{model_names[model_ind]}.ms",
            fix_autos=True,
            clobber=True,
        )


def run_newcal_Jun12():

    datafile = "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj.ms"
    model_files = [
        "/data03/rbyrne/20231222/test_diffuse_normalization/cal46_time11_conj_deGasperin_sources_plus_48MHz_with_mmode_sim.ms",
    ]
    model_names = [
        "deGasperin_sources_plus_48MHz_with_mmode",
    ]
    output_directory = "/data03/rbyrne/20231222/test_diffuse_normalization"

    for model_ind, model_file in enumerate(model_files):
        uvcal = calibration_wrappers.calibration_per_pol(
            datafile,
            model_file,
            data_use_column="DATA",
            model_use_column="DATA",
            min_cal_baseline_lambda=10,
            verbose=True,
            log_file_path=f"{output_directory}/calibration_logs/cal_log_{model_names[model_ind]}_Jun12.txt",
        )
        uvcal.write_calfits(
            f"{output_directory}/calfits_files/cal46_time11_newcal_{model_names[model_ind]}.calfits",
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
            f"{output_directory}/cal46_time11_newcal_{model_names[model_ind]}.ms",
            fix_autos=True,
            clobber=True,
        )


def run_newcal_Jul29():

    datafile = "/lustre/gh/2024-03-02/calibration/ruby/46.ms"
    model_file = (
        "/lustre/rbyrne/2024-03-02/calibration_models/46_deGasperin_point_sources.ms"
    )

    uvcal = calibration_wrappers.calibration_per_pol(
        datafile,
        model_file,
        data_use_column="DATA",
        model_use_column="DATA",
        conjugate_model=True,
        min_cal_baseline_lambda=10,
        max_cal_baseline_lambda=125,
        verbose=True,
        get_crosspol_phase=False,
        log_file_path="/lustre/rbyrne/2024-03-02/calibration_outputs/48_cal_log_v1.txt",
    )
    uvcal.write_calfits(
        "/lustre/rbyrne/2024-03-02/calibration_outputs/48_v2.calfits",
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
        "/lustre/rbyrne/2024-03-02/calibration_outputs/48_calibrated.ms",
        fix_autos=True,
        clobber=True,
    )


def run_newcal_Jul31():

    use_filenames = [
        "18",
        "23",
        "27",
        "36",
        "41",
        # "46",
        "50",
        "55",
        "59",
        "64",
        "73",
        "78",
        "82",
    ]

    for file_name in use_filenames:

        datafile = f"/lustre/gh/2024-03-02/calibration/ruby/{file_name}.ms"
        model_file = f"/lustre/rbyrne/2024-03-02/calibration_models/{file_name}_deGasperin_point_sources.ms"

        uvcal = calibration_wrappers.calibration_per_pol(
            datafile,
            model_file,
            data_use_column="DATA",
            model_use_column="DATA",
            conjugate_model=True,
            min_cal_baseline_lambda=10,
            max_cal_baseline_lambda=125,
            verbose=True,
            get_crosspol_phase=False,
            log_file_path=f"/lustre/rbyrne/2024-03-02/calibration_outputs/{file_name}_cal_log_v1.txt",
        )
        uvcal.write_calfits(
            f"/lustre/rbyrne/2024-03-02/calibration_outputs/{file_name}.calfits",
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
            f"/lustre/rbyrne/2024-03-02/calibration_outputs/{file_name}_calibrated.ms",
            fix_autos=True,
            clobber=True,
        )


def run_newcal_Aug12():

    use_filenames = [
        "18",
        "23",
        "27",
        "36",
        "41",
        "46",
        "50",
        "55",
        "59",
        "64",
        "73",
        "78",
        "82",
    ]

    for file_name in use_filenames:

        datafile = f"/lustre/gh/2024-03-02/calibration/ruby/{file_name}.ms"
        model_file = f"/lustre/rbyrne/2024-03-02/calibration_models/{file_name}_deGasperin_sources.ms"

        uvcal = calibration_wrappers.calibration_per_pol(
            datafile,
            model_file,
            data_use_column="DATA",
            model_use_column="DATA",
            conjugate_model=True,
            min_cal_baseline_lambda=10,
            max_cal_baseline_lambda=125,
            verbose=True,
            get_crosspol_phase=False,
            log_file_path=f"/lustre/rbyrne/2024-03-02/calibration_outputs/{file_name}_cal_log_v2.txt",
            xtol=1e-5,
            maxiter=200,  # reduce maxiter for debugging
            antenna_flagging_iterations=0,
        )
        uvcal.write_calfits(
            f"/lustre/rbyrne/2024-03-02/calibration_outputs/{file_name}_extended_sources.calfits",
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
            f"/lustre/rbyrne/2024-03-02/calibration_outputs/{file_name}_extended_sources_calibrated.ms",
            fix_autos=True,
            clobber=True,
        )


def run_newcal_CygA_Oct2():

    use_filenames = [
        "18",
        "23",
        "27",
        "36",
        "41",
        "46",
        "50",
        "55",
        "59",
        "64",
        "73",
        "78",
        "82",
    ]

    for file_name in use_filenames:

        datafile = f"/lustre/gh/2024-03-02/calibration/ruby/{file_name}.ms"
        model_file = f"/lustre/rbyrne/2024-03-02/ruby/calibration_models/{file_name}_cygA_point_source.ms"

        uvcal = calibration_wrappers.calibration_per_pol_wrapper(
            datafile,
            model_file,
            data_use_column="DATA",
            model_use_column="DATA",
            conjugate_model=True,
            min_cal_baseline_lambda=10,
            max_cal_baseline_lambda=125,
            verbose=True,
            get_crosspol_phase=False,
            log_file_path=f"/lustre/rbyrne/2024-03-02/ruby/calibration_outputs/{file_name}_cal_log_cygA.txt",
            xtol=1e-5,
            maxiter=200,  # reduce maxiter for debugging
            antenna_flagging_iterations=0,
        )
        uvcal.write_calfits(
            f"/lustre/rbyrne/2024-03-02/ruby/calibration_outputs/{file_name}_cygA_point.calfits",
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
            f"/lustre/rbyrne/2024-03-02/ruby/calibration_outputs/{file_name}_cygA_point_calibrated.ms",
            fix_autos=True,
            clobber=True,
        )


def run_newcal_single_freq_Oct10():

    file_name = 18
    datafile = f"/lustre/rbyrne/2024-03-02/ruby/{file_name}_1freq_1time.ms"
    model_file = f"/lustre/rbyrne/2024-03-02/ruby/{file_name}_1freq_1time_model.ms"

    """uvcal = calibration_wrappers.calibration_per_pol_wrapper(
        datafile,
        model_file,
        data_use_column="DATA",
        model_use_column="DATA",
        conjugate_model=True,
        min_cal_baseline_lambda=10,
        max_cal_baseline_lambda=125,
        verbose=True,
        get_crosspol_phase=False,
        log_file_path=f"/lustre/rbyrne/2024-03-02/ruby/calibration_outputs/{file_name}_cal_log_cygA_1freq_1time.txt",
        xtol=1e-5,
        maxiter=200,  # reduce maxiter for debugging
        antenna_flagging_iterations=0,
    )
    uvcal.write_calfits(
        f"/lustre/rbyrne/2024-03-02/ruby/calibration_outputs/{file_name}_cygA_point_1freq_1time.calfits",
        clobber=True,
    )
    
    #uvcal = pyuvdata.UVCal()
    #uvcal.read(f"/lustre/rbyrne/2024-03-02/ruby/calibration_outputs/{file_name}_cygA_point_1freq_1time.calfits")

    data = pyuvdata.UVData()
    data.read_ms(
        datafile,
        data_column="DATA",
        ignore_single_chan=False,
    )
    pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)
    data.reorder_pols(order="CASA")
    data.write_ms(
        f"/lustre/rbyrne/2024-03-02/ruby/calibration_outputs/{file_name}_cygA_point_1freq_1time_calibrated.ms",
        fix_autos=True,
        clobber=True,
    )"""

    uvcal = calibration_wrappers.calibration_per_pol_wrapper(
        datafile,
        model_file,
        data_use_column="DATA",
        model_use_column="DATA",
        conjugate_model=True,
        min_cal_baseline_lambda=10,
        max_cal_baseline_lambda=125,
        gains_multiply_model=True,
        verbose=True,
        get_crosspol_phase=False,
        log_file_path=f"/lustre/rbyrne/2024-03-02/ruby/calibration_outputs/{file_name}_cal_log_cygA_1freq_1time_inverse.txt",
        xtol=1e-10,  # reduce for inverse gains
        maxiter=200,  # reduce maxiter for debugging
        antenna_flagging_iterations=0,
    )
    uvcal.write_calfits(
        f"/lustre/rbyrne/2024-03-02/ruby/calibration_outputs/{file_name}_cygA_point_1freq_1time_inverse_lowtol.calfits",
        clobber=True,
    )
    data = pyuvdata.UVData()
    data.read_ms(
        datafile,
        data_column="DATA",
        ignore_single_chan=False,
    )
    pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)
    data.reorder_pols(order="CASA")
    data.write_ms(
        f"/lustre/rbyrne/2024-03-02/ruby/calibration_outputs/{file_name}_cygA_point_1freq_1time_calibrated_inverse_lowtol.ms",
        fix_autos=True,
        clobber=True,
    )


def calibrate_data_Oct14(freq_band):

    datafile = f"/lustre/rbyrne/2024-03-03/20240303_093000-093151_{freq_band}MHz.uvfits"
    model_file = (
        f"/lustre/rbyrne/2024-03-03/20240303_093000-093151_{freq_band}MHz_model.uvfits"
    )

    uvcal = calibration_wrappers.calibration_per_pol_wrapper(
        datafile,
        model_file,
        conjugate_data=True,
        min_cal_baseline_lambda=10,
        max_cal_baseline_lambda=125,
        verbose=True,
        get_crosspol_phase=False,
        log_file_path=f"/lustre/rbyrne/2024-03-03/calibration_logs/20240303_093000-093151_{freq_band}MHz_cal_log.txt",
        xtol=1e-5,
        maxiter=200,  # reduce maxiter for debugging
        antenna_flagging_iterations=0,
        parallel=False,
    )
    uvcal.write_calfits(
        f"/lustre/rbyrne/2024-03-03/20240303_093000-093151_{freq_band}MHz.calfits",
        clobber=True,
    )

    data = pyuvdata.UVData()
    data.read_uvfits(datafile)
    data.data_array = np.conj(data.data_array)  # Conjugate data
    pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)
    data.write_uvfits(
        f"/lustre/rbyrne/2024-03-03/20240303_093000-093151_{freq_band}MHz_calibrated.uvfits",
        fix_autos=True,
    )


def casa_cal_comparison_Oct21():

    uvcal = calibration_wrappers.calibration_per_pol_wrapper(
        "/lustre/rbyrne/2024-03-02/ruby/18_1freq_1time.ms",
        "/lustre/rbyrne/2024-03-02/ruby/18_1freq_1time.ms",
        data_use_column="DATA",
        model_use_column="MODEL_DATA",
        # gain_init_calfile="/lustre/rbyrne/2024-03-02/ruby/18_1freq_1time.calfits",
        gain_init_to_vis_ratio=True,
        conjugate_model=False,
        gains_multiply_model=True,
        min_cal_baseline_lambda=15,
        verbose=True,
        get_crosspol_phase=True,
        log_file_path=f"/lustre/rbyrne/2024-03-02/ruby/18_1freq_1time_casa_comparison_log.txt",
        xtol=1e-9,
        maxiter=200,  # reduce maxiter for debugging
        antenna_flagging_iterations=0,
        lambda_val=0,
    )
    uvcal.write_calfits(
        f"/lustre/rbyrne/2024-03-02/ruby/18_1freq_1time_casa_compare.calfits",
        clobber=True,
    )

    data = pyuvdata.UVData()
    data.read_ms(
        "/lustre/rbyrne/2024-03-02/ruby/18_1freq_1time.ms",
        data_column="DATA",
        ignore_single_chan=False,
    )
    pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)
    data.write_ms(
        f"/lustre/rbyrne/2024-03-02/ruby/18_1freq_1time_casa_compare_newcal.ms",
        fix_autos=True,
        clobber=True,
    )


def calibrate_data_Dec2024(data_filepath):

    data_filepath = "/lustre/rbyrne/2024-03-03/20240303_093000-093151_41MHz_reprocess_Dec2024.ms"  # Created with concatenate_ms_files.py

    flag_ants = [
        "LWA009",
        "LWA041",
        "LWA044",
        "LWA052",
        "LWA058",
        "LWA076",
        "LWA095",
        "LWA105",
        "LWA111",
        "LWA120",
        "LWA124",
        "LWA138",
        "LWA150",
        "LWA159",
        "LWA191",
        "LWA204",
        "LWA208",
        "LWA209",
        "LWA232",
        "LWA234",
        "LWA255",
        "LWA267",
        "LWA280",
        "LWA288",
        "LWA292",
        "LWA302",
        "LWA307",
        "LWA309",
        "LWA310",
        "LWA314",
        "LWA325",
        "LWA341",
        "LWA352",
        "LWA364",
        "LWA365",
    ]

    # Convert to uvdata object
    uv = pyuvdata.UVData()
    print(f"Reading file {data_filepath}.")
    uv.read(data_filepath, data_column="DATA")
    uv.set_uvws_from_antenna_positions(update_vis=False)
    uv.data_array = np.conj(uv.data_array)
    uv.phase_to_time(np.mean(uv.time_array))

    # Flag antennas
    LWA_preprocessing.flag_antennas(
        uv,
        antenna_names=flag_ants,
        flag_pol="all",  # Options are "all", "X", "Y", "XX", "YY", "XY", or "YX"
        inplace=True,
    )

    model_file_name = data_filepath.replace(".ms", "_model.ms")
    # Get model
    model_filepath = "/lustre/rbyrne/simulation_outputs"
    lst_lookup_table_path = f"{model_filepath}/lst_lookup_table.csv"
    with open(lst_lookup_table_path, "r") as f:
        lst_data = f.readlines()
    model_lsts = np.array([])
    model_lst_filenames = np.array([])
    for line in lst_data[1:]:
        line_split = line.replace("\n", "").strip().split(",")
        model_lsts = np.append(model_lsts, float(line_split[0]))
        model_lst_filenames = np.append(
            model_lst_filenames,
            f"{line_split[1]}_{freq_band}MHz_source_sim.uvfits".strip(),
        )

    model_uv_list = []
    for time_ind, use_lst in enumerate(list(set(uv.lst_array))):
        print(
            f"Calculating model visibilities for time step {time_ind+1} of {len(list(set(uv.lst_array)))}."
        )
        lst_distance = np.abs(model_lsts - use_lst)
        ind1 = np.where(lst_distance == np.min(lst_distance))[0]
        ind2 = np.where(lst_distance == np.sort(lst_distance)[1])[0]
        lst1 = model_lsts[ind1]
        model_filename1 = model_lst_filenames[ind1][0]
        lst2 = model_lsts[ind2]
        model_filename2 = model_lst_filenames[ind2][0]

        # Interpolate models
        model1_uv = pyuvdata.UVData()
        model1_uv.read(f"{model_filepath}/{model_filename1}")
        model1_uv.select(lsts=[lst1])
        model1_uv.filename = [""]
        model1_uv_diffuse = pyuvdata.UVData()
        model1_uv_diffuse.read(
            f"{model_filepath}/{model_filename1.removesuffix('source_sim.uvfits')}diffuse_sim.uvfits"
        )
        model1_uv_diffuse.select(lsts=[lst1])
        model1_uv_diffuse.filename = [""]
        model1_uv.sum_vis(model1_uv_diffuse, inplace=True)

        model2_uv = pyuvdata.UVData()
        model2_uv.read(f"{model_filepath}/{model_filename2}")
        model2_uv.select(lsts=[lst2])
        model2_uv.filename = [""]
        model2_uv_diffuse = pyuvdata.UVData()
        model2_uv_diffuse.read(
            f"{model_filepath}/{model_filename2.removesuffix('source_sim.uvfits')}diffuse_sim.uvfits"
        )
        model2_uv_diffuse.select(lsts=[lst2])
        model2_uv_diffuse.filename = [""]
        model2_uv.sum_vis(model2_uv_diffuse, inplace=True)

        # Phase to consistent phase center
        model1_uv.phase_to_time(np.mean(uv.time_array))
        model2_uv.phase_to_time(np.mean(uv.time_array))

        # Combine data
        model1_uv.data_array *= np.abs(lst1 - use_lst) / np.abs(lst2 - lst1)
        model2_uv.data_array *= np.abs(lst2 - use_lst) / np.abs(lst2 - lst1)
        model_uv = model1_uv.sum_vis(
            model2_uv,
            inplace=False,
            run_check=False,
            check_extra=False,
            run_check_acceptability=False,
            override_params=["lst_array", "time_array", "uvw_array", "filename"],
        )
        # Correct for decoherence
        model1_uv.data_array = np.abs(model1_uv.data_array) + 0 * 1j
        model2_uv.data_array = np.abs(model2_uv.data_array) + 0 * 1j
        model_uv_abs = model1_uv.sum_vis(
            model2_uv,
            inplace=False,
            run_check=False,
            check_extra=False,
            run_check_acceptability=False,
            override_params=["lst_array", "time_array", "uvw_array", "filename"],
        )
        model_uv.data_array *= np.abs(model_uv_abs.data_array) / np.abs(
            model_uv.data_array
        )

        model_uv.time_array = np.full(
            model_uv.Nblts, np.sort(list(set(uv.time_array)))[time_ind]
        )
        model_uv.lst_array = np.full(model_uv.Nblts, use_lst)
        model_uv_list.append(model_uv)

    # Combine LSTs
    combined_model_uv = model_uv_list[0]
    if len(model_uv_list) > 1:
        for model_uv_use in model_uv_list[1:]:
            combined_model_uv.fast_concat(model_uv_use, "blt", inplace=True)

    print(f"Saving model file to {model_file_name}.")
    combined_model_uv.write_ms(model_file_name, force_phase=True)


if __name__ == "__main__":
    # args = sys.argv
    # freq_band = args[1]
    freq_band = 41
    calibrate_data_Dec2024(freq_band)
    # casa_cal_comparison_Oct21()
