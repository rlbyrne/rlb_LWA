from dwcal import delay_weighted_cal as dwcal
import numpy as np
import pyuvdata
from pyuvdata import utils
import sys
import time


def run_calibration_Jan5():

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        data_use_model=False,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=False,
        cal_savefile="/safepool/rbyrne/calibration_outputs/vanilla_cal/excluded_sources_vanilla_cal.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/vanilla_cal/1061316296_excluded_sources_vanilla_cal.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/vanilla_cal/excluded_sources_vanilla_cal_log.txt",
    )

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        data_use_model=False,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=True,
        cal_savefile="/safepool/rbyrne/calibration_outputs/wedge_exclusion_cal/excluded_sources_wedge_excluded.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/wedge_exclusion_cal/1061316296_excluded_sources_wedge_excluded.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/wedge_exclusion_cal/excluded_sources_wedge_excluded_log.txt",
    )


def create_randomized_gains_Jan11(randomized_gains_uvfits_path):

    # Load data
    data_path = (
        "/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021"
    )
    data_use_model = False
    obsid = "1061316296"
    pol = "XX"
    data_filelist = [
        "{}/{}".format(data_path, file)
        for file in [
            "vis_data/{}_vis_{}.sav".format(obsid, pol),
            "vis_data/{}_vis_model_{}.sav".format(obsid, pol),
            "vis_data/{}_flags.sav".format(obsid),
            "metadata/{}_params.sav".format(obsid),
            "metadata/{}_settings.txt".format(obsid),
            "metadata/{}_layout.sav".format(obsid),
        ]
    ]
    data = pyuvdata.UVData()
    data.read_fhd(data_filelist, use_model=data_use_model)

    # Create random gains
    antenna_list = np.sort(np.unique([data.ant_1_array, data.ant_2_array]))
    gain_stddev = 0.05
    gains = np.random.normal(
        1.0,
        gain_stddev,
        size=(data.Nants_data, data.Nfreqs),
    ) + 1.0j * np.random.normal(
        0.0,
        gain_stddev,
        size=(data.Nants_data, data.Nfreqs),
    )
    cal_obj = dwcal.initialize_cal(data, antenna_list, gains=gains)
    cal_obj.write_calfits(
        "/safepool/rbyrne/calibration_outputs/randomized_gains/true_gains.calfits",
        clobber=True,
    )

    # Apply gains
    data_calibrated = utils.uvcalibrate(data, cal_obj, inplace=False)
    data_calibrated.write_uvfits(randomized_gains_uvfits_path)


def run_calibration_Jan11():

    randomized_gains_uvfits_path = "/safepool/rbyrne/calibration_outputs/randomized_gains/data_with_randomized_gains.uvfits"

    create_randomized_gains_Jan11(randomized_gains_uvfits_path)

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        model_use_model=True,
        data_path=randomized_gains_uvfits_path,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=False,
        cal_savefile="/safepool/rbyrne/calibration_outputs/randomized_gains/data_with_randomized_gains_vanilla_cal.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/randomized_gains/data_with_randomized_gains_vanilla_cal.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/randomized_gains/data_with_randomized_gains_vanilla_cal_log.txt",
    )

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        model_use_model=True,
        data_path=randomized_gains_uvfits_path,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=True,
        cal_savefile="/safepool/rbyrne/calibration_outputs/randomized_gains/data_with_randomized_gains_wedge_excluded.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/randomized_gains/data_with_randomized_gains_wedge_excluded.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/randomized_gains/data_with_randomized_gains_wedge_excluded_log.txt",
    )


def run_calibration_Jan18():

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        data_use_model=False,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=False,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_Jan18/vanilla_cal.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/caltest_Jan18/vanilla_cal.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_Jan18/vanilla_cal_log.txt",
    )

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        data_use_model=False,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=True,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_Jan18/wedge_excluded.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/caltest_Jan18/wedge_excluded.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_Jan18/wedge_excluded_log.txt",
    )


def test_constrained_optimization_Mar1():

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        data_use_model=False,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=False,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_Mar1/vanilla_cal.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/caltest_Mar1/vanilla_cal.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_Mar1/vanilla_cal_log.txt",
    )

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        data_use_model=False,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=True,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_Mar1/wedge_excluded.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/caltest_Mar1/wedge_excluded.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_Mar1/wedge_excluded_log.txt",
    )


def test_constrained_optimization_Mar14():

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        data_use_model=False,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=False,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_Mar14/vanilla_cal.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/caltest_Mar14/vanilla_cal.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_Mar14/vanilla_cal_log.txt",
    )

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        data_use_model=False,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=True,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_Mar14/wedge_excluded.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/caltest_Mar14/wedge_excluded.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_Mar14/wedge_excluded_log.txt",
    )


def test_updated_weight_mat_calculation_Mar31():

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        data_use_model=False,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=False,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_Mar31/vanilla_cal.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/caltest_Mar31/vanilla_cal.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_Mar31/vanilla_cal_log.txt",
    )

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Dec2021",
        data_use_model=False,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=True,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_Mar31/wedge_excluded.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/caltest_Mar31/wedge_excluded.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_Mar31/wedge_excluded_log.txt",
    )


def dwcal_test_Apr11():

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Apr2022",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Apr2022",
        data_use_model=True,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=False,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_Apr11/vanilla_cal.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/caltest_Apr11/vanilla_cal.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_Apr11/vanilla_cal_log.txt",
    )

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Apr2022",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Apr2022",
        data_use_model=True,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=True,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_Apr11/wedge_excluded.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/caltest_Apr11/wedge_excluded.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_Apr11/wedge_excluded_log.txt",
    )


def dwcal_test_Apr12():

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Apr2022",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Apr2022",
        data_use_model=True,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=False,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_Apr12/vanilla_cal.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/caltest_Apr12/vanilla_cal.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_Apr12/vanilla_cal_log.txt",
    )

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Apr2022",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Apr2022",
        data_use_model=True,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=True,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_Apr12/wedge_excluded.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/caltest_Apr12/wedge_excluded.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_Apr12/wedge_excluded_log.txt",
    )


def dwcal_newtons_method_test_Apr15():

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Apr2022",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Apr2022",
        data_use_model=True,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=False,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_newtons_method_Apr12/vanilla_cal.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/caltest_newtons_method_Apr12/vanilla_cal.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_newtons_method_Apr12/vanilla_cal_log.txt",
        use_newtons_method=True,
    )

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Apr2022",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Apr2022",
        data_use_model=True,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=True,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_newtons_method_Apr12/wedge_excluded.calfits",
        calibrated_data_savefile="/safepool/rbyrne/calibration_outputs/caltest_newtons_method_Apr12/wedge_excluded.uvfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_newtons_method_Apr12/wedge_excluded_log.txt",
        use_newtons_method=True,
    )


def dwcal_newtons_method_test_Apr22():

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Apr2022",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Apr2022",
        data_use_model=True,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=False,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_newtons_method_Apr22/vanilla_cal.calfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_newtons_method_Apr22/vanilla_cal_log.txt",
        use_newtons_method=True,
        gain_init_stddev=0.0,
        gain_init_calfile="/safepool/rbyrne/calibration_outputs/caltest_Apr12/vanilla_cal.calfits",
    )

    dwcal.calibrate(
        model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Apr2022",
        model_use_model=True,
        data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Apr2022",
        data_use_model=True,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        use_wedge_exclusion=True,
        cal_savefile="/safepool/rbyrne/calibration_outputs/caltest_newtons_method_Apr22/wedge_excluded.calfits",
        log_file_path="/safepool/rbyrne/calibration_outputs/caltest_newtons_method_Apr22/wedge_excluded_log.txt",
        use_newtons_method=True,
        gain_init_stddev=0.0,
        gain_init_calfile="/safepool/rbyrne/calibration_outputs/caltest_Apr12/wedge_excluded.calfits",
    )


def random_gains_test_Apr25():

    save_dir = "/safepool/rbyrne/calibration_outputs/random_gains_test_Apr25"

    model_path = (
        "/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Apr2022"
    )
    model_use_model = True
    data_path = "/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Apr2022"
    data_use_model = True
    obsid = "1061316296"
    pol = "XX"
    use_autos = False

    data, model = dwcal.get_test_data(
        model_path=model_path,
        model_use_model=model_use_model,
        data_path=data_path,
        data_use_model=data_use_model,
        obsid=obsid,
        pol=pol,
        use_autos=use_autos,
        debug_limit_freqs=None,
        use_antenna_list=None,
        use_flagged_baselines=False,
    )

    # Create randomized gains and apply to data
    randomized_gains_cal_savefile = f"{save_dir}/random_initial_gains.calfits"
    randomized_gains_data_uvfits = f"{save_dir}/random_initial_gains_data.uvfits"
    random_gains_stddev = 0.01
    random_gains = np.random.normal(
        1.0,
        random_gains_stddev,
        size=(data.Nants_data, data.Nfreqs),
    ) + 1.0j * np.random.normal(
        0.0,
        random_gains_stddev,
        size=(data.Nants_data, data.Nfreqs),
    )

    # Ensure that the phase of the gains is mean-zero for each frequency
    avg_angle = np.arctan2(
        np.mean(np.sin(np.angle(random_gains)), axis=0),
        np.mean(np.cos(np.angle(random_gains)), axis=0),
    )
    random_gains *= np.cos(avg_angle) - 1j * np.sin(avg_angle)

    # Save randomized gains
    antenna_list = np.unique([data.ant_1_array, data.ant_2_array])
    random_gains_cal = dwcal.initialize_cal(data, antenna_list, gains=random_gains)
    random_gains_cal.gain_convention = "divide"  # Apply initial calibration as division
    random_gains_cal.write_calfits(randomized_gains_cal_savefile, clobber=True)

    # Apply gains to data
    pyuvdata.utils.uvcalibrate(data, random_gains_cal, inplace=True, time_check=False)

    # Save data as uvfits
    data.write_uvfits(randomized_gains_data_uvfits)

    # Do vanilla cal
    use_wedge_exclusion = False
    cal_savefile = f"{save_dir}/vanilla_cal.calfits"
    log_file_path = f"{save_dir}/vanilla_cal_log.txt"
    calibrated_data_savefile = (f"{save_dir}/vanilla_cal.uvfits",)

    if log_file_path is not None:
        sys.stdout = open(log_file_path, "w")
        sys.stderr = sys.stdout

    cal = dwcal.calibration_optimization(
        data,
        model,
        use_wedge_exclusion=use_wedge_exclusion,
        log_file_path=log_file_path,
    )

    if cal_savefile is not None:
        print(f"Saving calibration solutions to {cal_savefile}")
        sys.stdout.flush()
        cal.write_calfits(cal_savefile, clobber=True)

    if log_file_path is not None:
        sys.stdout.close()

    # Do wedge excluded cal
    use_wedge_exclusion = True
    cal_savefile = f"{save_dir}/wedge_excluded.calfits"
    log_file_path = f"{save_dir}/wedge_excluded_log.txt"
    calibrated_data_savefile = (f"{save_dir}/wedge_excluded.uvfits",)

    if log_file_path is not None:
        sys.stdout = open(log_file_path, "w")
        sys.stderr = sys.stdout

    cal = dwcal.calibration_optimization(
        data,
        model,
        use_wedge_exclusion=use_wedge_exclusion,
        log_file_path=log_file_path,
    )

    if cal_savefile is not None:
        print(f"Saving calibration solutions to {cal_savefile}")
        sys.stdout.flush()
        cal.write_calfits(cal_savefile, clobber=True)

    if log_file_path is not None:
        sys.stdout.close()


def gain_ripple_test_May6():

    save_dir = "/safepool/rbyrne/calibration_outputs/gain_ripple_test_May6"

    model_path = (
        "/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Apr2022"
    )
    model_use_model = True
    data_path = "/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Apr2022"
    data_use_model = True
    obsid = "1061316296"
    pol = "XX"
    use_autos = False

    data, model = dwcal.get_test_data(
        model_path=model_path,
        model_use_model=model_use_model,
        data_path=data_path,
        data_use_model=data_use_model,
        obsid=obsid,
        pol=pol,
        use_autos=use_autos,
        debug_limit_freqs=None,
        use_antenna_list=None,
        use_flagged_baselines=False,
    )

    # Create gain with frequency ripple and apply to data
    ripple_gains_cal_savefile = f"{save_dir}/ripple_initial_gains.calfits"
    ripple_gains_data_uvfits = f"{save_dir}/ripple_initial_gains_data.uvfits"
    delay_mode = 1e-6
    use_ants = [
        "Tile143",
        "Tile072",
        "Tile012",
        "Tile088",
        "Tile042",
    ]  # 5 randomly selected antennas
    delay_array = np.fft.fftfreq(data.Nfreqs, d=data.channel_width)
    ripple_delay = np.zeros_like(delay_array)
    ripple_delay[np.argmin(np.abs(delay_array - 1e-6))] = 1  # Add ripple
    ripple_gains = np.fft.ifft(ripple_delay)
    ripple_gains /= np.mean(np.abs(ripple_gains))  # Set mean amp to 1

    antenna_list = np.unique([data.ant_1_array, data.ant_2_array])
    gains = np.ones((data.Nants_data, data.Nfreqs), dtype=complex)
    for antname in use_ants:
        ant_ind = np.where(np.array(data.antenna_names) == antname)[0][0]
        gains[np.where(antenna_list == ant_ind)[0], :] = ripple_gains

    # Ensure that the phase of the gains is mean-zero for each frequency
    avg_angle = np.arctan2(
        np.mean(np.sin(np.angle(gains)), axis=0),
        np.mean(np.cos(np.angle(gains)), axis=0),
    )
    gains *= np.cos(avg_angle) - 1j * np.sin(avg_angle)

    # Save gains
    ripple_gains_cal = dwcal.initialize_cal(data, antenna_list, gains=gains)
    ripple_gains_cal.gain_convention = "divide"  # Apply initial calibration as division
    ripple_gains_cal.write_calfits(ripple_gains_cal_savefile, clobber=True)

    # Apply gains to data
    pyuvdata.utils.uvcalibrate(data, ripple_gains_cal, inplace=True, time_check=False)

    # Save data as uvfits
    data.write_uvfits(ripple_gains_data_uvfits)

    # Do vanilla cal
    use_wedge_exclusion = False
    cal_savefile = f"{save_dir}/vanilla_cal.calfits"
    log_file_path = f"{save_dir}/vanilla_cal_log.txt"
    calibrated_data_savefile = (f"{save_dir}/vanilla_cal.uvfits",)

    if log_file_path is not None:
        sys.stdout = open(log_file_path, "w")
        sys.stderr = sys.stdout

    cal = dwcal.calibration_optimization(
        data,
        model,
        use_wedge_exclusion=use_wedge_exclusion,
        log_file_path=log_file_path,
    )

    if cal_savefile is not None:
        print(f"Saving calibration solutions to {cal_savefile}")
        sys.stdout.flush()
        cal.write_calfits(cal_savefile, clobber=True)

    if log_file_path is not None:
        sys.stdout.close()

    # Do wedge excluded cal
    use_wedge_exclusion = True
    cal_savefile = f"{save_dir}/wedge_excluded.calfits"
    log_file_path = f"{save_dir}/wedge_excluded_log.txt"
    calibrated_data_savefile = (f"{save_dir}/wedge_excluded.uvfits",)

    if log_file_path is not None:
        sys.stdout = open(log_file_path, "w")
        sys.stderr = sys.stdout

    cal = dwcal.calibration_optimization(
        data,
        model,
        use_wedge_exclusion=use_wedge_exclusion,
        log_file_path=log_file_path,
    )

    if cal_savefile is not None:
        print(f"Saving calibration solutions to {cal_savefile}")
        sys.stdout.flush()
        cal.write_calfits(cal_savefile, clobber=True)

    if log_file_path is not None:
        sys.stdout.close()


def gain_ripple_newtons_test_May17():

    save_dir = "/safepool/rbyrne/calibration_outputs/gain_ripple_newtons_test_May17"
    ripple_gains_cal_savefile = "/safepool/rbyrne/calibration_outputs/gain_ripple_test_May6/ripple_initial_gains.calfits"
    gain_init_calfile = "/safepool/rbyrne/calibration_outputs/gain_ripple_test_May6/wedge_excluded.calfits"

    model_path = (
        "/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Apr2022"
    )
    model_use_model = True
    data_path = "/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Apr2022"
    data_use_model = True
    obsid = "1061316296"
    pol = "XX"
    use_autos = False

    data, model = dwcal.get_test_data(
        model_path=model_path,
        model_use_model=model_use_model,
        data_path=data_path,
        data_use_model=data_use_model,
        obsid=obsid,
        pol=pol,
        use_autos=use_autos,
        debug_limit_freqs=None,
        use_antenna_list=None,
        use_flagged_baselines=False,
    )

    ripple_gains_cal.UVCal()
    ripple_gains_cal.read_calfits(ripple_gains_cal_savefile)

    # Apply gains to data
    pyuvdata.utils.uvcalibrate(data, ripple_gains_cal, inplace=True, time_check=False)

    # Do wedge excluded cal
    use_wedge_exclusion = True
    cal_savefile = f"{save_dir}/wedge_excluded.calfits"
    log_file_path = f"{save_dir}/wedge_excluded_log.txt"

    if log_file_path is not None:
        sys.stdout = open(log_file_path, "w")
        sys.stderr = sys.stdout

    cal = dwcal.calibration_optimization(
        data,
        model,
        use_wedge_exclusion=use_wedge_exclusion,
        log_file_path=log_file_path,
        use_newtons_method=True,
        gain_init_calfile=gain_init_calfile,
    )

    if cal_savefile is not None:
        print(f"Saving calibration solutions to {cal_savefile}")
        sys.stdout.flush()
        cal.write_calfits(cal_savefile, clobber=True)

    if log_file_path is not None:
        sys.stdout.close()


if __name__ == "__main__":

    gain_ripple_newtons_test_May17()
