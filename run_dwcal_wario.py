from dwcal import delay_weighted_cal as dwcal
import numpy as np
import pyuvdata
from pyuvdata import utils


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
        1.0, gain_stddev, size=(data.Nants_data, data.Nfreqs),
    ) + 1.0j * np.random.normal(0.0, gain_stddev, size=(data.Nants_data, data.Nfreqs),)
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


if __name__ == "__main__":

    dwcal_newtons_method_test_Apr15()
