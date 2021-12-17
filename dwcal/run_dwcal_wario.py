import delay_weighted_cal as dwcal

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
