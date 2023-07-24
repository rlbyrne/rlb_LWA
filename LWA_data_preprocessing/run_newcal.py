import numpy as np
import pyuvdata
from newcal import calibration_wrappers, calibration_optimization


data = pyuvdata.UVData()
data.read("/data10/rbyrne/24-hour-run-flagged/20230309_225134_73MHz.ms")
model = pyuvdata.UVData()
model.read("/home/rbyrne/calibration_testing_Jul2023/20230309_225134_73MHz_model.uvfits")

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
    log_file_path="/home/rbyrne/calibration_testing_Jul2023/20230309_225134_73MHz_cal_log.txt",
)

cal = calibration_optimization.create_uvcal_obj(data, antenna_names, gains=gains_fit)
cal.write_calfits("/home/rbyrne/calibration_testing_Jul2023/20230309_225134_73MHz.calfits")
