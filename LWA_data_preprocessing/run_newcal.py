import numpy as np
import pyuvdata
import newcal


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
    model_visibilities,
    data_visibilities,
    visibility_weights,
    gains_exp_mat_1,
    gains_exp_mat_2,
) = newcal.uvdata_calibration_setup(
    data,
    model,
    gain_init_calfile=None,
    gain_init_stddev=0.0,
    N_feed_pols=2,
)

gains_fit = newcal.run_calibration_optimization_per_pol(
    gains_init,
    Nants,
    Nbls,
    Nfreqs,
    2,
    model_visibilities,
    data_visibilities,
    visibility_weights,
    gains_exp_mat_1,
    gains_exp_mat_2,
    0.01,
    xtol=1e-8,
    verbose=False,
)
