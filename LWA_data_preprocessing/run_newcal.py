import numpy as np
import pyuvdata
from newcal import calibration_wrappers, calibration_optimization

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

    cal = calibration_optimization.create_uvcal_obj(data, antenna_names, gains=gains_fit)
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
