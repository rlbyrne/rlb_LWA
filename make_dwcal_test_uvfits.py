import pyuvdata
from pyuvdata import utils
import numpy as np

data_path = "/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Jun2022"
data_use_model = True
obsid = "1061316296"

data = pyuvdata.UVData()
filelist = [
    "{}/{}".format(data_path, file)
    for file in [
        "vis_data/{}_vis_XX.sav".format(obsid),
        "vis_data/{}_vis_YY.sav".format(obsid),
        "vis_data/{}_vis_model_XX.sav".format(obsid),
        "vis_data/{}_vis_model_YY.sav".format(obsid),
        "vis_data/{}_flags.sav".format(obsid),
        "metadata/{}_params.sav".format(obsid),
        "metadata/{}_settings.txt".format(obsid),
        "metadata/{}_layout.sav".format(obsid),
    ]
]
data.read_fhd(filelist, use_model=data_use_model)

uvfits_output_dir = "/safepool/rbyrne/calibration_outputs/caltest_Jun17"

# Unity gains
cal_filenames = [
    "/safepool/rbyrne/calibration_outputs/caltest_Jun16/unity_gains_diagonal.calfits",
    "/safepool/rbyrne/calibration_outputs/caltest_Jun17/unity_gains_dwcal.calfits",
]
uvfits_output_filenames = [
    f"{uvfits_output_dir}/unity_gains_diagonal_2pol.uvfits",
    f"{uvfits_output_dir}/unity_gains_dwcal_2pol.uvfits",
]
data.write_uvfits(f"{uvfits_output_dir}/unity_gains_uncalib_2pol.uvfits")
for ind in range(len(cal_filenames)):
    cal = pyuvdata.UVCal()
    cal.read_calfits(cal_filenames[ind])
    # Transfer calibration to the YY pol
    cal.jones_array = np.append(cal.jones_array, [-6])
    cal.gain_array = np.repeat(cal.gain_array, 2, axis=4)
    cal.flag_array = np.repeat(cal.flag_array, 2, axis=4)
    cal.quality_array = np.repeat(cal.quality_array, 2, axis=4)
    cal.Njones = 2
    # Apply calibration
    data_calibrated = pyuvdata.utils.uvcalibrate(
        data, cal, inplace=False, time_check=False
    )
    data_calibrated.write_uvfits(uvfits_output_filenames[ind])

if False:
    # Random gains
    cal_filenames = [
        "/safepool/rbyrne/calibration_outputs/caltest_May19/random_gains_diagonal.calfits",
        "/safepool/rbyrne/calibration_outputs/caltest_May19/random_gains_dwcal.calfits",
    ]
    uvfits_output_filenames = [
        f"{uvfits_output_dir}/random_gains_diagonal.uvfits",
        f"{uvfits_output_dir}/random_gains_dwcal.uvfits",
    ]
    true_gains_calfits = "/safepool/rbyrne/calibration_outputs/caltest_May19/random_initial_gains.calfits"
    true_gains_cal = pyuvdata.UVCal()
    true_gains_cal.read_calfits(true_gains_calfits)
    data_orig = pyuvdata.utils.uvcalibrate(
        data, true_gains_cal, inplace=False, time_check=False
    )
    data_orig.write_uvfits(f"{uvfits_output_dir}/random_gains_uncalib.uvfits")
    for ind in range(len(cal_filenames)):
        cal = pyuvdata.UVCal()
        cal.read_calfits(cal_filenames[ind])
        data_calibrated = pyuvdata.utils.uvcalibrate(
            data_orig, cal, inplace=False, time_check=False
        )
        data_calibrated.write_uvfits(uvfits_output_filenames[ind])

    # Ripple gains
    cal_filenames = [
        "/safepool/rbyrne/calibration_outputs/caltest_May19/ripple_gains_diagonal.calfits",
        "/safepool/rbyrne/calibration_outputs/caltest_May19/ripple_gains_dwcal.calfits",
    ]
    uvfits_output_filenames = [
        f"{uvfits_output_dir}/ripple_gains_diagonal.uvfits",
        f"{uvfits_output_dir}/ripple_gains_dwcal.uvfits",
    ]
    true_gains_calfits = "/safepool/rbyrne/calibration_outputs/caltest_May19/ripple_initial_gains.calfits"
    true_gains_cal = pyuvdata.UVCal()
    true_gains_cal.read_calfits(true_gains_calfits)
    data_orig = pyuvdata.utils.uvcalibrate(
        data, true_gains_cal, inplace=False, time_check=False
    )
    data_orig.write_uvfits(f"{uvfits_output_dir}/ripple_gains_uncalib.uvfits")
    for ind in range(len(cal_filenames)):
        cal = pyuvdata.UVCal()
        cal.read_calfits(cal_filenames[ind])
        data_calibrated = pyuvdata.utils.uvcalibrate(
            data_orig, cal, inplace=False, time_check=False
        )
        data_calibrated.write_uvfits(uvfits_output_filenames[ind])
