import pyuvdata
from pyuvdata import utils

data_path = "/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Apr2022"
data_use_model = True
obsid = "1061316296"
pol = "XX"

data = pyuvdata.UVData()
filelist = [
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
data.read_fhd(filelist, use_model=data_use_model)

uvfits_output_dir = "/safepool/rbyrne/calibration_outputs/uvfits"

# Unity gains
cal_filenames = [
    "/safepool/rbyrne/calibration_outputs/caltest_Apr12/vanilla_cal.calfits",
    "/safepool/rbyrne/calibration_outputs/caltest_Apr12/wedge_excluded.calfits",
]
uvfits_output_filenames = [
    f"{uvfits_output_dir}/unity_gains_diagonal.uvfits",
    f"{uvfits_output_dir}/unity_gains_dwcal.uvfits",
]
data.write_uvfits(f"{uvfits_output_dir}/unity_gains_uncalib.uvfits")
for ind in range(len(cal_filenames)):
    cal = pyuvdata.UVCal()
    cal.read_calfits(cal_filenames[ind])
    data_calibrated = pyuvdata.utils.uvcalibrate(
        data, cal, inplace=False, time_check=False
    )
    data_calibrated.write_uvfits(uvfits_output_filenames[ind])

# Random gains
cal_filenames = [
    "/safepool/rbyrne/calibration_outputs/random_gains_test_Apr25/vanilla_cal.calfits",
    "/safepool/rbyrne/calibration_outputs/random_gains_test_Apr25/wedge_excluded.calfits",
]
uvfits_output_filenames = [
    f"{uvfits_output_dir}/random_gains_diagonal.uvfits",
    f"{uvfits_output_dir}/random_gains_dwcal.uvfits",
]
true_gains_calfits = "/safepool/rbyrne/calibration_outputs/random_gains_test_Apr25/random_initial_gains.calfits"
true_gains_cal = pyuvdata.UVCal()
true_gains_cal.read_calfits(true_gains_calfits)
data_calibrated = pyuvdata.utils.uvcalibrate(
    data, true_gains_cal, inplace=False, time_check=False
)
data_calibrated.write_uvfits(f"{uvfits_output_dir}/random_gains_uncalib.uvfits")
for ind in range(len(cal_filenames)):
    cal = pyuvdata.UVCal()
    cal.read_calfits(cal_filenames[ind])
    pyuvdata.utils.uvcalibrate(
        data_calibrated, cal, inplace=True, time_check=False
    )
    data_calibrated.write_uvfits(uvfits_output_filenames[ind])

# Ripple gains
cal_filenames = [
    "/safepool/rbyrne/calibration_outputs/gain_ripple_test_May6/vanilla_cal.calfits",
    "/safepool/rbyrne/calibration_outputs/gain_ripple_test_May6/wedge_excluded.calfits",
]
uvfits_output_filenames = [
    f"{uvfits_output_dir}/ripple_gains_diagonal.uvfits",
    f"{uvfits_output_dir}/ripple_gains_dwcal.uvfits",
]
true_gains_calfits = "/safepool/rbyrne/calibration_outputs/gain_ripple_test_May6/ripple_initial_gains.calfits"
true_gains_cal = pyuvdata.UVCal()
true_gains_cal.read_calfits(true_gains_calfits)
data_calibrated = pyuvdata.utils.uvcalibrate(
    data, true_gains_cal, inplace=False, time_check=False
)
data_calibrated.write_uvfits(f"{uvfits_output_dir}/ripple_gains_uncalib.uvfits")
for ind in range(len(cal_filenames)):
    cal = pyuvdata.UVCal()
    cal.read_calfits(cal_filenames[ind])
    pyuvdata.utils.uvcalibrate(
        data_calibrated, cal, inplace=True, time_check=False
    )
    data_calibrated.write_uvfits(uvfits_output_filenames[ind])