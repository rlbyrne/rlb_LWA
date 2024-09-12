import os
import pyuvdata

calfits_files = [file for file in os.listdir("/lustre/rbyrne/2024-03-02/ruby/calibration_outputs") if file.endswith(".calfits")]
for use_file in calfits_files:
    cal = pyuvdata.UVCal()
    cal.read_calfits(f"/lustre/rbyrne/2024-03-02/ruby/calibration_outputs/{use_file}")
    cal.gain_convention = "divide"
    cal.gain_array = 1 / cal.gain_array
    filename = use_file.removesuffix(".calfits")
    cal.write_ms_cal(f"/lustre/rbyrne/2024-03-02/ruby/calibration_outputs/{filename}.bcal")