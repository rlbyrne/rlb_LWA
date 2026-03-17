from LWA_calibrate import calibration_pipeline
import pyuvdata
import os

def calibrate_Mar16():

    use_freqs = np.array(
        [
            "27",
            "32",
            "36",
            "41",
            "46",
            "50",
            "55",
            "59",
            "64",
            "69",
            "73",
            "78",
            "82",
        ]
    )
    date = "2025-01-12"
    hour = "05"

    calfits_filenames = []
    for freq in use_freqs:
        os.system(
            f"cp -r /lustre/pipeline/calibration/{freq}MHz/{date}/{hour}/ /fast/rbyrne/"
        )

        # Concatenate files
        filenames = np.sort(os.listdir(f"/fast/rbyrne/{hour}/"))
        concatenated_filepath = f"/fast/rbyrne/{filenames[0][:15]}-{filenames[-1][15:]}"
        for file_ind, filename in enumerate(filenames):
            uv_new = pyuvdata.UVData()
            uv_new.read(f"/fast/rbyrne/{hour}/{filename}")
            uv_new.select(polarizations=[-5, -6])
            if file_ind == 0:
                uv_new.phase_to_time(np.min(uv_new.time_array))
                uv = uv_new
            else:
                uv_freq_new.phase_to_time(np.min(uv.time_array))
                uv.fast_concat(uv_new, "blt", inplace=True, run_check=False)
        uv.write_ms(concatenated_filepath)
        if os.path.isdir(concatenated_filepath):
            os.system(f"rm -r /fast/rbyrne/{hour}/")  # Delete individual files

        # Calibrate
        calibration_pipeline(
            concatenated_filepath,
            output_dir="/fast/rbyrne",
            run_aoflagger=True,
            flag_antennas_from_autocorrs=True,
            flag_antenna_list=[],
            plot_gains=True,
            apply_calibration=True,
            smooth_cal=False,
            plot_images=True,
        )

        calfits_filenames.append(f"/fast/rbyrne/{os.path.splitext(os.path.basename(concatenated_filepath))[0]}.calfits")

    # Concatenate calfits
    concatenated_calfits_filename = f"/fast/rbyrne/{calfits_filenames[0][:20]}_{np.min(use_freqs)}-{np.max(use_freqs)}MHz.calfits"
    for file_ind, filename in enumerate(calfits_filenames):
        cal_new = pyuvdata.UVCal()
        cal_new.read(filename)
        if file_ind == 0:
            cal = cal_new
        else:
            cal = cal + cal_new
    cal.write_calfits(concatenated_calfits_filename)
    if os.path.isfile(concatenated_calfits_filename):  # Delete calfits
        for filename in enumerate(calfits_filenames):
            os.system(f"rm {filename}")

if __name__=="__main__":
    calibrate_Mar16()