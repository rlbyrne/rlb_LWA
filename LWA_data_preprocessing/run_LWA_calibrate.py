from LWA_calibrate import calibration_pipeline, peel_sources
from LWA_calibrate import peel_sources
import pyuvdata
import os
import numpy as np
import sys


def calibrate_Apr17():

    use_freqs = np.array(
        [
            "34",
            "44",
            "52",
            "62",
            "72",
            "79",
            "83",
        ]
    )
    date = "2026-04-07"
    hour = "12"
    filenames = []

    for freq in use_freqs:
        filenames = [
            f"/lustre/pipeline/cosmology/concatenated_data/{freq}MHz/2026-04-07/07/20260407_070009-070159_{freq}MHz.ms",
            f"/lustre/pipeline/cosmology/concatenated_data/{freq}MHz/2026-04-07/12/20260407_123010-123201_{freq}MHz.ms",
        ]
        for filename in filenames:
            os.system(f"sudo chmod -R a+r {filename}")
            calibration_pipeline(
                filename,
                output_dir="/fast/rbyrne",
                cal_trial_name="10h_cal",
                run_aoflagger=True,
                flag_antennas_from_autocorrs=True,
                apply_cal_path="/fast/rbyrne/calibration_2026-04-07_10h_spwcorrected.B.flagged",
                apply_calibration=True,
                smooth_cal=False,
                plot_images=True,
            )
            calibration_pipeline(
                filename,
                output_dir="/fast/rbyrne",
                cal_trial_name="10h_cal_smoothed",
                run_aoflagger=True,
                flag_antennas_from_autocorrs=True,
                apply_cal_path="/fast/rbyrne/calibration_2026-04-07_10h_spwcorrected.B.flagged",
                apply_calibration=True,
                smooth_cal=True,
                plot_images=True,
            )


def calibrate_Apr13():

    use_freqs = np.array(
        [
            "34",
            "44",
            "52",
            "62",
            "72",
            "79",
            "83",
        ]
    )
    date = "2026-04-07"
    hour = "12"

    for freq in use_freqs:
        filename = f"20260407_123010-123201_{freq}MHz.ms"
        os.system(
            f"cp -r /lustre/pipeline/cosmology/concatenated_data/{freq}MHz/{date}/{hour}/{filename} /fast/rbyrne/"
        )

        # Calibrate
        calibration_pipeline(
            f"/fast/rbyrne/{filename}",
            output_dir="/fast/rbyrne",
            run_aoflagger=True,
            flag_antennas_from_autocorrs=True,
            flag_antenna_list=[],
            plot_gains=True,
            apply_calibration=True,
            smooth_cal=False,
            plot_images=True,
        )


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
        concatenated_filepath = f"/fast/rbyrne/{filenames[0][:15]}-{filenames[-1][9:]}"
        for file_ind, filename in enumerate(filenames):
            uv_new = pyuvdata.UVData()
            uv_new.read(f"/fast/rbyrne/{hour}/{filename}")
            uv_new.select(polarizations=[-5, -6])
            uv_new.scan_number_array = None
            if file_ind == 0:
                uv_new.phase_to_time(np.min(uv_new.time_array))
                uv = uv_new
            else:
                uv_new.phase_to_time(np.min(uv.time_array))
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

        calfits_filenames.append(
            f"{os.path.splitext(os.path.basename(concatenated_filepath))[0]}.calfits"
        )

    # Concatenate calfits
    concatenated_calfits_filename = f"/fast/rbyrne/{calfits_filenames[0][:22]}_{use_freqs[0]}-{use_freqs[-1]}MHz.calfits"
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


def calibrate_Mar30():

    cal_freqs = np.array(
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
    data_freqs = ["34", "44", "52", "62", "72", "79", "83"]
    for freq_ind, freq in enumerate(cal_freqs):
        cal_new = pyuvdata.UVCal()
        cal_new.read(f"/fast/rbyrne/20250112_055752-055952_{freq}MHz.calfits")
        if freq_ind == 0:
            cal = cal_new
        else:
            cal = cal + cal_new

    # Average in time
    cal.flag_array[np.where(cal.flag_array)] = True
    cal.gain_array[np.where(cal.flag_array)] = np.nan + 1j * np.nan
    cal.Ntimes = 1
    cal.gain_array = np.nanmean(cal.gain_array, axis=2)[:, :, np.newaxis, :]
    cal.flag_array = np.min(cal.flag_array, axis=2)[:, :, np.newaxis, :]
    cal.integration_time = np.array([np.mean(cal.integration_time)])
    cal.lst_array = np.array([np.mean(cal.lst_array)])
    cal.time_array = np.array([np.mean(cal.time_array)])
    cal.check()

    cal.write_calfits(
        "/fast/rbyrne/20250112_055752-055952_27MHz-82MHz.calfits", clobber=True
    )

    if False:
        for freq in data_freqs:
            calibration_pipeline(
                f"/fast/rbyrne/20260112_120008-120158_{freq}MHz.ms",
                output_dir="/fast/rbyrne",
                cal_trial_name="05h",
                apply_cal_path="/fast/rbyrne/20250112_055752-055952_27MHz-82MHz.calfits",
                run_aoflagger=True,
                flag_antennas_from_autocorrs=True,
                flag_antenna_list=[],
                plot_gains=False,
                apply_calibration=True,
                smooth_cal=False,
                plot_images=True,
            )

    for freq in data_freqs:
        calibration_pipeline(
            f"/fast/rbyrne/20260112_120008-120158_{freq}MHz.ms",
            output_dir="/fast/rbyrne",
            cal_trial_name="05h_smoothed",
            apply_cal_path="/fast/rbyrne/20250112_055752-055952_27MHz-82MHz.calfits",
            run_aoflagger=True,
            flag_antennas_from_autocorrs=True,
            flag_antenna_list=[],
            plot_gains=False,
            apply_calibration=True,
            smooth_cal=True,
            plot_images=True,
        )


def convert_to_uvfits_Mar30():

    data_freqs = ["34", "44", "52", "62", "72", "79", "83"]
    for freq in data_freqs:
        for filename in [
            f"20260112_120008-120158_{freq}MHz_calibrated",
            f"20260112_120008-120158_{freq}MHz_05h_calibrated",
            f"20260112_120008-120158_{freq}MHz_05h_smoothed_calibrated",
        ]:
            uv = pyuvdata.UVData()
            uv.read(f"/fast/rbyrne/{filename}.ms")
            uv.write_uvfits(f"/fast/rbyrne/{filename}.uvfits", uvw_double=False)


def calibrate_May11():

    # use_freqs = ["34", "44", "52", "62", "72", "79", "83"]
    use_freqs = ["52", "62", "72", "79", "83"]
    for freq in use_freqs:
        # os.system(
        #    f"cp -r /lustre/pipeline/cosmology/concatenated_data/{freq}MHz/2026-04-19/11/20260419_112643-112833_{freq}MHz.ms /fast/rbyrne/20260407_123010-123201_{freq}MHz.ms"
        # )
        calibration_pipeline(
            f"/lustre/rbyrne/2026-04-19/20260419_112643-112833_{freq}MHz.ms",
            output_dir="/lustre/rbyrne/2026-04-19",
            tmp_dir="/fast/rbyrne",
            cal_trial_name="17h_cal",
            run_aoflagger=False,  # Changed because aoflagger was already run
            flag_antennas_from_autocorrs=True,
            flag_antenna_list=[],
            plot_gains=False,
            apply_cal_path="/lustre/pipeline/calibration/results/2026-04-19/17h/successful/20260430_233320/tables/calibration_2026-04-19_17h.B.flagged",
            flip_gain_conj=True,
            apply_calibration=True,
            smooth_cal=True,
            plot_images=True,
            peel=True,
        )


def test_peeling_May11():

    out_ms_path = "/fast/rbyrne/20260419_112643-112833_34MHz_17h_cal_tmp_dir/20260419_112643-112833_34MHz_17h_cal_peeled.ms"
    orig_ms_path = "/fast/rbyrne/20260419_112643-112833_34MHz_17h_cal_tmp_dir/20260419_112643-112833_34MHz.ms"
    peel_sources(out_ms_path, orig_ms_path, Ntimes=12)


def rerun_calibration_May15():

    use_freqs = ["34", "44"]
    for freq in use_freqs:
        # os.system(
        #    f"cp -r /lustre/pipeline/cosmology/concatenated_data/{freq}MHz/2026-04-19/11/20260419_112643-112833_{freq}MHz.ms /fast/rbyrne/20260407_123010-123201_{freq}MHz.ms"
        # )
        calibration_pipeline(
            f"/fast/rbyrne/20260419_112643-112833_{freq}MHz_17h_cal_tmp_dir/20260419_112643-112833_{freq}MHz.ms",
            output_dir="/lustre/rbyrne/2026-04-19",
            tmp_dir="/fast/rbyrne",
            cal_trial_name="17h_cal",
            run_aoflagger=False,  # Changed because aoflagger was already run
            flag_antennas_from_autocorrs=True,
            flag_antenna_list=[],
            plot_gains=False,
            apply_cal_path="/lustre/pipeline/calibration/results/2026-04-19/17h/successful/20260430_233320/tables/calibration_2026-04-19_17h.B.flagged",
            flip_gain_conj=True,
            apply_calibration=True,
            smooth_cal=True,
            plot_images=True,
            peel=True,
        )


def peel_horizon_sources_May15():

    # Run in environment ttcal_dev
    peel_sources(
        "/fast/rbyrne/20260419_112643-112833_34MHz_17h_cal_peeled_horizon.ms",
        "/fast/rbyrne/20260419_112643-112833_34MHz_17h_cal_peeled.ms",
        julia_path="/opt/devel/pipeline/envs/ttcal_dev/bin/julia",
        ttcal_path="/opt/devel/pipeline/envs/ttcal_dev/bin/ttcal.jl",
        peel_sources_path="/lustre/gh/calibration/pipeline/reference/sources/rfi_43.2_ver20251101.json",
        peel_mode="zest",
        beam="constant",
        maxiter=5,
        tolerance=1e-4,
        minuvw=5,
        Ntimes=12,
    )


def test_aoflagger_strategy_May18():

    calibration_pipeline(
        f"/lustre/rbyrne/2026-04-19/20260419_112643-112833_34MHz.ms",
        output_dir="/lustre/rbyrne/2026-04-19",
        tmp_dir="/fast/rbyrne",
        cal_trial_name="17h_cal_new_aoflagger_strategy",
        run_aoflagger=True,
        flag_antennas_from_autocorrs=True,
        flag_antenna_list=[],
        refresh_flags=True,
        plot_gains=False,
        apply_cal_path="/lustre/pipeline/calibration/results/2026-04-19/17h/successful/20260430_233320/tables/calibration_2026-04-19_17h.B.flagged",
        flip_gain_conj=True,
        apply_calibration=True,
        smooth_cal=True,
        plot_images=True,
        peel=True,
    )


def calibrate_May19():

    # use_freqs = ["34", "44", "52", "62", "72", "79", "83"]
    use_freqs = ["62", "72", "79", "83"]
    times = [
        "112643-112833",
        "112843-113033",
        "113043-113234",
        "113244-113434",
        "113444-113635",
    ]
    for freq in use_freqs:
        for time in times:
            calibration_pipeline(
                f"/lustre/pipeline/cosmology/concatenated_data/{freq}MHz/2026-04-19/11/20260419_{time}_{freq}MHz.ms",
                output_dir="/lustre/rbyrne/2026-04-19",
                tmp_dir="/fast/rbyrne",
                cal_trial_name="17h_cal",
                run_aoflagger=True,
                flag_antennas_from_autocorrs=True,
                flag_antenna_list=[],
                refresh_flags=True,
                plot_gains=False,
                apply_cal_path="/lustre/pipeline/calibration/results/2026-04-19/17h/successful/20260430_233320/tables/calibration_2026-04-19_17h.B.flagged",
                flip_gain_conj=True,
                apply_calibration=True,
                smooth_cal=True,
                plot_images=True,
                peel=True,
            )


def calibrate_with_calico_May22():

    use_freqs = [13, 18, 23, 27, 32, 36, 41, 46, 50, 55, 59, 64, 69, 73, 78, 82]
    for freq in use_freqs:
        calibration_pipeline(
            f"/fast/rbyrne/20260419_112543-112633_{freq}MHz.ms",
            output_dir="/lustre/rbyrne/2026-04-19",
            tmp_dir="/fast/rbyrne",
            cal_trial_name="calico",
            run_aoflagger=True,
            flag_antennas_from_autocorrs=True,
            flag_antenna_list=[],
            refresh_flags=True,
            plot_gains=False,
            flip_gain_conj=False,
            apply_calibration=False,
            plot_images=False,
            peel=False,
        )


def apply_calibration_May28():

    use_freqs = [13, 18, 23, 27, 32, 36, 41, 46, 50, 55, 59, 64, 69, 73, 78, 82]
    for freq in use_freqs:
        calibration_pipeline(
            f"/lustre/rbyrne/2026-04-19/20260419_112543-112633_{freq}MHz.ms",
            output_dir="/lustre/rbyrne/2026-04-19",
            tmp_dir="/fast/rbyrne",
            cal_trial_name="calico",
            apply_cal_path=f"/lustre/rbyrne/2026-04-19/20260419_112543-112633_{freq}MHz_calico.calfits",
            run_aoflagger=False,  # Already run
            flag_antennas_from_autocorrs=False,  # Already run
            flag_antenna_list=[],
            refresh_flags=True,
            plot_gains=False,
            flip_gain_conj=False,
            apply_calibration=True,
            plot_images=True,
            peel=False,
        )
        calibration_pipeline(
            f"/lustre/rbyrne/2026-04-19/20260419_112543-112633_{freq}MHz.ms",
            output_dir="/lustre/rbyrne/2026-04-19",
            tmp_dir="/fast/rbyrne",
            cal_trial_name="17h_cal",
            apply_cal_path=f"/fast/rbyrne/calibration_2026-04-19_17h.B.flagged",
            run_aoflagger=False,  # Already run
            flag_antennas_from_autocorrs=False,  # Already run
            flag_antenna_list=[],
            refresh_flags=True,
            plot_gains=False,
            flip_gain_conj=True,
            apply_calibration=True,
            plot_images=True,
            peel=False,
        )


def calibrate_with_calico_new_beam_May28():

    use_freqs = [13, 18, 23, 27, 32, 36, 41, 46, 50, 55, 59, 64, 69, 73, 78, 82]
    for freq in use_freqs:
        calibration_pipeline(
            f"/lustre/rbyrne/2026-04-19/20260419_112543-112633_{freq}MHz.ms",
            output_dir="/lustre/rbyrne/2026-04-19",
            tmp_dir="/fast/rbyrne",
            cal_trial_name="calico_updated_beam",
            beam_path="/fast/rbyrne/OVRO_LWA_MROsoil_updatedheight.fits",
            run_aoflagger=True,
            flag_antennas_from_autocorrs=True,
            flag_antenna_list=[],
            refresh_flags=True,
            plot_gains=False,
            flip_gain_conj=False,
            apply_calibration=False,
            plot_images=False,
            peel=False,
        )


def image_data_Jun4():

    date = "2026-04-19"
    hours = ["05", "06", "07", "08", "11"]
    use_freqs = [
        "34",
        "44",
        "52",
        "62",
        "72",
        "79",
        "83",
    ]
    for hour in hours:
        for freq in use_freqs:
            filenames = os.listdir(
                f"/lustre/pipeline/cosmology/concatenated_data/{freq}MHz/{date}/{hour}"
            )
            for filename in filenames:
                calibration_pipeline(
                    f"/lustre/pipeline/cosmology/concatenated_data/{freq}MHz/{date}/{hour}/{filename}",
                    output_dir="/lustre/rbyrne/2026-04-19",
                    tmp_dir="/fast/rbyrne",
                    cal_trial_name="17h_cal",
                    apply_cal_path="/fast/rbyrne/calibration_2026-04-19_17h.B.flagged",
                    run_aoflagger=True,
                    flag_antennas_from_autocorrs=True,
                    flag_antenna_list=[],
                    refresh_flags=True,
                    plot_gains=False,
                    flip_gain_conj=True,
                    apply_calibration=True,
                    plot_images=True,
                    peel=True,
                )


def recalibrate_data_Jun30():

    calibration_pipeline(
        f"/lustre/rbyrne/2026-04-19/20260419_055641-055832_44MHz_17h_cal_calibrated.ms",
        output_dir="/lustre/rbyrne/2026-04-19",
        tmp_dir="/fast/rbyrne",
        cal_trial_name="recalib",
        beam_path="/fast/rbyrne/OVRO_LWA_MROsoil_updatedheight.fits",
        run_aoflagger=True,
        flag_antennas_from_autocorrs=True,
        flag_antenna_list=[],
        refresh_flags=True,
        refresh_calibration=False,
        plot_gains=True,
        flip_gain_conj=False,
        apply_calibration=True,
        plot_images=True,
        peel=True,
    )


def recalibrate_data_with_diffuse_Jun30():

    calibration_pipeline(
        f"/lustre/rbyrne/2026-04-19/20260419_055641-055832_44MHz_17h_cal_calibrated.ms",
        output_dir="/lustre/rbyrne/2026-04-19",
        tmp_dir="/fast/rbyrne",
        cal_trial_name="recalib_diffuse",
        beam_path="/fast/rbyrne/OVRO_LWA_MROsoil_updatedheight.fits",
        include_diffuse=True,
        diffuse_skymodel_path="/lustre/rbyrne/skymodels/ovro_lwa_sky_map_46.992MHz_nside512.skyh5",
        run_aoflagger=True,
        flag_antennas_from_autocorrs=True,
        flag_antenna_list=[],
        refresh_flags=True,
        plot_gains=True,
        flip_gain_conj=False,
        apply_calibration=True,
        plot_images=True,
        peel=True,
    )


if __name__ == "__main__":
    fn_name = sys.argv[1]
    fn = globals().get(fn_name)
    if callable(fn):
        fn()
    else:
        print(f"No function named '{fn_name}'")
