import LWA_preprocessing
import pyuvdata
import os
import numpy as np


def ssins_flagging_Mar25():

    data_path = "/lustre/rbyrne/LWA_data_20220210"
    output_path = "/lustre/rbyrne/LWA_data_20220210/uvfits_ssins_flagged"
    ssins_plot_save_path = "/lustre/rbyrne/LWA_data_20220210/ssins_plots"
    autos_plot_save_path = "/lustre/rbyrne/LWA_data_20220210/autocorrelation_plots"

    filenames = os.listdir(data_path)
    filenames = [file for file in filenames if file.endswith(".uvfits")]

    for file in filenames:

        file_split = ".".join(file.split(".")[:-1])
        autos_all_ants_plot_prefix = f"{file_split}_all_ants"
        plot_prefix = f"{file_split}"

        uvd = pyuvdata.UVData()
        uvd.read(f"{data_path}/{file}")

        LWA_preprocessing.plot_autocorrelations(
            uvd,
            plot_save_path=autos_plot_save_path,
            plot_file_prefix=autos_all_ants_plot_prefix,
            time_average=True,
            plot_legend=False,
        )

        LWA_preprocessing.flag_inactive_antennas(
            uvd, autocorr_thresh=10.0, inplace=True
        )

        LWA_preprocessing.plot_autocorrelations(
            uvd,
            plot_save_path=autos_plot_save_path,
            plot_file_prefix=plot_prefix,
            time_average=True,
            plot_legend=False,
        )

        LWA_preprocessing.ssins_flagging(
            uvd,
            sig_thresh=1,  # Flagging threshold in std. dev.
            inplace=True,
            plot_no_flags=True,
            plot_orig_flags=True,
            plot_ssins_flags=True,
            plot_save_path=ssins_plot_save_path,
            plot_file_prefix=plot_prefix,
        )
        uvd.write_uvfits(f"{output_path}/{file}")


def ssins_flagging_Mar31():

    data_path = "/lustre/rbyrne/LWA_data_20220210"
    output_path = "/lustre/rbyrne/LWA_data_20220210/uvfits_ssins_flagged"
    ssins_plot_save_path = "/lustre/rbyrne/LWA_data_20220210/ssins_plots"
    autos_plot_save_path = "/lustre/rbyrne/LWA_data_20220210/autocorrelation_plots"

    # Find raw ms files
    subbands = ["70MHz"]
    start_time_stamp = 191447
    end_time_stamp = 194824
    nfiles_per_uvfits = 12

    for subband in subbands:

        ms_filenames = os.listdir(data_path)
        ms_filenames = [
            file
            for file in ms_filenames
            if (file.endswith(".ms.tar") and subband in file)
        ]
        ms_filenames = sorted(
            [
                file
                for file in ms_filenames
                if (
                    int(file.split("_")[1]) >= start_time_stamp
                    and int(file.split("_")[1]) <= end_time_stamp
                )
            ]
        )

        ms_filenames_grouped = []
        file_ind = 0
        while file_ind < len(ms_filenames):
            max_ind = np.min([file_ind + nfiles_per_uvfits, len(ms_filenames)])
            file_group = ms_filenames[file_ind:max_ind]
            ms_filenames_grouped.append(file_group)
            file_ind += nfiles_per_uvfits

        # Process files
        for file_group in range(len(ms_filenames_grouped)):

            file_split = ms_filenames_grouped[file_group][0].split(".")[0]
            autos_all_ants_plot_prefix = f"{file_split}_all_ants"
            plot_prefix = f"{file_split}"
            autos_with_flags_plot_prefix = f"{file_split}_with_flags"

            ms_group_full_paths = [
                f"{data_path}/{file}" for file in ms_filenames_grouped[file_group]
            ]
            uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(ms_group_full_paths)

            # Plot autocorrelations before any preprocessing
            LWA_preprocessing.plot_autocorrelations(
                uvd,
                plot_save_path=autos_plot_save_path,
                plot_file_prefix=autos_all_ants_plot_prefix,
                time_average=True,
                plot_legend=False,
            )

            # Remove inactive antennas
            LWA_preprocessing.flag_inactive_antennas(
                uvd, autocorr_thresh=10.0, inplace=True
            )

            # Replot autocorrelations with antenna removal
            LWA_preprocessing.plot_autocorrelations(
                uvd,
                plot_save_path=autos_plot_save_path,
                plot_file_prefix=plot_prefix,
                time_average=True,
                plot_legend=False,
            )

            # RFI flagging
            LWA_preprocessing.ssins_flagging(
                uvd,
                sig_thresh=20.0,  # Flagging threshold in std. dev.
                inplace=True,
                plot_no_flags=True,
                plot_orig_flags=True,
                plot_ssins_flags=True,
                plot_save_path=ssins_plot_save_path,
                plot_file_prefix=plot_prefix,
            )

            # Replot autocorrelations with flagging
            LWA_preprocessing.plot_autocorrelations(
                uvd,
                plot_save_path=autos_plot_save_path,
                plot_file_prefix=autos_with_flags_plot_prefix,
                time_average=True,
                plot_legend=False,
                plot_flagged_data=False,
            )

            # Write uvfits
            uvd.write_uvfits(
                f"{output_path}/{file_split}.uvfits",
                force_phase=True,
                spoof_nonessential=True,
            )


def process_data_Apr6():

    data_path = "/lustre/data/stageiii/20220304/feon"
    output_dir = "/lustre/rbyrne/LWA_data_20220304"
    output_path = f"{output_dir}/uvfits_ssins_flagged"
    ssins_plot_save_path = f"{output_dir}/ssins_plots"
    ssins_flags_save_path = f"{output_dir}/ssins_flags"
    autos_plot_save_path = f"{output_dir}/autocorrelation_plots"

    # Find raw ms files
    subbands = [
        "15MHz",
        "20MHz",
        "24MHz",
        "29MHz",
        "34MHz",
        "38MHz",
        "43MHz",
        "47MHz",
        "52MHz",
        "57MHz",
        "61MHz",
        "66MHz",
        "70MHz",
        "75MHz",
        "80MHz",
        "84MHz",
    ]
    start_time_stamp = 181704
    end_time_stamp = 181754
    nfiles_per_uvfits = 12

    for subband in subbands:

        ms_filenames = os.listdir(data_path)
        ms_filenames = [
            file
            for file in ms_filenames
            if (file.endswith(".ms.tar") and subband in file)
        ]
        ms_filenames = sorted(
            [
                file
                for file in ms_filenames
                if (
                    int(file.split("_")[1]) >= start_time_stamp
                    and int(file.split("_")[1]) <= end_time_stamp
                )
            ]
        )

        ms_filenames_grouped = []
        file_ind = 0
        while file_ind < len(ms_filenames):
            max_ind = np.min([file_ind + nfiles_per_uvfits, len(ms_filenames)])
            file_group = ms_filenames[file_ind:max_ind]
            ms_filenames_grouped.append(file_group)
            file_ind += nfiles_per_uvfits

        # Process files
        for file_group in range(len(ms_filenames_grouped)):

            file_split = ms_filenames_grouped[file_group][0].split(".")[0]
            autos_all_ants_plot_prefix = f"{file_split}_all_ants"
            plot_prefix = f"{file_split}"
            autos_with_flags_plot_prefix = f"{file_split}_with_flags"
            save_flags_filepath = f"{ssins_flags_save_path}/{file_split}_flags.hdf5"

            ms_group_full_paths = [
                f"{data_path}/{file}" for file in ms_filenames_grouped[file_group]
            ]
            uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(
                ms_group_full_paths,
                untar_dir=output_dir,
            )

            # Plot autocorrelations before any preprocessing
            LWA_preprocessing.plot_autocorrelations(
                uvd,
                plot_save_path=autos_plot_save_path,
                plot_file_prefix=autos_all_ants_plot_prefix,
                time_average=True,
                plot_legend=False,
            )

            # Remove inactive antennas
            LWA_preprocessing.flag_inactive_antennas(
                uvd, autocorr_thresh=10.0, inplace=True
            )

            # Replot autocorrelations with antenna removal
            LWA_preprocessing.plot_autocorrelations(
                uvd,
                plot_save_path=autos_plot_save_path,
                plot_file_prefix=plot_prefix,
                time_average=True,
                plot_legend=False,
            )

            # RFI flagging
            LWA_preprocessing.ssins_flagging(
                uvd,
                sig_thresh=20.0,  # Flagging threshold in std. dev.
                inplace=True,
                save_flags_filepath=save_flags_filepath,
                plot_no_flags=True,
                plot_orig_flags=False,
                plot_ssins_flags=True,
                plot_save_path=ssins_plot_save_path,
                plot_file_prefix=plot_prefix,
            )

            # Write uvfits
            uvd.write_uvfits(
                f"{output_path}/{file_split}.uvfits",
                force_phase=True,
                spoof_nonessential=True,
            )


def pipeline_comparison_Apr8():

    ms_filename = (
        "/lustre/mmanders/stageiii/phaseiii/20220307/calint/20220307_175923_61MHz.ms"
    )
    output_dir = "/lustre/rbyrne/LWA_data_20220307"
    output_path = f"{output_dir}/uvfits"

    uvd_uncalib = LWA_preprocessing.convert_raw_ms_to_uvdata(
        ms_filename, untar_dir=output_dir, data_column="DATA"
    )

    # Write uvfits
    uvd_uncalib.write_uvfits(
        f"{output_path}/20220307_175923_61MHz_uncalib.uvfits",
        force_phase=True,
        spoof_nonessential=True,
    )

    uvd_calib = LWA_preprocessing.convert_raw_ms_to_uvdata(
        ms_filename, untar_dir=output_dir, data_column="CORRECTED_DATA"
    )

    # Write uvfits
    uvd_calib.write_uvfits(
        f"{output_path}/20220307_175923_61MHz_calib.uvfits",
        force_phase=True,
        spoof_nonessential=True,
    )


def process_data_Apr15():

    ms_filename = (
        "/lustre/mmanders/stageiii/phaseiii/20220307/calint/20220307_175923_61MHz.ms"
    )
    output_dir = "/lustre/rbyrne/LWA_data_20220307"
    output_path = f"{output_dir}/uvfits"

    uvd_uncalib = LWA_preprocessing.convert_raw_ms_to_uvdata(
        ms_filename, untar_dir=output_dir, data_column="DATA"
    )

    LWA_preprocessing.flag_outriggers(
        uvd_uncalib,
        inplace=True,
    )

    # Write uvfits
    uvd_uncalib.write_uvfits(
        f"{output_path}/20220307_175923_61MHz_uncalib.uvfits",
        force_phase=True,
        spoof_nonessential=True,
    )


def reprocessing_Apr25():

    data_dir = "/lustre/rbyrne/LWA_data_20220210"
    output_uvfits_dir = f"{data_dir}/uvfits_ssins_flagged"
    ssins_plot_dir = f"{data_dir}/ssins_plots"
    ssins_flags_dir = f"{data_dir}/ssins_flags"
    autos_plot_dir = f"{data_dir}/autocorrelation_plots"

    # Find raw ms files
    subbands = ["70MHz"]
    ssins_thresholds = [1, 5, 10, 20]
    start_time_stamp = 191447
    end_time_stamp = 194824
    nfiles_per_uvfits = 12

    for subband in subbands:

        ms_filenames = os.listdir(data_dir)
        ms_filenames = [
            file
            for file in ms_filenames
            if (file.endswith(".ms.tar") and subband in file)
        ]
        ms_filenames = sorted(
            [
                file
                for file in ms_filenames
                if (
                    int(file.split("_")[1]) >= start_time_stamp
                    and int(file.split("_")[1]) <= end_time_stamp
                )
            ]
        )

        ms_filenames_grouped = []
        file_ind = 0
        while file_ind < len(ms_filenames):
            max_ind = np.min([file_ind + nfiles_per_uvfits, len(ms_filenames)])
            file_group = ms_filenames[file_ind:max_ind]
            ms_filenames_grouped.append(file_group)
            file_ind += nfiles_per_uvfits

        # Process files
        for file_group in range(len(ms_filenames_grouped)):

            file_split = ms_filenames_grouped[file_group][0].split(".")[0]
            autos_all_ants_plot_prefix = f"{file_split}_all_ants"
            plot_prefix = f"{file_split}"
            autos_with_flags_plot_prefix = f"{file_split}_with_flags"

            ms_group_full_paths = [
                f"{data_dir}/{file}" for file in ms_filenames_grouped[file_group]
            ]
            uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(ms_group_full_paths)

            # Plot autocorrelations before any preprocessing
            LWA_preprocessing.plot_autocorrelations(
                uvd,
                plot_save_dir=autos_plot_dir,
                plot_file_prefix=f"{file_split}_all_ants",
                time_average=True,
                plot_legend=False,
                plot_flagged_data=False,
                plot_antennas_individually=False,
            )

            # Flag outriggers
            LWA_preprocessing.flag_outriggers(uvd, inplace=True)

            # Flag inactive antennas
            LWA_preprocessing.flag_inactive_antennas(
                uvd, autocorr_thresh=20.0, inplace=True, flag_only=True
            )

            # Replot autocorrelations with antenna flagging
            LWA_preprocessing.plot_autocorrelations(
                uvd,
                plot_save_dir=autos_plot_dir,
                plot_file_prefix=f"{file_split}_ant_flags",
                time_average=True,
                plot_legend=False,
                plot_flagged_data=False,
                plot_antennas_individually=False,
            )

            plot_orig_flags = True
            for ssins_thresh in ssins_thresholds:

                # RFI flagging
                uvd_ssins_flagged = LWA_preprocessing.ssins_flagging(
                    uvd,
                    sig_thresh=ssins_thresh,  # Flagging threshold in std. dev.
                    inplace=False,
                    save_flags_filepath=f"{ssins_flags_dir}/{file_split}_flags_thresh_{ssins_thresh}.hdf5",
                    plot_no_flags=plot_orig_flags,
                    plot_orig_flags=True,
                    plot_ssins_flags=True,
                    plot_save_dir=ssins_plot_dir,
                    plot_file_prefix=f"{file_split}",
                )
                plot_orig_flags = False

                # Replot autocorrelations with flagging
                LWA_preprocessing.plot_autocorrelations(
                    uvd_ssins_flagged,
                    plot_save_dir=autos_plot_dir,
                    plot_file_prefix=f"{file_split}_ssins_flagged_thresh_{ssins_thresh}",
                    time_average=True,
                    plot_legend=False,
                    plot_flagged_data=False,
                    plot_antennas_individually=False,
                )

                # Write uvfits
                uvd_ssins_flagged.write_uvfits(
                    f"{output_uvfits_dir}/{file_split}_ssins_thresh_{ssins_thresh}.uvfits",
                    force_phase=True,
                    spoof_nonessential=True,
                )


def get_rfi_occupancy_Jun6():

    data_dir = "/lustre/rbyrne/LWA_data_20220210"
    uvfits_dir = f"{data_dir}/uvfits_ssins_flagged"
    txtfilepath = "/lustre/rbyrne/LWA_data_20220210/ssins_occupany.txt"
    txtfile = open(txtfilepath, "w")

    # Find raw ms files
    ssins_thresholds = [1, 5, 10, 20]

    filenames = os.listdir(uvfits_dir)
    filenames = [file for file in filenames if file.endswith("thresh_20.uvfits")]
    filenames = [file for file in filenames if "20220210_194804" not in file]

    # Process files
    for filename in filenames:

        obsid = filename[0:15]

        uvd = pyuvdata.UVData()
        uvd.read_uvfits(f"{uvfits_dir}/{filename}")
        uvd.flag_array[:, :, :, :] = False  # Unflag all

        # Flag outriggers
        LWA_preprocessing.flag_outriggers(uvd, inplace=True)

        # Flag inactive antennas
        LWA_preprocessing.flag_inactive_antennas(
            uvd, autocorr_thresh=20.0, inplace=True, flag_only=True
        )

        for ssins_thresh in ssins_thresholds:

            # RFI flagging
            uvd_ssins_flagged = LWA_preprocessing.ssins_flagging(
                uvd,
                sig_thresh=ssins_thresh,  # Flagging threshold in std. dev.
                inplace=False,
            )
            flagging_frac = np.sum(uvd_ssins_flagged.flag_array) / np.size(
                uvd_ssins_flagged.flag_array
            )
            txtfile.write(f"{obsid}, {ssins_thresh}, {flagging_frac}\n")

    txtfile.close()


def plot_autocorrelations_Aug16():

    data_dir = "/safepool/rbyrne/lwa_data"
    autos_plot_dir = f"{data_dir}/autocorrelation_plots"

    # Find raw ms files
    filenames = os.listdir(data_dir)
    ms_filenames = [filename for filename in filenames if filename.endswith(".ms")]
    timestamps = np.unique(np.array([filename[:15] for filename in ms_filenames]))
    for time in timestamps:
        use_filenames = [
            filename for filename in ms_filenames if filename.startswith(time)
        ]
        uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(
            [f"{data_dir}/{filename}" for filename in use_filenames]
        )
        LWA_preprocessing.plot_autocorrelations(
            uvd,
            plot_save_dir=autos_plot_dir,
            plot_file_prefix=time,
            time_average=True,
            plot_legend=False,
            plot_flagged_data=False,
            yrange=[0, 100],
            plot_antennas_together=False,
            plot_antennas_individually=True,
        )


def ssins_flagging_Aug26():

    data_dir = "/safepool/rbyrne/lwa_data"
    ssins_flags_dir = f"{data_dir}/ssins_flags"
    ssins_plot_dir = f"{data_dir}/ssins_plots"

    # Find raw ms files
    filenames = os.listdir(data_dir)
    ms_filenames = [filename for filename in filenames if filename.endswith(".ms")]
    uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(
        [f"{data_dir}/{filename}" for filename in ms_filenames]
    )
    uvd.phase_to_time(np.mean(uvd.time_array))
    LWA_preprocessing.flag_outriggers(uvd, inplace=True)
    uvd.write_uvfits(
        f"{data_dir}/20220812_000008_000158.uvfits",
        force_phase=True,
        spoof_nonessential=True,
    )

    plot_orig_flags = True
    plot_no_flags = True
    for ssins_thresh in [5, 10, 20]:
        uvd_ssins_flagged = LWA_preprocessing.ssins_flagging(
            uvd,
            sig_thresh=ssins_thresh,  # Flagging threshold in std. dev.
            inplace=False,
            save_flags_filepath=f"{ssins_flags_dir}/flags_thresh_{ssins_thresh}.hdf5",
            plot_no_flags=plot_no_flags,
            plot_orig_flags=plot_orig_flags,
            plot_ssins_flags=True,
            plot_save_dir=ssins_plot_dir,
            plot_file_prefix=f"20220812_000008_000158",
        )
        flagging_frac = (
            np.sum(uvd_ssins_flagged.flag_array) - np.sum(uvd.flag_array)
        ) / (np.size(uvd.flag_array) - np.sum(uvd.flag_array))
        print(f"Flagging fraction {flagging_frac} at SSINS threshold {ssins_thresh}")
        plot_orig_flags = False
        plot_no_flags = False


def plot_autocorrelations_Aug30():

    data_dir = "/safepool/rbyrne/lwa_data"
    autos_plot_dir = f"{data_dir}/autocorrelation_plots"

    uvd = pyuvdata.UVData()
    uvd.read(f"{data_dir}/20220812_000008_000158.uvfits", ant_str="auto")

    LWA_preprocessing.plot_autocorrelation_waterfalls(
        uvd,
        plot_save_dir=autos_plot_dir,
        plot_file_prefix="20220812_000008_000158",
        plot_flagged_data=False,
        colorbar_range=[0, 100],
    )

    LWA_preprocessing.plot_autocorrelations(
        uvd,
        plot_save_dir=autos_plot_dir,
        plot_file_prefix="20220812_000008_000158",
        time_average=True,
        plot_flagged_data=False,
        yrange=[0, 100],
    )


def flagging_Sept6():

    data_dir = "/safepool/rbyrne/lwa_data"
    ssins_flags_dir = f"{data_dir}/ssins_flags"
    ssins_plot_dir = f"{data_dir}/ssins_plots"

    uvd = pyuvdata.UVData()
    uvd.read(f"{data_dir}/20220812_000008_000158.uvfits")

    LWA_preprocessing.flag_antennas(
        uvd,
        antenna_names=[
            "LWA009",
            "LWA147",
            "LWA187",
            "LWA038",
            "LWA149",
            "LWA195",
            "LWA040",
            "LWA150",
            "LWA211",
            "LWA041",
            "LWA151",
            "LWA212",
            "LWA071",
            "LWA176",
            "LWA215",
            "LWA095",
            "LWA177",
            "LWA221",
            "LWA108",
            "LWA178",
            "LWA246",
            "LWA109",
            "LWA179",
            "LWA247",
            "LWA126",
            "LWA180",
        ],
        inplace=True,
    )

    plot_orig_flags = True
    plot_no_flags = True
    for ssins_thresh in [5, 10, 20]:
        uvd_ssins_flagged = LWA_preprocessing.ssins_flagging(
            uvd,
            sig_thresh=ssins_thresh,  # Flagging threshold in std. dev.
            inplace=False,
            save_flags_filepath=f"{ssins_flags_dir}/20220812_000008_000158_flags_thresh_{ssins_thresh}.hdf5",
            plot_no_flags=plot_no_flags,
            plot_orig_flags=plot_orig_flags,
            plot_ssins_flags=True,
            plot_save_dir=ssins_plot_dir,
            plot_file_prefix=f"20220812_000008_000158",
        )
        flagging_frac = (
            np.sum(uvd_ssins_flagged.flag_array) - np.sum(uvd.flag_array)
        ) / (np.size(uvd.flag_array) - np.sum(uvd.flag_array))
        print(f"Flagging fraction {flagging_frac} at SSINS threshold {ssins_thresh}")
        plot_orig_flags = False
        plot_no_flags = False


def ssins_flagging_Nov28():

    data_dir = "/data06/slow"
    data_output_dir = "/home/rbyrne/lwa_testing_Nov2022"
    ssins_flags_dir = "/home/rbyrne/lwa_testing_Nov2022"
    ssins_plot_dir = "/home/rbyrne/lwa_testing_Nov2022"

    # Find raw ms files
    ms_filenames = [
        "20221128_052946_70MHz.ms",
        "20221128_052956_70MHz.ms",
        "20221128_053006_70MHz.ms",
        "20221128_053016_70MHz.ms",
        "20221128_053026_70MHz.ms",
        "20221128_053036_70MHz.ms",
        "20221128_053046_70MHz.ms",
        "20221128_053056_70MHz.ms",
        "20221128_053106_70MHz.ms",
        "20221128_053116_70MHz.ms",
        "20221128_053126_70MHz.ms",
        "20221128_053136_70MHz.ms",
    ]
    uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(
        [f"{data_dir}/{filename}" for filename in ms_filenames]
    )
    uvd.phase_to_time(np.mean(uvd.time_array))
    LWA_preprocessing.flag_outriggers(uvd, inplace=True)

    ssins_thresh = 10.0
    uvd_ssins_flagged = LWA_preprocessing.ssins_flagging(
        uvd,
        sig_thresh=ssins_thresh,  # Flagging threshold in std. dev.
        inplace=False,
        save_flags_filepath=f"{ssins_flags_dir}/flags_thresh_{ssins_thresh}.hdf5",
        plot_no_flags=True,
        plot_orig_flags=True,
        plot_ssins_flags=True,
        plot_save_dir=ssins_plot_dir,
        plot_file_prefix=f"20221128_052946_70MHz",
    )
    flagging_frac = (np.sum(uvd_ssins_flagged.flag_array) - np.sum(uvd.flag_array)) / (
        np.size(uvd.flag_array) - np.sum(uvd.flag_array)
    )
    print(f"Flagging fraction {flagging_frac} at SSINS threshold {ssins_thresh}")

    uvd_ssins_flagged.write_uvfits(
        f"{data_output_dir}/20221128_052946_70MHz.uvfits",
        force_phase=True,
        spoof_nonessential=True,
    )


def ssins_flagging_Apr17():

    data_dir = "/safepool/rbyrne/lwa_data"
    data_output_dir = "/safepool/rbyrne/lwa_data/lwa_testing_20230415"
    ssins_flags_dir = "/safepool/rbyrne/lwa_data/lwa_testing_20230415"
    ssins_plot_dir = "/safepool/rbyrne/lwa_data/lwa_testing_20230415"
    flag_ants_file1 = "/safepool/rbyrne/lwa_data/lwa_testing_20230415/zero_ant.csv"
    flag_ants_file2 = (
        "/safepool/rbyrne/lwa_data/lwa_testing_20230415/misbehaving_ant.csv"
    )

    # Find raw ms files
    ms_filenames = [
        "20230415_050552_70MHz.ms",
        "20230415_050602_70MHz.ms",
        "20230415_050612_70MHz.ms",
        "20230415_050622_70MHz.ms",
        "20230415_050632_70MHz.ms",
        "20230415_050642_70MHz.ms",
        "20230415_050653_70MHz.ms",
        "20230415_050703_70MHz.ms",
        "20230415_050713_70MHz.ms",
        "20230415_050723_70MHz.ms",
        "20230415_050733_70MHz.ms",
        "20230415_050743_70MHz.ms",
    ]
    uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(
        [f"{data_dir}/{filename}" for filename in ms_filenames]
    )
    uvd.phase_to_time(np.mean(uvd.time_array))

    flag_ants1 = np.loadtxt(flag_ants_file1, delimiter=",", dtype=str)[:, 0]
    flag_ants2 = np.loadtxt(flag_ants_file2, delimiter=",", dtype=str)[:, 0]
    flag_ants = np.concatenate((flag_ants1, flag_ants2))
    flag_ants = np.array(
        [antname[:-1].replace("-", "") for antname in flag_ants]
    )  # Strip pol name
    LWA_preprocessing.flag_antennas(
        uvd,
        antenna_names=flag_ants,
        inplace=True,
    )

    ssins_thresh = 10.0
    uvd_ssins_flagged = LWA_preprocessing.ssins_flagging(
        uvd,
        sig_thresh=ssins_thresh,  # Flagging threshold in std. dev.
        inplace=False,
        save_flags_filepath=f"{ssins_flags_dir}/flags_thresh_{ssins_thresh}.hdf5",
        plot_no_flags=True,
        plot_orig_flags=True,
        plot_ssins_flags=True,
        plot_save_dir=ssins_plot_dir,
        plot_file_prefix=f"20230415_050552_70MHz",
    )
    flagging_frac = (np.sum(uvd_ssins_flagged.flag_array) - np.sum(uvd.flag_array)) / (
        np.size(uvd.flag_array) - np.sum(uvd.flag_array)
    )
    print(f"Flagging fraction {flagging_frac} at SSINS threshold {ssins_thresh}")

    uvd_ssins_flagged.write_uvfits(
        f"{data_output_dir}/20230415_050552_70MHz.uvfits",
        force_phase=True,
        spoof_nonessential=True,
    )


def flag_24hr_run_Jun13():

    data_dir = "/mnt/24-hour-run/73MHz"
    data_output_dir = "/data10/rbyrne/24-hour-run-flagged"
    ssins_plot_dir = "/data10/rbyrne/24-hour-run-ssins-plots"
    ssins_thresh = 15.0

    # Find raw ms files
    ms_filenames = np.sort(os.listdir(data_dir))

    start_file_ind = 0
    files_per_chunk = 30  # Process in 5 min chunks
    while start_file_ind < len(ms_filenames):
        uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(
            [
                f"{data_dir}/{filename}"
                for filename in ms_filenames[
                    start_file_ind : start_file_ind + files_per_chunk
                ]
            ]
        )
        uvd.phase_to_time(np.mean(uvd.time_array))

        # Antenna-based flagging
        # fmt: off
        flag_ants = [
            38,  51,  82,  87,  94, 105, 107, 111, 118, 122, 145, 148, 151,
            167, 173, 176, 179, 190, 197, 215, 219, 230, 231, 232, 246, 256,
            257, 258, 259, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
            271, 272, 273, 275, 276, 277, 279, 280, 281, 282, 283, 284, 285,
            287, 288, 290, 291, 293, 294, 295, 297, 298, 300, 301, 302, 303,
            304, 305, 306, 307, 308, 309, 311, 313, 316, 317, 318, 319, 320,
            322, 323, 326, 329, 330, 331, 334, 335, 336, 337, 338, 339, 340,
            341, 342, 343, 348, 349, 350, 351, 353, 359, 361, 363, 364, 365,
            366
        ]  # Flags from Nivedita
        # fmt: on
        flag_ants = [f"LWA{str(ant).zfill(3)}" for ant in flag_ants]
        LWA_preprocessing.flag_antennas(
            uvd,
            antenna_names=flag_ants,
            inplace=True,
        )

        LWA_preprocessing.ssins_flagging(
            uvd,
            sig_thresh=ssins_thresh,  # Flagging threshold in std dev
            inplace=True,
            save_flags_filepath=None,
            plot_no_flags=False,
            plot_orig_flags=True,
            plot_ssins_flags=True,
            plot_save_dir=ssins_plot_dir,
            plot_file_prefix=ms_filenames[start_file_ind].replace(".ms", ""),
        )
        # Apply flags by zeroing out flagged data
        uvd.data_array[np.where(uvd.flag_array)] = 0.0

        # Save each time step individually
        unique_times = np.unique(uvd.time_array)
        for time_ind in range(len(unique_times)):
            uvd_timestep = uvd.select(times=unique_times[time_ind], inplace=False)
            uvd_timestep.phase_to_time(np.mean(uvd_timestep.time_array))
            uvd_timestep.write_ms(
                f"{data_output_dir}/{ms_filenames[start_file_ind+time_ind]}"
            )

        start_file_ind += files_per_chunk


def plot_autocorrelations_Aug2():

    files = os.listdir("/data03/rbyrne")
    start_time_stamp = "091000"
    end_time_stamp = "091020"
    files = np.array(
        [
            f"/data03/rbyrne/{file}"
            for file in files
            if (
                file.startswith("20230801")
                and (int(file.split("_")[1]) >= int(start_time_stamp))
                and (int(file.split("_")[1]) <= int(end_time_stamp))
            )
        ]
    )
    uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(files)
    LWA_preprocessing.plot_autocorrelations(
        uvd,
        plot_save_dir="/data03/rbyrne/autocorr_plots",
        plot_file_prefix="20230801",
        time_average=True,
        plot_flagged_data=True,
        yrange=[0, 100],
    )


def flag_data_Aug3():

    use_dates = ["20230801", "20230802", "20230803", "20230804"]
    use_dates = use_dates[1:]  # First date already processed
    for date_stamp in use_dates:
        files = os.listdir("/data03/rbyrne")
        freq_stamp = "73MHz"
        start_time_stamp = "091100"
        end_time_stamp = "091600"
        files = [
            file
            for file in files
            if file.startswith(date_stamp) and file.endswith(".ms")
        ]
        files = np.array(
            [
                f"/data03/rbyrne/{file}"
                for file in files
                if (
                    (int(file.split("_")[1]) >= int(start_time_stamp))
                    and (int(file.split("_")[1]) <= int(end_time_stamp))
                    and (freq_stamp in file)
                )
            ]
        )
        uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(files)
        # fmt: off
        offline_ants = [
            39, 83, 88, 106, 119, 146, 152, 168, 174, 216, 220, 231, 365
        ]
        # fmt: on
        offline_ants = [f"LWA{str(ant).zfill(3)}" for ant in offline_ants]
        LWA_preprocessing.flag_antennas(
            uvd,
            antenna_names=offline_ants,
            inplace=True,
        )
        # fmt: off
        flag_ants = [
            "095A", "191A", "207A", "255A", "267A", "280B", "288B", "292B", "302A",
            "302B", "310A", "314B", "325B", "352A", "355B",
        ]  # Identified based on the Antenna Health Tracker spreadsheet
        # fmt: on
        flag_x = [
            f"LWA{antname[:-1]}" for antname in flag_ants if antname.endswith("A")
        ]
        flag_y = [
            f"LWA{antname[:-1]}" for antname in flag_ants if antname.endswith("B")
        ]
        LWA_preprocessing.flag_antennas(
            uvd,
            antenna_names=flag_x,
            flag_pol="X",
            inplace=True,
        )
        LWA_preprocessing.flag_antennas(
            uvd,
            antenna_names=flag_y,
            flag_pol="Y",
            inplace=True,
        )

        # Flag with SSINS
        n_unflagged_bls_start = np.size(uvd.flag_array) - np.sum(uvd.flag_array)
        LWA_preprocessing.ssins_flagging(
            uvd,
            sig_thresh=15.0,  # Flagging threshold in std dev
            inplace=True,
            save_flags_filepath=None,
            plot_no_flags=False,
            plot_orig_flags=True,
            plot_ssins_flags=True,
            plot_save_dir="/data03/rbyrne/ssins_plots",
            plot_file_prefix=f"{date_stamp}_{start_time_stamp}-{end_time_stamp}_{freq_stamp}",
        )
        n_unflagged_bls_end = np.size(uvd.flag_array) - np.sum(uvd.flag_array)
        print(
            f"SSINS flagging fraction: {float(n_unflagged_bls_end)/float(n_unflagged_bls_start) * 100.0}%"
        )

        # Save flagged data
        uvd.phase_to_time(np.mean(uvd.time_array))
        uvd.write_uvfits(
            f"/data03/rbyrne/{date_stamp}_{start_time_stamp}-{end_time_stamp}_{freq_stamp}.uvfits"
        )


def debug_flag_24hr_run_Nov2():

    data_dir = "/mnt/24-hour-run/73MHz"
    data_output_dir = "/data10/rbyrne/debug_flagging"
    ssins_thresh = 15.0

    # Find raw ms files
    ms_filenames = np.sort(os.listdir(data_dir))

    start_file_ind = 0
    files_per_chunk = 6

    uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(
        [
            f"{data_dir}/{filename}"
            for filename in ms_filenames[
                start_file_ind : start_file_ind + files_per_chunk
            ]
        ]
    )
    uvd.phase_to_time(np.mean(uvd.time_array))

    # Antenna-based flagging
    # fmt: off
    flag_ants = [
        38,  51,  82,  87,  94, 105, 107, 111, 118, 122, 145, 148, 151,
        167, 173, 176, 179, 190, 197, 215, 219, 230, 231, 232, 246, 256,
        257, 258, 259, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
        271, 272, 273, 275, 276, 277, 279, 280, 281, 282, 283, 284, 285,
        287, 288, 290, 291, 293, 294, 295, 297, 298, 300, 301, 302, 303,
        304, 305, 306, 307, 308, 309, 311, 313, 316, 317, 318, 319, 320,
        322, 323, 326, 329, 330, 331, 334, 335, 336, 337, 338, 339, 340,
        341, 342, 343, 348, 349, 350, 351, 353, 359, 361, 363, 364, 365,
        366
    ]  # Flags from Nivedita
    # fmt: on
    flag_ants = [f"LWA{str(ant).zfill(3)}" for ant in flag_ants]
    LWA_preprocessing.flag_antennas(
        uvd,
        antenna_names=flag_ants,
        inplace=True,
    )

    LWA_preprocessing.ssins_flagging(
        uvd,
        sig_thresh=ssins_thresh,  # Flagging threshold in std dev
        inplace=True,
        save_flags_filepath=None,
        plot_no_flags=False,
        plot_orig_flags=False,
        plot_ssins_flags=False,
    )
    # Apply flags by zeroing out flagged data
    uvd.data_array[np.where(uvd.flag_array)] = 0.0

    # Save each time step individually
    unique_times = np.unique(uvd.time_array)
    for time_ind in range(len(unique_times)):
        uvd_timestep = uvd.select(times=unique_times[time_ind], inplace=False)
        uvd_timestep.phase_to_time(np.mean(uvd_timestep.time_array))
        uvd_timestep.write_ms(
            f"{data_output_dir}/{ms_filenames[start_file_ind+time_ind]}"
        )

    start_file_ind += files_per_chunk


def test_ssins_Nov20():

    date_stamp = "20230819"
    files = os.listdir("/data09/xhall/2023-08-19_24hour_run")
    freq_stamp = "73MHz"
    start_time_stamp = "092823"
    end_time_stamp = "093223"
    files = [
        file for file in files if file.startswith(date_stamp) and file.endswith(".ms")
    ]
    files = np.sort(
        np.array(
            [
                f"/data09/xhall/2023-08-19_24hour_run/{file}"
                for file in files
                if (
                    (int(file.split("_")[1]) >= int(start_time_stamp))
                    and (int(file.split("_")[1]) <= int(end_time_stamp))
                    and (freq_stamp in file)
                )
            ]
        )
    )
    uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(files)
    # offline_corr_nums = [79,150,201,224,229,215,221,242,246,272,294,299,332,334,33,34,37,38,41,42,44,92,51,21,190,154,56,29,28,222,126,127]
    # offline_ants calculated with mapping.correlator_to_antname() on the development branch
    offline_ants = [
        "LWA041",
        "LWA095",
        "LWA111",
        "LWA124",
        "LWA128",
        "LWA150",
        "LWA178",
        "LWA187",
        "LWA191",
        "LWA195",
        "LWA204",
        "LWA207",
        "LWA218",
        "LWA221",
        "LWA255",
        "LWA260",
        "LWA263",
        "LWA272",
        "LWA280",
        "LWA288",
        "LWA292",
        "LWA302",
        "LWA303",
        "LWA314",
        "LWA319",
        "LWA325",
        "LWA336",
        "LWA341",
        "LWA352",
        "LWA355",
        "LWA365",
        "LWA364",
    ]
    LWA_preprocessing.flag_antennas(
        uvd,
        antenna_names=offline_ants,
        inplace=True,
    )

    # Flag with SSINS
    n_unflagged_bls_start = np.size(uvd.flag_array) - np.sum(uvd.flag_array)
    LWA_preprocessing.ssins_flagging(
        uvd,
        sig_thresh=15.0,  # Flagging threshold in std dev
        inplace=True,
        save_flags_filepath=None,
        plot_no_flags=False,
        plot_orig_flags=True,
        plot_ssins_flags=True,
        plot_save_dir="/data09/rbyrne/ssins_testing_Nov2023",
        plot_file_prefix=f"{date_stamp}_{start_time_stamp}-{end_time_stamp}_{freq_stamp}",
    )
    n_unflagged_bls_end = np.size(uvd.flag_array) - np.sum(uvd.flag_array)
    print(
        f"SSINS flagging fraction: {(1 - float(n_unflagged_bls_end)/float(n_unflagged_bls_start)) * 100.0}%"
    )

    # Save each time step individually
    unique_times = np.unique(uvd.time_array)
    for time_ind in range(len(unique_times)):
        save_file_name = (files[time_ind].split("/")[-1]).split(".")[0]
        save_file_name = f"{save_file_name}_ssins_flagged.ms"
        uvd_timestep = uvd.select(times=unique_times[time_ind], inplace=False)
        uvd_timestep.phase_to_time(np.mean(uvd_timestep.time_array))
        uvd_timestep.reorder_pols(order="CASA", run_check=False)
        uvd_timestep.write_ms(
            f"/data09/rbyrne/ssins_testing_Nov2023/{save_file_name}",
            run_check=False,
        )


def test_ssins_Dec21():

    date_stamp = "20230819"
    files = os.listdir("/data09/xhall/2023-08-19_24hour_run")
    freq_stamp = "73MHz"
    start_time_stamp = "093003"
    end_time_stamp = "093043"
    files = [
        file for file in files if file.startswith(date_stamp) and file.endswith(".ms")
    ]
    files = np.sort(
        np.array(
            [
                f"/data09/xhall/2023-08-19_24hour_run/{file}"
                for file in files
                if (
                    (int(file.split("_")[1]) >= int(start_time_stamp))
                    and (int(file.split("_")[1]) <= int(end_time_stamp))
                    and (freq_stamp in file)
                )
            ]
        )
    )
    uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(files)
    # offline_corr_nums = [79,150,201,224,229,215,221,242,246,272,294,299,332,334,33,34,37,38,41,42,44,92,51,21,190,154,56,29,28,222,126,127]
    # offline_ants calculated with mapping.correlator_to_antname() on the development branch
    offline_ants = [
        "LWA041",
        "LWA095",
        "LWA111",
        "LWA124",
        "LWA128",
        "LWA150",
        "LWA178",
        "LWA187",
        "LWA191",
        "LWA195",
        "LWA204",
        "LWA207",
        "LWA218",
        "LWA221",
        "LWA255",
        "LWA260",
        "LWA263",
        "LWA272",
        "LWA280",
        "LWA288",
        "LWA292",
        "LWA302",
        "LWA303",
        "LWA314",
        "LWA319",
        "LWA325",
        "LWA336",
        "LWA341",
        "LWA352",
        "LWA355",
        "LWA365",
        "LWA364",
    ]
    # LWA_preprocessing.flag_antennas(
    #    uvd,
    #    antenna_names=offline_ants,
    #    inplace=True,
    # )

    # Flag with SSINS
    n_unflagged_bls_start = np.size(uvd.flag_array) - np.sum(uvd.flag_array)
    LWA_preprocessing.ssins_flagging(
        uvd,
        sig_thresh=15.0,  # Flagging threshold in std dev
        inplace=True,
        save_flags_filepath=None,
        plot_no_flags=False,
        plot_orig_flags=True,
        plot_ssins_flags=True,
        plot_save_dir="/data09/rbyrne/ssins_testing_Dec2023",
        plot_file_prefix=f"{date_stamp}_{start_time_stamp}-{end_time_stamp}_{freq_stamp}",
    )
    n_unflagged_bls_end = np.size(uvd.flag_array) - np.sum(uvd.flag_array)
    print(
        f"SSINS flagging fraction: {(1 - float(n_unflagged_bls_end)/float(n_unflagged_bls_start)) * 100.0}%"
    )

    # Remove flags
    uvd.flag_array[:, :, :, :] = False

    # Save each time step individually
    unique_times = np.unique(uvd.time_array)
    for time_ind in range(len(unique_times)):
        save_file_name = (files[time_ind].split("/")[-1]).split(".")[0]
        save_file_name = f"{save_file_name}_ssins_no_flagging.ms"
        uvd_timestep = uvd.select(times=unique_times[time_ind], inplace=False)
        uvd_timestep.phase_to_time(np.mean(uvd_timestep.time_array))
        uvd_timestep.reorder_pols(order="CASA", run_check=False)
        uvd_timestep.write_ms(
            f"/data09/rbyrne/ssins_testing_Dec2023/{save_file_name}",
            flip_conj=True,
            run_check=False,
        )


if __name__ == "__main__":
    test_ssins_Dec21()
