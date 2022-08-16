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
    ms_filenames = [
        f"{data_dir}/{filename}"
        for filename in os.listdir(data_dir)
        if filenames.endswith(".ms")
    ]
    uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(ms_filenames)
    plot_autocorrelations(
        uvd,
        plot_save_dir=autos_plot_dir,
        plot_file_prefix="20220812_000008",
        time_average=True,
        plot_legend=False,
        plot_flagged_data=False,
        yrange=[0, 100],
        plot_antennas_together=False,
        plot_antennas_individually=True,
    )


if __name__ == "__main__":
    plot_autocorrelations_Aug16()
