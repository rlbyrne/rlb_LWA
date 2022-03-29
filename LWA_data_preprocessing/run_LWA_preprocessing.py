import LWA_preprocessing
import pyuvdata
import os


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

        LWA_preprocessing.remove_inactive_antennas(
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


if __name__ == "__main__":
    ssins_flagging_Mar25()
