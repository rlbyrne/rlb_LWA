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
        ssins_plot_name = f"{file_split}_ssins_plot.png"
        autos_all_ants_plot_prefix = f"{file_split}_all_ants"
        autos_plot_prefix = f"{file_split}"

        uvd = pyuvdata.UVData()
        uvd.read(f"{data_path}/{file}")

        LWA_preprocessing.plot_autocorrelations(
            uvd,
            plot_save_path=autos_plot_save_path,
            plot_file_prefix=autos_all_ants_plot_prefix,
            time_average=True,
            plot_legend=False
        )

        LWA_preprocessing.remove_inactive_antennas(uvd, autocorr_thresh=1.0, inplace=True)

        LWA_preprocessing.plot_autocorrelations(
            uvd,
            plot_save_path=autos_plot_save_path,
            plot_file_prefix=autos_plot_prefix,
            time_average=True,
            plot_legend=False
        )

        LWA_preprocessing.ssins_flagging(
            uvd,
            sig_thresh=5,  # Flagging threshold in std. dev.
            inplace=True,
            plot=True,
            plot_save_filename=f"{ssins_plot_save_path}/{ssins_plot_name}",
        )
        uvd.write_uvfits(f"{output_path}/{file}")


if __name__ == "__main__":
    ssins_flagging_Mar25()
