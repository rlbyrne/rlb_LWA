import LWA_preprocessing
import pyuvdata
import os


def ssins_flagging_Mar25():

    data_path = "/lustre/rbyrne/LWA_data_20220210/uvfits_antenna_selected"
    output_path = "/lustre/rbyrne/LWA_data_20220210/uvfits_ssins_flagged"
    plot_save_path = "/lustre/rbyrne/LWA_data_20220210/ssins_plots"

    filenames = os.listdir(data_path)
    filenames = [file for file in filenames if file.enswith(".uvfits")]

    for file in filenames:
        file_split = file.split(".")[:-1]
        plot_name = f"{file_split}_ssins_plot.png"
        uvd = pyuvdata.UVData()
        uvd.read(f"{data_path}/{file}")
        LWA_preprocessing.ssins_flagging(
            uvd,
            sig_thresh=5,  # Flagging threshold in std. dev.
            inplace=True,
            plot=True,
            plot_save_path=f"{plot_save_path}/{plot_name}"
        )
        uvd.write_uvfits(f"{output_path}/{file}")

if __name__=="__main__":
    ssins_flagging_Mar25()
