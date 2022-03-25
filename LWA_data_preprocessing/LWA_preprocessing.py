import pyuvdata
import os
import numpy as np
import subprocess
import shlex
import matplotlib.pyplot as plt
from matplotlib import cm
import SSINS
from SSINS import plot_lib


def convert_raw_ms_to_uvdata():

    data_path = "/lustre/rbyrne/LWA_data_20220210"
    freq = "70MHz"
    start_time_stamp = 191447
    end_time_stamp = 194824
    nfiles_per_uvfits = 12

    filenames = os.listdir(data_path)
    use_files = []
    for file in filenames:
        file_split = file.split("_")
        if file_split[2] == "{}.ms.tar".format(freq):
            if (
                int(file_split[1]) >= start_time_stamp
                and int(file_split[1]) <= end_time_stamp
            ):
                use_files.append(file)

    for file_ind, file in enumerate(use_files):
        subprocess.call(shlex.split(f"tar -xvf {data_path}/{file} -C {data_path}"))
        file_split = file.split(".")
        uv_new = pyuvdata.UVData()
        uv_new.read_ms(f"{data_path}/{file_split[0]}.ms")
        subprocess.call(shlex.split(f"rm -r {data_path}/{file_split[0]}.ms"))
        uv_new.unphase_to_drift()
        if file_ind % nfiles_per_uvfits == 0:
            uv = uv_new
            time_stamp = (file.split("_")[1]).split(".")[0]
            outfile_name = f"/{data_path}/20220210_{freq}_{time_stamp}_combined.uvfits"
        else:
            uv = uv + uv_new
        if (file_ind + 1) % nfiles_per_uvfits == 0:
            uv.instrument = "OVRO-LWA"
            uv.telescope_name = "OVRO-LWA"
            uv.set_telescope_params()
            print("Saving file to {}".format(outfile_name))
            uv.write_uvfits(outfile_name, force_phase=True, spoof_nonessential=True)


def remove_zeroed_antennas():
    # Script to remove "data" from antennas that are not plugged in, i.e. visibilities are zeroed

    data_path = "/lustre/rbyrne/LWA_data_20220210"
    save_path = f"{data_path}/uvfits_antenna_selected"

    filenames = os.listdir(data_path)
    use_files = []
    use_files = [
        filename
        for filename in filenames
        if len(filename.split(".")) > 1 and filename.split(".")[1] == "uvfits"
    ]

    Nants = 352
    autocorr_bls = [(ant_ind, ant_ind) for ant_ind in range(Nants)]
    for file_ind, uvfits_file in enumerate(use_files):
        uv = pyuvdata.UVData()
        uv.read_uvfits(f"{data_path}/{uvfits_file}", bls=autocorr_bls)

        unused_ants = []
        used_ants = []
        for ant_ind in range(Nants):
            ant_name = uv.antenna_names[uv.ant_1_array[ant_ind]]
            avg_autocorr = np.mean(np.abs(uv.data_array[ant_ind, 0, :, :]))
            if avg_autocorr < 1.0:
                unused_ants.append(ant_name)
            else:
                used_ants.append(ant_name)

        print(f"Used antennas: {used_ants}")
        print(f"Unused antennas: {unused_ants}")

        uv.write_uvfits(f"{save_path}/{uvfits_file}")


def plot_autocorrelations():

    data_path = "/lustre/rbyrne/LWA_data_20220210"
    save_path = f"{data_path}/autocorrelation_plots"

    filenames = os.listdir(data_path)
    use_files = []
    use_files = [
        filename
        for filename in filenames
        if len(filename.split(".")) > 1 and filename.split(".")[1] == "uvfits"
    ]

    Nants = 352
    autocorr_bls = [(ant_ind, ant_ind) for ant_ind in range(Nants)]
    for uvfits_file in use_files:
        uv = pyuvdata.UVData()
        uv.read_uvfits(
            f"{data_path}/{uvfits_file}", bls=autocorr_bls, polarizations=[-5, -6]
        )
        times = list(set(uv.time_array))
        for time_ind, time_val in enumerate(times):
            uv_time_selected = uv.select(times=time_val, inplace=False)
            for pol_ind, pol_name in enumerate(["xx", "yy"]):
                for ant_ind in range(Nants):
                    ant_name = uv_time_selected.antenna_names[
                        uv_time_selected.ant_1_array[ant_ind]
                    ]
                    plt.plot(
                        uv_time_selected.freq_array[0, :],
                        np.real(uv_time_selected.data_array[ant_ind, 0, :, pol_ind]),
                        "-o",
                        markersize=2,
                        linewidth=1,
                        label=ant_name,
                    )
                # plt.legend(prop={"size": 4})
                plt.xlabel("Frequency (MHz)")
                plt.ylabel("Autocorr. Power")
                plt.savefig(
                    f"{save_path}/LWA_data_20220210_autocorr_{pol_name}_{time_ind:05d}"
                )
                plt.close()


def ssins_flagging(
    uvd,
    sig_thresh=5,  # Flagging threshold in std. dev.
    inplace=False,
    plot=False,
    plot_save_path=None,
):

    ss = uvd.copy()
    ss.MLE = None
    ss.__class__ = SSINS.sky_subtract.SS
    ss.diff()

    ss.apply_flags(flag_choice="original")

    # Create Incoherent Noise Spectrum
    incoherent_noise_spec = SSINS.INS(ss)
    match_filter = SSINS.MF(
        incoherent_noise_spec.freq_array,
        sig_thresh,
        streak=False,
        narrow=True,
        shape_dict={},
    )
    match_filter.apply_match_test(incoherent_noise_spec)

    if plot and plot_save_path is None:
        print("WARNING: plot_save_path not supplied. Skipping plotting.")
        plot = False
    if plot:
        Npols = uvd.Npols

        pol_names = np.empty(Npols, dtype=object)
        for pol_ind, pol in enumerate(uvd.polarization_array):
            if pol == -5:
                pol_name = "XX"
            elif pol == -6:
                pol_name = "YY"
            elif pol == -7:
                pol_name = "XY"
            elif pol == -8:
                pol_name = "YX"
            else:
                pol_name = str(pol)
            pol_names[pol_ind] = pol_name

        xticks = np.arange(0, len(incoherent_noise_spec.freq_array), 50)
        xticklabels = ["%.1f" % (incoherent_noise_spec.freq_array[tick] * 10 ** (-6)) for tick in xticks]

        fig, ax = plt.subplots(nrows=2 * Npols, figsize=(16, 4.*Npols+1))
        subfig_ind = 0
        # Plot averaged amplitudes:
        for pol_ind, pol_name in enumerate(pol_names):
            plot_lib.image_plot(
                fig,
                ax[subfig_ind],
                incoherent_noise_spec.metric_array[:, :, pol_ind],
                title=f"{pol_name} Amplitudes",
                xticks=xticks,
                xticklabels=xticklabels,
            )
            ax[subfig_ind].set_xlabel("Frequency (MHz)")
            ax[subfig_ind].set_ylabel("Time (2s)")
            subfig_ind += 1
        # Plot z-scores:
        for pol_ind, pol_name in enumerate(pol_names):
            plot_lib.image_plot(
                fig,
                ax[subfig_ind],
                incoherent_noise_spec.metric_ms[:, :, pol_ind],
                title=f"{pol_name} z-scores",
                xticks=xticks,
                xticklabels=xticklabels,
                cmap=cm.coolwarm,
                midpoint=True,
            )
            ax[subfig_ind].set_xlabel("Frequency (MHz)")
            ax[subfig_ind].set_ylabel("Time (2s)")
        plt.tight_layout()
        fig.savefig(plot_save_path, dpi=200)

    uvf = pyuvdata.UVFlag(uvd, waterfall=True, mode="flag")
    incoherent_noise_spec.flag_uvf(uvf, inplace=True)
    if inplace:
        pyuvdata.utils.apply_uvflag(uvd, uvf, inplace=True)
    else:
        uvd_flagged = pyuvdata.utils.apply_uvflag(uvd, uvf, inplace=False)
        return uvd_flagged


if __name__ == "__main__":

    data_path = (
        "/Users/ruby/Astro/SSINS/SSINS/data/1061313128_99bl_1pol_half_time.uvfits"
    )
    output_path = "/Users/ruby/Astro/SSINS_tutorial_outputs/tutorial_test_writeout.uvfits"
    plot_save_path = "/Users/ruby/Astro/SSINS_tutorial_outputs/tutorial_test.png"

    uvd = pyuvdata.UVData()
    uvd.read(data_path)
    ssins_flagging(
        uvd,
        sig_thresh=5,  # Flagging threshold in std. dev.
        inplace=True,
        plot=True,
        plot_save_path=plot_save_path
    )
    uvd.write_uvfits(output_path)
