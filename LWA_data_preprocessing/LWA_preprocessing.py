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


def get_pol_names(polarization_array):

    pol_names = np.array_like(polarization_array, dtype=object)
    for pol_ind, pol in polarization_array:
        # Instrumental polarizations:
        if pol == -5:
            pol_names[pol_ind] = "XX"
        elif pol == -6:
            pol_names[pol_ind] = "YY"
        elif pol == -7:
            pol_names[pol_ind] = "XY"
        elif pol == -8:
            pol_names[pol_ind] = "YX"
        # Pseudo-Stokes polarizations:
        elif pol == 1:
            pol_names[pol_ind] = "pI"
        elif pol == 2:
            pol_names[pol_ind] = "pQ"
        elif pol == 3:
            pol_names[pol_ind] = "pU"
        elif pol == 4:
            pol_names[pol_ind] = "pV"
        # Circular polarizations:
        elif pol == -1:
            pol_names[pol_ind] = "RR"
        elif pol == -2:
            pol_names[pol_ind] = "LL"
        elif pol == -3:
            pol_names[pol_ind] = "RL"
        elif pol == -4:
            pol_names[pol_ind] = "LR"
        else:
            print(f"WARNING: Unknown polarization mode {pol}.")
            pol_names[pol_ind] = str(pol)

    return pol_names


def plot_autocorrelations(
    uvd, plot_save_path="", plot_file_prefix="", time_average=True, plot_legend=False
):

    if time_average:
        n_time_plots = 1
    else:
        n_time_plots = uvd.Ntimes
        times = np.unique(uv.time_array)

    for time_plot_ind in range(n_time_plots):

        if time_average:
            uvd_autos = uvd.select(ant_str="auto", inplace=False)
        else:
            uvd_autos = uvd.select(
                ant_str="auto", times=times[time_plot_ind], inplace=False
            )

        pol_names = get_pol_names(uvd_autos.polarization_array)
        for pol_ind in range(uvd_autos.Npols):

            if time_average:
                plot_name = f"{plot_file_prefix}_autocorr_{pol_names[pol_ind]}.png"
            else:
                plot_name = f"{plot_file_prefix}_autocorr_{pol_names[pol_ind]}_time{time_plot_ind:05d}.png"

            for ant_ind in range(uvd_autos.Nants_data):
                ant_name = uvd_autos.antenna_names[uvd_autos.ant_1_array[ant_ind]]
                bl_inds = np.where(uvd.ant_1_array == ant_ind)
                plt.plot(
                    uvd_autos.freq_array[0, :],
                    np.mean(
                        np.real(uvd_autos.data_array[bl_inds, 0, :, pol_ind]), axis=0
                    ),
                    "-o",
                    markersize=2,
                    linewidth=1,
                    label=ant_name,
                )

            if plot_legend:
                plt.legend(prop={"size": 4})
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Autocorr. Power")
            plt.savefig(f"{plot_save_path}/{plot_name}")
            plt.close()


def remove_inactive_antennas(uvd, autocorr_thresh=1.0, inplace=False):
    # Remove unused antennas based on low autocorrelation values

    used_antennas = []
    for ant_ind in range(uvd.Nants_data):
        bl_inds = np.intersect1d(
            np.where(uvd.ant_1_array == ant_ind), np.where(uvd.ant_2_array == ant_ind)
        )
        avg_autocorr = np.mean(np.abs(uvd.data_array[bl_inds, :, :, :]))
        if avg_autocorr > autocorr_thresh:
            used_antennas.append(uvd.antenna_names[ant_ind])

    if inplace:
        uvd.select(antenna_names=used_antennas, inplace=True)
    else:
        uvd_new = uvd.select(antenna_names=used_antennas, inplace=False)
        return uvd_new


def plot_ssins(incoherent_noise_spec, plot_save_filename, Npols, pol_names):

    xticks = np.arange(0, len(incoherent_noise_spec.freq_array), 50)
    xticklabels = [
        "%.1f" % (incoherent_noise_spec.freq_array[tick] * 10 ** (-6))
        for tick in xticks
    ]

    fig, ax = plt.subplots(nrows=2 * Npols, figsize=(16, 4.0 * Npols + 1))
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
    fig.savefig(plot_save_filename, dpi=200)


def ssins_flagging(
    uvd,
    sig_thresh=5,  # Flagging threshold in std. dev.
    inplace=False,
    plot=False,
    plot_save_filename=None,
):

    ss = uvd.copy()
    ss.MLE = None
    ss.__class__ = SSINS.sky_subtract.SS
    ss.diff()

    ss.apply_flags(flag_choice="original")

    # Create Incoherent Noise Spectrum and flag with a match filter
    incoherent_noise_spec = SSINS.INS(ss)
    match_filter = SSINS.MF(
        incoherent_noise_spec.freq_array,
        sig_thresh,
        streak=False,
        narrow=True,
        shape_dict={},
    )
    match_filter.apply_match_test(incoherent_noise_spec)

    # Plot
    if plot and plot_save_filename is None:
        print("WARNING: plot_save_filename not supplied. Skipping plotting.")
        plot = False
    if plot:
        pol_names = get_pol_names(uvd.polarization_array)
        plot_ssins(incoherent_noise_spec, plot_save_filename, uvd.Npols, pol_names)

    # Apply flags and save
    uvf = pyuvdata.UVFlag(uvd, waterfall=True, mode="flag")
    incoherent_noise_spec.flag_uvf(uvf, inplace=True)
    if inplace:
        pyuvdata.utils.apply_uvflag(uvd, uvf, inplace=True)
    else:
        uvd_flagged = pyuvdata.utils.apply_uvflag(uvd, uvf, inplace=False)
        return uvd_flagged
