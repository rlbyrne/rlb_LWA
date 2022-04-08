import pyuvdata
import os
import numpy as np
import subprocess
import shlex
import matplotlib.pyplot as plt
from matplotlib import cm
import SSINS
from SSINS import plot_lib


def convert_raw_ms_to_uvdata(
    ms_filenames,  # String or list of strings
    untar_dir=None,  # Used if files are tar-ed. None defaults to original data dir.
):

    if type(ms_filenames) == str:
        ms_filenames = [ms_filenames]

    uvd_new = pyuvdata.UVData()
    for file_ind, ms_file in enumerate(ms_filenames):
        print(f"Reading data file {ms_file}")
        uvd_new = pyuvdata.UVData()
        if ms_file.endswith(".tar"):
            ms_file_split = ms_file.split("/")
            data_dir = "/".join(ms_file_split[:-1])
            if untar_dir is None:
                untar_dir = data_dir
            filename = ms_file_split[-1]
            subprocess.call(
                shlex.split(f"tar -xvf {data_dir}/{filename} -C {untar_dir}")
            )
            untar_filename = ".".join(filename.split(".")[:-1])
            uvd_new.read_ms(f"{untar_dir}/{untar_filename}")
            subprocess.call(shlex.split(f"rm -r {untar_dir}/{untar_filename}"))
        else:
            uvd_new.read_ms(ms_file)
        uvd_new.scan_number_array = None  # Fixes a pyuvdata bug
        uvd_new.unphase_to_drift()
        if file_ind == 0:
            uvd = uvd_new
        else:
            uvd = uvd + uvd_new

    uvd.instrument = "OVRO-LWA"
    uvd.telescope_name = "OVRO-LWA"
    uvd.set_telescope_params()
    uvd.check()
    return uvd


def get_pol_names(polarization_array):

    pol_names = np.empty_like(polarization_array, dtype=object)
    for pol_ind, pol in enumerate(polarization_array):
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
    uvd,
    plot_save_path="",
    plot_file_prefix="",
    time_average=True,
    plot_legend=False,
    plot_flagged_data=True,
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

        if not plot_flagged_data:
            uvd_autos.data_array[np.where(uvd_autos.flag_array)] = np.nan

        pol_names = get_pol_names(uvd_autos.polarization_array)
        ant_inds = np.intersect1d(uvd_autos.ant_1_array, uvd_autos.ant_2_array)
        for pol_ind in range(uvd_autos.Npols):

            if time_average:
                plot_name = f"{plot_file_prefix}_autocorr_{pol_names[pol_ind]}.png"
            else:
                plot_name = f"{plot_file_prefix}_autocorr_{pol_names[pol_ind]}_time{time_plot_ind:05d}.png"

            for ant_ind in ant_inds:
                ant_name = uvd_autos.antenna_names[ant_ind]
                bl_inds = np.where(uvd_autos.ant_1_array == ant_ind)[0]
                plt.plot(
                    uvd_autos.freq_array[0, :],
                    np.nanmean(
                        np.abs(uvd_autos.data_array[bl_inds, 0, :, pol_ind]), axis=0
                    ),
                    "-o",
                    markersize=0.5,
                    linewidth=0.3,
                    label=ant_name,
                )

            if plot_legend:
                plt.legend(prop={"size": 4})
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Autocorr. Power")
            plt.title(f"{pol_names[pol_ind]} Autocorrelations")
            print(f"Saving figure to {plot_save_path}/{plot_name}")
            plt.savefig(f"{plot_save_path}/{plot_name}", dpi=200)
            plt.close()


def remove_inactive_antennas(uvd, autocorr_thresh=5.0, inplace=False, flag_only=False):
    # Remove unused antennas based on low autocorrelation values
    # If flag_only=False, antennas with low autocorrelations across all polarizations are removed
    # Antenna polarizations with low autocorrelations are flagged

    uvd_autos = uvd.select(ant_str="auto", inplace=False)
    flag_arr = uvd.flag_array

    ant_inds = np.intersect1d(uvd_autos.ant_1_array, uvd_autos.ant_2_array)
    used_antennas = []
    for ant_ind in ant_inds:
        ant_name = uvd_autos.antenna_names[ant_ind]
        bl_inds = np.where(uvd_autos.ant_1_array == ant_ind)[0]
        avg_autocorr = np.mean(
            np.abs(uvd_autos.data_array[bl_inds, :, :, :]), axis=(0, 1, 2)
        )
        for pol_ind in range(uvd_autos.Npols):
            if avg_autocorr[pol_ind] < autocorr_thresh:
                ant_ind_full_array = np.where(uvd.antenna_names == ant_name)[0]
                flag_bls = np.unique(
                    np.concatenate(
                        (
                            np.where(uvd.ant_1_array == ant_ind_full_array),
                            np.where(uvd.ant_2_array == ant_ind_full_array),
                        )
                    )
                )
                flag_arr[flag_bls, :, :, pol_ind] = True
        if np.mean(avg_autocorr) > autocorr_thresh:
            used_antennas.append(ant_name)

    print(
        f"{uvd_autos.Nants_data-len(used_antennas)}/{uvd_autos.Nants_data} antennas removed due to low autocorrelation power."
    )

    if inplace:
        uvd.flag_array = flag_arr
        uvd.select(antenna_names=used_antennas, inplace=True)
    else:
        uvd_new = uvd.copy()
        uvd_new.flag_array = flag_arr
        uvd_new.select(antenna_names=used_antennas, inplace=True)
        return uvd_new


def plot_ssins(
    incoherent_noise_spec,
    plot_save_filename,
    Npols,
    pol_names,
    vmin_avg_amp=0,
    vmax_avg_amp=0.4,
    vmin_zscore=-10,
    vmax_zscore=10,
):

    xticks = np.arange(0, len(incoherent_noise_spec.freq_array), 50)
    xticklabels = [
        "%.1f" % (incoherent_noise_spec.freq_array[tick] * 10 ** (-6))
        for tick in xticks
    ]

    fig, ax = plt.subplots(nrows=2 * Npols, figsize=(16, 4.0 * Npols + 1))
    subfig_ind = 0

    # Plot averaged amplitudes:
    for pol_ind, pol_name in enumerate(pol_names):

        cmap = cm.plasma
        cax = ax[subfig_ind].imshow(
            incoherent_noise_spec.metric_array[:, :, pol_ind],
            cmap=cmap,
            vmin=vmin_avg_amp,
            vmax=vmax_avg_amp,
            extent=[
                np.min(incoherent_noise_spec.freq_array) * 1e-6,
                np.max(incoherent_noise_spec.freq_array) * 1e-6,
                incoherent_noise_spec.Ntimes + 0.5,
                -0.5,
            ],
            aspect="auto",
            interpolation="none",
        )

        cmap.set_bad(color="white")
        cbar = fig.colorbar(cax, ax=ax[subfig_ind], extend="max")
        cbar.set_label("Amplitude", rotation=270, labelpad=15)

        ax[subfig_ind].set_title(f"{pol_name} Amplitudes")
        ax[subfig_ind].set_xlabel("Frequency (MHz)")
        ax[subfig_ind].set_ylabel("Time Step")
        subfig_ind += 1

    # Plot z-scores:
    for pol_ind, pol_name in enumerate(pol_names):

        cmap = cm.coolwarm
        cax = ax[subfig_ind].imshow(
            incoherent_noise_spec.metric_ms[:, :, pol_ind],
            cmap=cmap,
            extent=[
                np.min(incoherent_noise_spec.freq_array) * 1e-6,
                np.max(incoherent_noise_spec.freq_array) * 1e-6,
                incoherent_noise_spec.Ntimes + 0.5,
                -0.5,
            ],
            vmin=vmin_zscore,
            vmax=vmax_zscore,
            aspect="auto",
            interpolation="none",
        )

        cmap.set_bad(color="white")
        cbar = fig.colorbar(cax, ax=ax[subfig_ind], extend="both")
        cbar.set_label("Deviation (Std. Dev.)", rotation=270, labelpad=15)

        ax[subfig_ind].set_title(f"{pol_name} z-Scores")
        ax[subfig_ind].set_xlabel("Frequency (MHz)")
        ax[subfig_ind].set_ylabel("Time Step")
        subfig_ind += 1

    plt.tight_layout()
    print(f"Saving figure to {plot_save_filename}")
    fig.savefig(plot_save_filename, facecolor="white", dpi=200)
    plt.close()


def ssins_flagging(
    uvd,
    sig_thresh=5,  # Flagging threshold in std. dev.
    inplace=False,
    save_flags_filepath=None,
    plot_no_flags=False,
    plot_orig_flags=False,
    plot_ssins_flags=False,
    plot_save_path="",
    plot_file_prefix="",
):

    ss = uvd.copy()
    ss.MLE = None
    if ss.phase_type == "drift":
        ss.phase_to_time(np.mean(ss.time_array))
    ss.__class__ = SSINS.sky_subtract.SS
    ss.diff()

    pol_names = get_pol_names(uvd.polarization_array)

    if plot_no_flags:
        ss.apply_flags(flag_choice=None)  # Remove flags
        incoherent_noise_spec = SSINS.INS(ss)
        plot_save_filename = f"{plot_save_path}/{plot_file_prefix}_ins_no_flags.png"
        plot_ssins(incoherent_noise_spec, plot_save_filename, uvd.Npols, pol_names)

    ss.apply_flags(flag_choice="original")  # Restore original flags
    incoherent_noise_spec = SSINS.INS(ss)  # Create Incoherent Noise Spectrum

    if plot_orig_flags:
        plot_save_filename = f"{plot_save_path}/{plot_file_prefix}_ins_orig_flags.png"
        plot_ssins(incoherent_noise_spec, plot_save_filename, uvd.Npols, pol_names)

    # Flag with a match filter
    match_filter = SSINS.MF(
        incoherent_noise_spec.freq_array,
        sig_thresh,
        streak=False,
        narrow=True,
        shape_dict={},
    )
    match_filter.apply_match_test(incoherent_noise_spec)

    # Plot with SSINS flags
    if plot_ssins_flags:
        plot_save_filename = f"{plot_save_path}/{plot_file_prefix}_ins_ssins_flags_thresh_{sig_thresh}.png"
        plot_ssins(incoherent_noise_spec, plot_save_filename, uvd.Npols, pol_names)

    # Apply flags
    uvf = pyuvdata.UVFlag(uvd, waterfall=True, mode="flag")
    incoherent_noise_spec.flag_uvf(uvf, inplace=True)

    # Optionally save flags to an .hdf5 file
    if save_flags_filepath is not None:
        uvf.write(save_flags_filepath)

    if inplace:
        pyuvdata.utils.apply_uvflag(uvd, uvf, inplace=True)
    else:
        uvd_flagged = pyuvdata.utils.apply_uvflag(uvd, uvf, inplace=False)
        return uvd_flagged
