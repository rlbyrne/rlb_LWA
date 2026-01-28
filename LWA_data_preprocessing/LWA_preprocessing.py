import pyuvdata
import os
import numpy as np
import sys
import subprocess
import shlex
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

try:
    import SSINS
except:
    print("WARNING: SSINS import failed. Some functionality will be unavailable.")


matplotlib.use("Agg")


def convert_raw_ms_to_uvdata(
    ms_filenames,  # String or list of strings
    untar_dir=None,  # Used if files are tar-ed. None defaults to original data dir.
    data_column="DATA",  # Other options are "CORRECTED_DATA" or "MODEL_DATA"
    combine_spws=True,  # Option to combine all spectral windows for compatibility
    run_aoflagger=False,
    conjugate_data=False,
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
            if run_aoflagger:
                subprocess.call(f"aoflagger {untar_dir}/{untar_filename}")
            uvd_new.read_ms(
                f"{untar_dir}/{untar_filename}",
                data_column=data_column,
                # run_check=False,  # Required to preserve baseline conjugation
                raise_error=False,  # May be needed when multiple spw are present
            )
            subprocess.call(shlex.split(f"rm -r {untar_dir}/{untar_filename}"))
        else:
            if run_aoflagger:
                os.system(f"aoflagger {ms_file}")
            uvd_new.read_ms(
                ms_file,
                data_column=data_column,
                # run_check=False,  # Required to preserve baseline conjugation
                raise_error=False,  # May be needed when multiple spw are present
            )
        if conjugate_data:
            uvd_new.data_array = np.conj(uvd_new.data_array)
        uvd_new.scan_number_array = None  # Fixes a pyuvdata bug
        uvd_new.instrument = "OVRO-LWA"
        uvd_new.telescope_name = "OVRO-LWA"
        uvd_new.set_telescope_params()
        uvd_new.set_uvws_from_antenna_positions(
            update_vis=False
        )  # Fixes incorrect telescope location
        uvd_new.unproject_phase()
        if file_ind == 0:
            uvd = uvd_new
        else:
            uvd = uvd + uvd_new

    if combine_spws and uvd.Nspws > 1:
        uvd.Nspws = 1
        uvd.spw_array = np.array([0])
        uvd.flex_spw_id_array[:] = 0
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
    plot_save_dir="",
    plot_file_prefix="",
    time_average=True,
    plot_flagged_data=False,
    yrange=[0, 100],
):

    if time_average:
        n_time_plots = 1
    else:
        n_time_plots = uvd.Ntimes
        times = np.unique(uvd.time_array)

    for time_plot_ind in range(n_time_plots):

        if time_average:
            uvd_autos = uvd.select(ant_str="auto", inplace=False)
        else:
            uvd_autos = uvd.select(
                ant_str="auto", times=times[time_plot_ind], inplace=False
            )

        if not plot_flagged_data:
            uvd_autos.data_array[np.where(uvd_autos.flag_array)] = np.nan

        # Do not plot both XY and YX (autocorrelation amplitude is identical)
        if -7 in uvd_autos.polarization_array and -8 in uvd_autos.polarization_array:
            uvd_autos.select(
                polarizations=[pol for pol in uvd_autos.polarization_array if pol != -8]
            )

        pol_names = get_pol_names(uvd_autos.polarization_array)
        ant_nums = np.intersect1d(uvd_autos.ant_1_array, uvd_autos.ant_2_array)
        ant_names = np.array(
            [
                uvd_autos.antenna_names[
                    np.where(uvd_autos.antenna_numbers == num)[0][0]
                ]
                for num in ant_nums
            ]
        )

        for ant_ind in range(len(ant_names)):
            ant_name = ant_names[ant_ind]
            if time_average:
                plot_name = f"{plot_file_prefix}_autocorr_ant_{ant_name}.png"
            else:
                plot_name = f"{plot_file_prefix}_autocorr_ant_{ant_name}_time{time_plot_ind:05d}.png"
            bl_inds = np.where(uvd_autos.ant_1_array == ant_nums[ant_ind])[0]
            plot_values = np.nanmean(
                np.abs(uvd_autos.data_array[bl_inds, 0, :, :]), axis=0
            )
            if not np.isnan(
                np.nanmean(plot_values)
            ):  # Check if antenna is completely flagged
                for pol_ind in range(uvd_autos.Npols):
                    plt.plot(
                        uvd_autos.freq_array[0, :] / 1e6,  # Convert to MHz
                        plot_values[:, pol_ind],
                        "-o",
                        markersize=0.5,
                        linewidth=0.3,
                        label=pol_names[pol_ind],
                    )
                plt.legend(prop={"size": 4})
                plt.xlabel("Frequency (MHz)")
                plt.ylabel("Autocorr. Power")
                plt.xlim(
                    [
                        np.nanmin(uvd_autos.freq_array) / 1e6,
                        np.nanmax(uvd_autos.freq_array) / 1e6,
                    ]
                )
                plt.ylim(yrange)
                plt.title(f"Antenna {ant_name} Autocorrelations")
                print(f"Saving figure to {plot_save_dir}/{plot_name}")
                plt.savefig(f"{plot_save_dir}/{plot_name}", dpi=200)
                plt.close()


def plot_autocorrelation_waterfalls(
    uvd,
    plot_save_dir="",
    plot_file_prefix="",
    plot_flagged_data=False,
    colorbar_range=[0, 100],
):

    uvd_autos = uvd.select(ant_str="auto", inplace=False)

    # Do not plot both XY and YX (autocorrelation amplitude is identical)
    if -7 in uvd.polarization_array and -8 in uvd.polarization_array:
        use_pols = [pol for pol in uvd.polarization_array if pol != -8]
        uvd_autos.select(polarizations=use_pols, inplace=True)

    ant_inds = np.intersect1d(uvd_autos.ant_1_array, uvd_autos.ant_2_array)
    times = np.unique(uvd_autos.time_array)
    ant_names = np.unique(uvd_autos.antenna_names)
    autocorr_vals = np.full(
        (np.size(ant_inds), uvd_autos.Ntimes, uvd_autos.Nfreqs, uvd_autos.Npols), np.nan
    )

    for time_plot_ind in range(uvd.Ntimes):
        uvd_timestep = uvd_autos.select(times=times[time_plot_ind], inplace=False)
        if not plot_flagged_data:
            uvd_timestep.data_array[np.where(uvd_timestep.flag_array)] = np.nan
        ant_inds = np.intersect1d(uvd_timestep.ant_1_array, uvd_timestep.ant_2_array)
        for ant_name_ind, ant_name in enumerate(ant_names):
            ant_ind = np.where(np.array(uvd_timestep.antenna_names) == ant_name)[0]
            bl_inds = np.where(uvd_timestep.ant_1_array == ant_ind)[0]
            autocorr_vals[ant_name_ind, time_plot_ind, :, :] = np.nanmean(
                np.abs(uvd_timestep.data_array[bl_inds, 0, :, :]), axis=0
            )

    use_cmap = cm.get_cmap("inferno").copy()
    use_cmap.set_bad(color="whitesmoke")
    for ant_ind, ant_name in enumerate(ant_names):

        if not np.isnan(np.nanmean(autocorr_vals[ant_ind, :, :, :])):
            plot_name = f"{plot_file_prefix}_autocorr_waterfall_ant_{ant_name}.png"
            fig, ax = plt.subplots(nrows=1, ncols=uvd_autos.Npols, figsize=(16, 6))
            pol_names = get_pol_names(uvd_autos.polarization_array)
            for pol_ind in range(uvd_autos.Npols):
                cax = ax[pol_ind].imshow(
                    autocorr_vals[ant_ind, ::-1, :, pol_ind],
                    origin="lower",
                    interpolation="none",
                    cmap=use_cmap,
                    vmin=np.min(colorbar_range),
                    vmax=np.max(colorbar_range),
                    extent=[
                        np.nanmin(uvd_autos.freq_array) / 1e6,
                        np.nanmax(uvd_autos.freq_array) / 1e6,
                        np.nanmax(uvd_autos.time_array),
                        np.nanmin(uvd_autos.time_array),
                    ],
                    aspect="auto",
                )
                ax[pol_ind].set_xlabel("Frequency (MHz)")
                ax[pol_ind].set_ylabel("Time (JD)")
                ax[pol_ind].set_title(pol_names[pol_ind])

            cbar = fig.colorbar(cax, ax=ax.ravel().tolist(), extend="max")
            cbar.set_label("Autocorrelation Value", rotation=270, labelpad=15)
            fig.suptitle(f"Antenna {ant_name} Autocorrelations")
            print(f"Saving figure to {plot_save_dir}/{plot_name}")
            plt.savefig(f"{plot_save_dir}/{plot_name}", dpi=200, bbox_inches="tight")
            plt.close()


def flag_outriggers(
    uvd,
    core_center_coords=None,  # If None, defaults to median ant position
    core_radius_m=130.0,
    remove_outriggers=False,
    inplace=False,
):

    # Get antennas positions in ECEF
    antpos = uvd.telescope.antenna_positions + uvd.telescope.telescope_location
    # Convert to topocentric (East, North, Up or ENU) coords.
    antpos = pyuvdata.utils.ENU_from_ECEF(antpos, center_loc=uvd.telescope.location)

    if core_center_coords is None:
        core_center_coords = [np.median(antpos[:, 0]), np.median(antpos[:, 1])]
    outrigger_ant_inds = np.where(
        np.sqrt(
            (antpos[:, 0] - core_center_coords[0]) ** 2.0
            + (antpos[:, 1] - core_center_coords[1]) ** 2.0
        )
        > core_radius_m
    )[0]
    outrigger_ants = uvd.telescope.antenna_numbers[outrigger_ant_inds]

    flag_arr = np.copy(uvd.flag_array)
    for ant in outrigger_ants:
        ant_1_bls = np.where(uvd.ant_1_array == ant)[0]
        if len(ant_1_bls) > 0:
            flag_arr[ant_1_bls, :, :] = True
        ant_2_bls = np.where(uvd.ant_2_array == ant)[0]
        if len(ant_2_bls) > 0:
            flag_arr[ant_2_bls, :, :] = True

    if inplace:
        uvd.flag_array = flag_arr
        if remove_outriggers:
            keep_ants = [
                antnum
                for antnum in uvd.telescope.antenna_numbers
                if antnum not in outrigger_ants
            ]
            uvd.select(antenna_nums=keep_ants)
    else:
        uvd_new = uvd.copy()
        uvd_new.flag_array = flag_arr
        if remove_outriggers:
            keep_ants = [
                antnum
                for antnum in uvd.telescope.antenna_numbers
                if antnum not in outrigger_ants
            ]
            uvd_new.select(antenna_nums=keep_ants)
        return uvd_new


def flag_antennas(
    uvd,
    antenna_names=[],
    flag_pol="all",  # Options are "all", "X", "Y", "XX", "YY", "XY", or "YX"
    inplace=False,
):

    flag_arr = np.copy(uvd.flag_array)
    pol_names = get_pol_names(uvd.polarization_array)

    for ant_name in antenna_names:
        ant_ind = np.where(np.array(uvd.antenna_names) == ant_name)[0]
        if np.size(ant_ind) == 0:
            print(f"WARNING: Antenna {ant_name} not found in antenna_names.")
        else:
            flag_bls_1 = np.where(uvd.ant_1_array == ant_ind)[0]
            flag_bls_2 = np.where(uvd.ant_2_array == ant_ind)[0]
            flag_bls = np.unique(np.concatenate((flag_bls_1, flag_bls_2)))
            if flag_pol == "all":
                flag_arr[flag_bls, :, :] = True
            elif flag_pol in ["XX", "YY", "XY", "YX"]:
                pol_ind = np.where(pol_names == flag_pol)[0]
                flag_arr[flag_bls, :, pol_ind] = True
            elif flag_pol == "X":
                pol_ind_xx = np.where(pol_names == "XX")[0]
                if np.size(pol_ind_xx) > 0:
                    flag_arr[flag_bls, :, pol_ind_xx] = True
                pol_ind_xy = np.where(pol_names == "XY")[0]
                if np.size(pol_ind_xy) > 0:
                    flag_arr[flag_bls_1, :, pol_ind_xy] = True
                pol_ind_yx = np.where(pol_names == "YX")[0]
                if np.size(pol_ind_yx) > 0:
                    flag_arr[flag_bls_2, :, pol_ind_yx] = True
            elif flag_pol == "Y":
                pol_ind_yy = np.where(pol_names == "YY")[0]
                if np.size(pol_ind_yy) > 0:
                    flag_arr[flag_bls, :, pol_ind_yy] = True
                pol_ind_xy = np.where(pol_names == "XY")[0]
                if np.size(pol_ind_xy) > 0:
                    flag_arr[flag_bls_2, :, pol_ind_xy] = True
                pol_ind_yx = np.where(pol_names == "YX")[0]
                if np.size(pol_ind_yx) > 0:
                    flag_arr[flag_bls_1, :, pol_ind_yx] = True
            else:
                print(f"ERROR: Unknown flag_pol option {flag_pol}.")
                sys.exit(1)

    if inplace:
        uvd.flag_array = flag_arr
    else:
        uvd_new = uvd.copy()
        uvd_new.flag_array = flag_arr
        return uvd_new


def flag_inactive_antennas(uvd, autocorr_thresh=5.0, inplace=False, flag_only=True):
    # Flag unused antennas based on low autocorrelation values
    # If flag_only=False, antennas with low autocorrelations across all polarizations are removed
    # Antenna polarizations with low autocorrelations are flagged

    flag_arr = np.copy(uvd.flag_array)
    uvd_autos = uvd.select(ant_str="auto", inplace=False)
    uvd_autos.data_array[np.where(uvd_autos.flag_array)] = np.nan
    ant_inds = np.intersect1d(uvd_autos.ant_1_array, uvd_autos.ant_2_array)

    pol_names = get_pol_names(uvd_autos.polarization_array)
    auto_pol_names = [name for name in pol_names if name in ["XX", "YY"]]
    cross_pol_names = [name for name in pol_names if name in ["XY", "YX"]]

    inactive_ants = []
    for ant_ind in ant_inds:
        ant_name = uvd_autos.antenna_names[ant_ind]
        bl_inds = np.where(uvd_autos.ant_1_array == ant_ind)[0]
        avg_autocorr = np.nanmean(
            np.abs(uvd_autos.data_array[bl_inds, :, :, :]), axis=(0, 1, 2)
        )
        for pol in auto_pol_names:
            pol_ind = np.where(pol_names == pol)[0][0]
            if avg_autocorr[pol_ind] < autocorr_thresh or np.isnan(
                avg_autocorr[pol_ind]
            ):
                ant_ind_full_array = np.where(np.array(uvd.antenna_names) == ant_name)[
                    0
                ]
                flag_bls_1 = np.where(uvd.ant_1_array == ant_ind_full_array)[0]
                flag_bls_2 = np.where(uvd.ant_2_array == ant_ind_full_array)[0]
                flag_bls = np.unique(np.concatenate((flag_bls_1, flag_bls_2)))
                flag_arr[flag_bls, :, :, pol_ind] = True
                for cross_pol in cross_pol_names:  # Flag cross polarizations also
                    cross_pol_ind = np.where(pol_names == cross_pol)[0][0]
                    if (pol == "XX" and cross_pol == "XY") or (
                        pol == "YY" and cross_pol == "YX"
                    ):
                        flag_arr[flag_bls_1, :, :, cross_pol_ind] = True
                    if (pol == "XX" and cross_pol == "YX") or (
                        pol == "YY" and cross_pol == "XY"
                    ):
                        flag_arr[flag_bls_2, :, :, cross_pol_ind] = True

        pol_max_autocorr = np.nanmax(avg_autocorr)
        if pol_max_autocorr < autocorr_thresh or np.isnan(pol_max_autocorr):
            inactive_ants.append(ant_name)

    print(
        f"{len(inactive_ants)}/{uvd_autos.Nants_data} antennas removed due to low autocorrelation power."
    )

    if inplace:
        uvd.flag_array = flag_arr
        if not flag_only:
            used_antennas = np.array(
                [
                    ant_name
                    for ant_name in uvd.antenna_names
                    if ant_name not in inactive_ants
                ]
            )
            uvd.select(antenna_names=used_antennas, inplace=True)
    else:
        uvd_new = uvd.copy()
        uvd_new.flag_array = flag_arr
        if not flag_only:
            used_antennas = np.array(
                [
                    ant_name
                    for ant_name in uvd.antenna_names
                    if ant_name not in inactive_ants
                ]
            )
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
    plot_save_dir="",
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
        plot_save_filename = f"{plot_save_dir}/{plot_file_prefix}_ins_no_flags.png"
        plot_ssins(incoherent_noise_spec, plot_save_filename, uvd.Npols, pol_names)

    ss.apply_flags(flag_choice="original")  # Restore original flags
    incoherent_noise_spec = SSINS.INS(ss)  # Create Incoherent Noise Spectrum

    if plot_orig_flags:
        plot_save_filename = f"{plot_save_dir}/{plot_file_prefix}_ins_orig_flags.png"
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
        plot_save_filename = f"{plot_save_dir}/{plot_file_prefix}_ins_ssins_flags_thresh_{sig_thresh}.png"
        plot_ssins(incoherent_noise_spec, plot_save_filename, uvd.Npols, pol_names)

    # Apply flags
    uvf = pyuvdata.UVFlag(uvd, waterfall=True, mode="flag")
    incoherent_noise_spec.flag_uvf(uvf, inplace=True)

    # Optionally save flags to an .hdf5 file
    if save_flags_filepath is not None:
        if not save_flags_filepath.endswith(".hdf5"):
            save_flags_filepath = save_flags_filepath + ".hdf5"
        uvf.write(save_flags_filepath, clobber=True)

    if inplace:
        pyuvdata.utils.apply_uvflag(uvd, uvf, inplace=True)
    else:
        uvd_flagged = pyuvdata.utils.apply_uvflag(uvd, uvf, inplace=False)
        return uvd_flagged
