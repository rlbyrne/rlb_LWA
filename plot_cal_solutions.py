import pyuvdata
import numpy as np
import matplotlib.pyplot as plt

fhd_output_path = "/lustre/rbyrne/fhd_outputs/fhd_rlb_LWA_caltest_mmode_with_cyg_cas_Apr2022"
obsid = "20220210_191447_70MHz_ssins_thresh_20"

data = pyuvdata.UVData()
pol = "XX"
filelist = [
    f"{fhd_output_path}/vis_data/{obsid}_vis_XX.sav",
    f"{fhd_output_path}/vis_data/{obsid}_vis_YY.sav",
    f"{fhd_output_path}/vis_data/{obsid}_vis_model_XX.sav",
    f"{fhd_output_path}/vis_data/{obsid}_vis_model_YY.sav",
    f"{fhd_output_path}/vis_data/{obsid}_flags.sav",
    f"{fhd_output_path}/metadata/{obsid}_params.sav",
    f"{fhd_output_path}/metadata/{obsid}_settings.txt",
    f"{fhd_output_path}/metadata/{obsid}_layout.sav",
    f"{fhd_output_path}/metadata/{obsid}_obs.sav",
]
data.read_fhd(filelist)

ant_inds = np.unique(np.concatenate((data.ant_1_array, data.ant_2_array)))
ants_with_data = []
for ant_ind in ant_inds:
    bls = np.unique(np.concatenate((
        np.where(data.ant_1_array == ant_ind)[0],
        np.where(data.ant_2_array == ant_ind)[0],
    )))
    if not np.min(data.flag_array[bls, :, :, :]):
        ant_name = data.antenna_names[ant_ind]
        ants_with_data.append(ant_name)

print(f"{len(ants_with_data)} antennas are not fully flagged")
data.select(antenna_names=ants_with_data)

cal = pyuvdata.UVCal()
cal.read_fhd_cal(
    f"{fhd_output_path}/calibration/{obsid}_cal.sav",
    f"{fhd_output_path}/metadata/{obsid}_obs.sav",
    layout_file=f"{fhd_output_path}/metadata/{obsid}_layout.sav",
    settings_file=f"{fhd_output_path}/metadata/{obsid}_settings.txt",
)
cal.select(antenna_names=ants_with_data)
plot_gains = cal.gain_array[:, 0, :, 0, :]  # Shape (Nants_data, 1, Nfreqs, Ntimes, Njones)

if False:
    for ant_ind in range(cal.Nants_data):
        ant_name = cal.antenna_names[cal.ant_array[ant_ind]]
        data_ant_ind = np.where(data.antenna_names == ant_name)[0]
        bls = np.unique(np.concatenate((
            np.where(data.ant_1_array == data_ant_ind)[0],
            np.where(data.ant_2_array == data_ant_ind)[0],
        )))
        if np.size(bls) > 0:
            plot_gains_ant = plot_gains[ant_ind, :, :]
            flag_channels = np.where(
                np.min(data.flag_array[bls, 0, :, :], axis=0)
            )[0]
            if np.size(flag_channels) > 0:
                plot_gains_ant[flag_channels] = np.nan
                plot_gains[ant_ind, :, :] = plot_gains_ant
        else:
            plot_gains[ant_ind, :, :] = np.nan

for ant_ind in range(cal.Nants_data):
    print(np.mean(np.abs(plot_gains[ant_ind, :, 0])))
    plt.plot(cal.freq_array[0, :]/1e6, np.abs(plot_gains[ant_ind, :, 0]))
plt.xlim([np.min(cal.freq_array[0, :]/1e6), np.max(cal.freq_array[0, :]/1e6)])
plt.xlabel("Frequency (MHz)")
plt.ylabel("Gain Amplitude")
plt.savefig(f"{fhd_output_path}/{obsid}_cal_amp.png", dpi=600)
plt.close()

for ant_ind in range(np.shape(cal.gain_array)[0]):
    ant_name = cal.antenna_names[cal.ant_array[ant_ind]]
    plt.plot(cal.freq_array[0, :]/1e6, np.angle(plot_gains[ant_ind, :, 0]))
plt.xlim([np.min(cal.freq_array[0, :]/1e6), np.max(cal.freq_array[0, :]/1e6)])
plt.xlabel("Frequency (MHz)")
plt.ylim([-np.pi, np.pi])
plt.ylabel("Gain Phase (rad.)")
plt.savefig(f"{fhd_output_path}/{obsid}_cal_phase.png", dpi=600)
plt.close()
