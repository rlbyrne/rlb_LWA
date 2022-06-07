import pyuvdata
import numpy as np
import matplotlib.pyplot as plt


cal = pyuvdata.UVCal()
cal.read_fhd_cal(
    "/lustre/rbyrne/fhd_outputs/fhd_rlb_LWA_caltest_mmode_with_cyg_cas_Apr2022/calibration/20220210_191447_70MHz_ssins_thresh_20_cal.sav",
    "/lustre/rbyrne/fhd_outputs/fhd_rlb_LWA_caltest_mmode_with_cyg_cas_Apr2022/metadata/20220210_191447_70MHz_ssins_thresh_20_obs.sav"
)
for ant_ind in range(np.shape(cal.gain_array)[0]):
    plt.plot(cal.freq_array[0, :]/1e6, np.abs(cal.gain_array[ant_ind, 0, :, 0, 0]))
plt.xlim([np.min(cal.freq_array[0, :]/1e6), np.max(cal.freq_array[0, :]/1e6)])
plt.xlabel("Frequency (MHz)")
plt.ylabel("Gain Amplitude")
plt.savefig("/lustre/rbyrne/fhd_outputs/fhd_rlb_LWA_caltest_mmode_with_cyg_cas_Apr2022/20220210_191447_70MHz_ssins_thresh_20_cal_amp.png", dpi=600)
plt.close()

for ant_ind in range(np.shape(cal.gain_array)[0]):
    plt.plot(cal.freq_array[0, :]/1e6, np.angle(cal.gain_array[ant_ind, 0, :, 0, 0]))
plt.xlim([np.min(cal.freq_array[0, :]/1e6), np.max(cal.freq_array[0, :]/1e6)])
plt.xlabel("Frequency (MHz)")
plt.ylim([-np.pi, np.pi])
plt.ylabel("Gain Phase (rad.)")
plt.savefig("/lustre/rbyrne/fhd_outputs/fhd_rlb_LWA_caltest_mmode_with_cyg_cas_Apr2022/20220210_191447_70MHz_ssins_thresh_20_cal_phase.png", dpi=600)
plt.close()
