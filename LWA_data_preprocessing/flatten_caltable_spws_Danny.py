#! /usr/bin/env python
#casa applycal requires cal files to have matching spw definitions
#  if a data file has only a subset of the band, you cannot apply cal files 
#  generated for the full band data set
#input1: ms data file where a subband has been selected
#input2: cal file from which to extract a subband

# This function is deprecated. Use LWA_calibrate.read_caltable


import numpy as np
import sys,os 
from pyuvdata import uvcal,uvdata


visfile = sys.argv[1]
casacaltable = sys.argv[2] 
outcal = sys.argv[3]


UVD = uvdata.UVData()
print(f'reading {visfile}')
UVD.read(visfile)


UVC = uvcal.UVCal()
UVC.read(casacaltable)


print(f'the cal file frequency range: {UVC.freq_array.min()/1e6:3.1f} - {UVC.freq_array.max()/1e6:3.1f}')
print(f'the ms file frequency range: {UVD.freq_array.min()/1e6:3.1f} - {UVD.freq_array.max()/1e6:3.1f}')


gain = np.ma.masked_where(UVC.flag_array,UVC.gain_array)
gain = np.ma.sum(gain,axis=2)
freq_inds = np.digitize(UVD.freq_array,UVC.freq_array)
print(f'selecting {len(freq_inds)} calibration channels from the available {UVC.Nfreqs}')
gain = gain[:,freq_inds,:]
gain.shape = (UVC.Nants_data,len(freq_inds),1,UVC.Njones)

#problem. we cant combine flags because the time staggering means that every time is flagged at least once
flags = np.zeros(gain.shape)
flags[np.abs(gain)>.999] = 1   #in my file, gain=1 was flagged.
flags[np.abs(gain)<1e-9] = 1   #no gain no pain
print(flags.shape)
print(f'flagging  {float(flags.sum()/flags.size)*100:4.1f}% of channels')
flags.dtype = bool

UVC2 = uvcal.UVCal()
UVC2.read(casacaltable)
UVC2.select(times = UVC2.time_array[-1], frequencies = UVD.freq_array)
UVC2.flex_spw_id_array = np.zeros_like(UVC2.freq_array)
UVC2.spw_array = np.array([0])
UVC2.Nspws = 1
UVC2.gain_array = gain
UVC2.flag_array = flags

print(f'writing cal file to match data file frequency range')
print(outcal)

UVC2.write_ms_cal(outcal, clobber=True)