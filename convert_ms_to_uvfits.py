import pyuvdata

freq_channels = ['02', '03', '04', '05', '06', '07']
outfile_name = '/lustre/rbyrne/2019-11-21T23:00:08.uvfits'

for chan_ind, chan in enumerate(freq_channels):
	uv_new = pyuvdata.UVData()
	uv_new.read_ms('/lustre/mmanders/exoplanet/processing/msfiles/2019-11-21/hh=23/2019-11-21T23:00:08/{}_2019-11-21T23:00:08.ms'.format(chan))
	if chan_ind == 0:
		uv = uv_new
	else:
		uv = uv + uv_new
print('Saving file to {}'.format(outfile_name))
uv.write_uvfits(outfile_name, spoof_nonessential=True)

