#!/usr/bin/python

import os

subband_dirs = [
	'00', '01', '02', '03', '04', '05', '06', '07', '08',
	'09', '10', '11', '12', '13', '14', '15', '16', '17',
	'18', '19', '20', '21'
]
datafile_path = '/lustre/data/2018-03-20_100hr_run'
datafiles = os.listdir(f'{datafile_path}/{subband_dirs[0]}')
datafiles.sort()
outfile_path = '/lustre/rbyrne/2018-03-20_100hr_run_ms_files'
outfile_ind = 0
combine_n_files = 1
for fileind, filename in enumerate(datafiles):
	for subband_ind, subband in enumerate(subband_dirs):
		if subband_ind==0 and fileind%combine_n_files==0:
			outfile_ind += 1
			outfile_name = f'{outfile_path}/2018-03-20_100hr_run_{filename[20:36]}.ms'
			print(f'Saving data to {outfile_name}')
			os.system(f'dada2ms-tst3 {datafile_path}/{subband}/{filename} {outfile_name} > /dev/null 2>&1')
		else:
			os.system(f'dada2ms-tst3 --append --addspw {datafile_path}/{subband}/{filename} {outfile_name} > /dev/null 2>&1')

