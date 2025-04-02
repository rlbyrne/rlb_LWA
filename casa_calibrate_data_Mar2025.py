# Run in CASA with command execfile("casa_calibrate_data_Mar2025.py")

freq_bands = [
    "41",
    "46",
    "50",
    "55",
    "59",
    "64",
    "69",
    "73",
    "78",
    "82",
]
filenames = [f"/lustre/rbyrne/2024-03-03/20240303_093000-093151_{freq_band}MHz.ms" for freq_band in freq_bands]
cl_files = [f"/lustre/rbyrne/2024-03-03/20240303_093000-093151_{freq_band}MHz.cl" for freq_band in freq_bands]
bcal_files = [f"/lustre/rbyrne/2024-03-03/20240303_093000-093151_{freq_band}MHz.bcal" for freq_band in freq_bands]
for file_ind in range(len(freq_bands)):
    ft(filenames[file_ind], complist=cl_files[file_ind], usescratch=True)
    bandpass(filenames[file_ind], bcal_files[file_ind], uvrange="10~125lambda", fillgaps=1)
    applycal(filenames[file_ind], gaintable=bcal_files[file_ind], flagbackup=False)