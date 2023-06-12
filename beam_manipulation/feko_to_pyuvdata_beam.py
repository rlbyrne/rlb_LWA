import numpy as np

with open("/data05/nmahesh/LWA_x_10to100.ffe", "r") as f:
    data = f.readlines()
f.close()

freq_array = np.array([], dtype=float)
theta_array = np.array([], dtype=float)
phi_array = np.array([], dtype=float)
etheta_array = np.array([], dtype=complex)
ephi_array = np.array([], dtype=complex)

start_chunk_lines = np.where(["Configuration Name:" in line for line in data])[0]

for chunk_ind in range(len(start_chunk_lines)):

    if chunk_ind < len(start_chunk_lines) - 1:
        chunk_lines = np.arange(
            start_chunk_lines[chunk_ind], start_chunk_lines[chunk_ind + 1]
        )
    else:
        chunk_lines = np.arange(start_chunk_lines[chunk_ind], len(freq_lines))

    freq_line = (
        [line_num for line_num in chunk_lines if "Frequency:" in data[line_num]]
    )[0]
    freq_hz = float((data[freq_line].split())[-1])

    # Parse header
    header_intro_line = (
        [
            line_num
            for line_num in chunk_lines
            if "No. of Header Lines:" in data[line_num]
        ]
    )[0]
    header_len = int((data[header_intro_line].split())[-1])
    header_line = header_intro_line + header_len
    header = np.array(data[header_line].strip("#").split())
    theta_col = np.where(header == '"Theta"')[0][0]
    phi_col = np.where(header == '"Phi"')[0][0]
    etheta_real_col = np.where(header == '"Re(Etheta)"')[0][0]
    etheta_imag_col = np.where(header == '"Im(Etheta)"')[0][0]
    ephi_real_col = np.where(header == '"Re(Ephi)"')[0][0]
    ephi_imag_col = np.where(header == '"Im(Ephi)"')[0][0]

    for data_line in data[header_line + 1 : np.max(chunk_lines)]:
        data_line_split = data_line.split()
        if len(data_line_split) == len(header):
            freq_array = np.append(freq_array, freq_hz)
            theta_array = np.append(theta_array, float(data_line_split[theta_col]))
            phi_array = np.append(phi_array, float(data_line_split[phi_col]))
            etheta_array = np.append(
                etheta_array,
                float(data_line_split[etheta_real_col]) + 1j * float(data_line_split[etheta_imag_col])
            )
            ephi_array = np.append(
                ephi_array,
                float(data_line_split[ephi_real_col]) + 1j * float(data_line_split[ephi_imag_col])
            )
