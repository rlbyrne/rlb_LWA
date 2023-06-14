import numpy as np
import pyuvdata
import time

beam_files = ["/data05/nmahesh/LWA_x_10to100.ffe", "/data05/nmahesh/LWA_y_10to100.ffe"]

for feed_ind, file in enumerate(beam_files):

    with open(file, "r") as f:
        data = f.readlines()
    f.close()

    start_chunk_lines = np.where(["Configuration Name:" in line for line in data])[0]

    # Debug
    # data = data[0 : start_chunk_lines[2]]
    # start_chunk_lines = start_chunk_lines[0:2]

    for chunk_ind in range(len(start_chunk_lines)):

        freq_array = np.array([], dtype=float)
        theta_array = np.array([], dtype=float)  # Zenith angle
        phi_array = np.array([], dtype=float)  # Azimuth
        etheta_array = np.array([], dtype=complex)
        ephi_array = np.array([], dtype=complex)

        if chunk_ind < len(start_chunk_lines) - 1:
            chunk_lines = np.arange(
                start_chunk_lines[chunk_ind], start_chunk_lines[chunk_ind + 1]
            )
        else:
            chunk_lines = np.arange(start_chunk_lines[chunk_ind], len(data))

        freq_line = (
            [line_num for line_num in chunk_lines if "Frequency:" in data[line_num]]
        )[0]
        freq_hz = float((data[freq_line].split())[-1])

        print(f"Processing frequency {freq_hz/1e6} MHz.")
        start_time = time.time()

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
                    float(data_line_split[etheta_real_col])
                    + 1j * float(data_line_split[etheta_imag_col]),
                )
                ephi_array = np.append(
                    ephi_array,
                    float(data_line_split[ephi_real_col])
                    + 1j * float(data_line_split[ephi_imag_col]),
                )

        # Reshape array
        print("Reformatting data...")
        theta_axis, theta_indices = np.unique(theta_array, return_index=True)
        phi_axis, phi_indices = np.unique(phi_array, return_index=True)
        freq_axis, freq_indices = np.unique(freq_array, return_index=True)
        jones = np.full(
            (2, 2, len(freq_axis), len(theta_axis), len(phi_axis)),
            np.nan,
            dtype=complex,
        )
        for phi_ind, phi in enumerate(phi_axis):
            for theta_ind, theta in enumerate(theta_axis):
                use_indices = np.intersect1d(
                    phi_indices[phi_ind], theta_indices[theta_ind]
                )
                for freq_ind, freq in enumerate(freq_axis):
                    index = np.intersect1d(use_indices, freq_indices[freq_ind])
                    if len(index) > 0:
                        jones[0, feed_ind, freq_ind, theta_ind, phi_ind] = ephi_array[
                            index[0]
                        ]
                        jones[1, feed_ind, freq_ind, theta_ind, phi_ind] = etheta_array[
                            index[0]
                        ]

        # Clear variables
        freq_array = theta_array = phi_array = etheta_array = ephi_array = None

        print("Generating beam object...")
        beam_obj = pyuvdata.UVBeam()
        beam_obj.Naxes_vec = 2
        beam_obj.Nfreqs = len(freq_axis)
        beam_obj.Nspws = 1
        beam_obj.antenna_type = "simple"
        beam_obj.bandpass_array = np.full((1, len(freq_axis)), 1.0)
        beam_obj.beam_type = "efield"
        beam_obj.data_array = np.copy(jones[:, np.newaxis, :, :, :, :])
        beam_obj.data_normalization = "physical"
        beam_obj.feed_name = ""
        beam_obj.feed_version = ""
        beam_obj.freq_array = freq_axis[np.newaxis, :]
        beam_obj.history = ""
        beam_obj.model_name = ""
        beam_obj.model_version = ""
        beam_obj.pixel_coordinate_system = "az_za"
        beam_obj.spw_array = [0]
        beam_obj.telescope_name = "LWA"
        beam_obj.Naxes1 = len(phi_axis)
        beam_obj.Naxes2 = len(theta_axis)
        beam_obj.Ncomponents_vec = 2
        beam_obj.Nfeeds = 2
        beam_obj.Npols = 2
        beam_obj.axis1_array = np.radians(phi_axis)
        beam_obj.axis2_array = np.radians(theta_axis)
        beam_obj.basis_vector_array = np.repeat(
            (
                np.repeat(
                    (np.identity(2, dtype=float))[:, :, np.newaxis],
                    len(theta_axis),
                    axis=2,
                )
            )[:, :, :, np.newaxis],
            len(phi_axis),
            axis=3,
        )
        beam_obj.feed_array = ["E", "N"]
        beam_obj.x_orientation = "east"
        # beam_obj.peak_normalize()  # Throws an "invalid value encountered in divide" error
        beam_obj.check()

        print("Writing beam object...")
        beam_obj.write_beamfits(
            f"/data05/rbyrne/LWA_10to100_{freq_hz/1e6}MHz.beamfits",
            clobber=True,
        )

        # Clear variables
        beam_obj = None
        theta_axis = theta_indices = None
        phi_axis = phi_indices = None
        freq_axis = freq_indices = None
        jones = None

        print("Done.")
        print(f"Timing: {(time.time() - start_time)/60.} minutes")
        print("")
