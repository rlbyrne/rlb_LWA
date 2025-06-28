import pyuvsim
import pyuvdata
import pyradiosky
import astropy
import numpy as np
import os
from astropy.units import Quantity

uv = pyuvdata.UVData()
uv.read("/lustre/cosmopipe/2025-06-13/20250613_070132_55MHz.ms")  # Choose an arbitrary data file for reference

# Set antenna positions
antpos = np.zeros_like(uv.telescope.antenna_positions)
antpos[1, 0] = 3000  # Baseline length
antpos_ecef = pyuvdata.utils.ECEF_from_ENU(antpos, center_loc=uv.telescope.location)
telescope_ecef_xyz = Quantity(uv.telescope.location.geocentric).to_value("m")
uv.telescope.antenna_positions  = antpos_ecef - telescope_ecef_xyz

freq_hz = 50e6
channel_width = 2e3
total_time_interval_s = 10
total_freq_interval_hz = 200e3
integration_time = 0.1
time_array_s = np.arange(-total_time_interval_s/2, total_time_interval_s/2, integration_time)
uv.freq_array = np.arange(freq_hz-total_freq_interval_hz/2, freq_hz+total_freq_interval_hz/2, channel_width)
uv.Nfreqs = len(uv.freq_array)
uv.time_array = np.mean(uv.time_array) + time_array_s/86400
uv.Ntimes = len(time_array_s)
uv.Nbls = 1
uv.Nblts = uv.Ntimes
uv.Npols = 1
uv.Nants_data = 2
uv.Nspws = 1
uv.flex_spw_id_array = np.full(uv.Nfreqs, np.mean(uv.spw_array), dtype=int)
uv.ant_1_array = np.zeros(uv.Nblts, dtype=int)
uv.ant_2_array = np.ones(uv.Nblts, dtype=int)
uv.channel_width = np.full(uv.Nfreqs, channel_width)
uv.integration_time = np.full(uv.Nblts, integration_time)
uv.baseline_array = np.full(uv.Nblts, 2048 + 2**16)
uv.data_array = np.zeros((uv.Nblts, uv.Nfreqs, uv.Npols), dtype=complex)
uv.flag_array = np.zeros((uv.Nblts, uv.Nfreqs, uv.Npols), dtype=bool)
uv.nsample_array = np.ones((uv.Nblts, uv.Nfreqs, uv.Npols), dtype=float)
uv.polarization_array = np.array([-5])
uv.set_lsts_from_time_array()
uv.phase_center_app_ra = uv.phase_center_app_ra[:uv.Nblts]
uv.phase_center_app_dec = uv.phase_center_app_dec[:uv.Nblts]
uv.phase_center_frame_pa = uv.phase_center_frame_pa[:uv.Nblts]
uv.phase_center_id_array = uv.phase_center_id_array[:uv.Nblts]
uv.scan_number_array = None
uv.uvw_array = np.zeros((uv.Nblts, 3), dtype=float)
uv.phase_to_time(np.mean(uv.time_array))
# Redefine uvws after phasing
uv.set_uvws_from_antenna_positions()
#uv.uvw_array = np.zeros((uv.Nblts, 3), dtype=float)
#uv.uvw_array[:, 0] = 3e3

beam = pyuvdata.UVBeam()
beam.read("/lustre/rbyrne/LWA_10to100_MROsoil_efields.fits")
beam.select(feeds="e")
beam.peak_normalize()
beam_use_freq = 50e6
use_freq_ind = np.where(beam.freq_array == beam_use_freq)[0]
beam.data_array[:, :, :, :, :] = beam.data_array[:, :, use_freq_ind, :, :]  # Make the beam frequency-invariant
beam_list = pyuvsim.BeamList(beam_list=[beam])

ra_offset_vals_deg = np.linspace(-60, 60, num=11)
dec_offset_vals_deg = np.linspace(-52, 52, num=11)
ra_vals_expanded, dec_vals_expanded = np.meshgrid(ra_offset_vals_deg, dec_offset_vals_deg)
ra_offset_vals_flattened = ra_vals_expanded.flatten()
dec_offset_vals_flattened = dec_vals_expanded.flatten()
source_coords = []
for ind in range(len(ra_offset_vals_flattened)):
    new_coords = astropy.coordinates.SkyCoord([astropy.coordinates.ICRS(
        ra=np.mean(uv.lst_array)*astropy.units.rad + ra_offset_vals_flattened[ind]*astropy.units.deg, 
        dec=uv.telescope.location.lat + dec_offset_vals_flattened[ind]*astropy.units.deg
    )])
    source_coords.append(new_coords)

for ind in range(len(ra_offset_vals_flattened)):
    output_filename = f"/lustre/rbyrne/decorr_sims/source{ind+1:03d}.uvfits"
    if os.path.isfile(output_filename):
        continue
    cat = pyradiosky.SkyModel(
        skycoord = source_coords[ind],
        name = ["source1"],
        spectral_type = "spectral_index",
        spectral_index = [0.],
        reference_frequency = ([np.mean(uv.freq_array)])*astropy.units.Hz,
        stokes = np.array([1,0,0,0]).reshape(4,1,1) * astropy.units.Jy,
    )
    output_uv = pyuvsim.uvsim.run_uvdata_uvsim(
        input_uv=uv,
        beam_list=beam_list,
        beam_dict=None,  # Same beam for all ants
        catalog=cat,
        quiet=False,
    )
    output_uv.phase_to_time(np.mean(output_uv.time_array))
    output_uv.write_uvfits(f"/lustre/rbyrne/decorr_sims/source{ind+1:03d}.uvfits")