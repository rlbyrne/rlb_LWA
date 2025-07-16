import pyuvsim
import pyuvdata
import pyradiosky
import astropy
import numpy as np
import os
from astropy.units import Quantity

uv = pyuvdata.UVData()
#uv.read("/lustre/cosmopipe/2025-06-13/20250613_070132_55MHz.ms")  # Choose an arbitrary data file for reference
uv.read("/lustre/pipeline/slow/73MHz/2025-07-14/19/20250714_192958_73MHz.ms")

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

use_azimuths = np.arange(0, 180, 45)
use_zas = np.arange(15, 90, 15)
azimuths, zenith_angles = np.meshgrid(use_azimuths, use_zas)
azimuths = azimuths.flatten()
zenith_angles = zenith_angles.flatten()
ras = np.mean(uv.lst_array) - np.arctan2(
    np.sin(uv.telescope.location.lat.rad) * np.sin(np.deg2rad(zenith_angles)) * np.sin(np.deg2rad(azimuths))
    - np.cos(uv.telescope.location.lat.rad) * np.cos(np.deg2rad(zenith_angles)),
    np.sin(np.deg2rad(zenith_angles)) * np.cos(np.deg2rad(azimuths))
) - np.pi/2
decs = np.pi/2 - np.arccos(
    np.cos(uv.telescope.location.lat.rad) * np.sin(np.deg2rad(zenith_angles)) * np.sin(np.deg2rad(azimuths))
    + np.sin(uv.telescope.location.lat.rad) * np.cos(np.deg2rad(zenith_angles))
)
source_coords = []
for ind in range(len(ras)):
    new_coords = astropy.coordinates.SkyCoord([astropy.coordinates.ICRS(
        ra=ras[ind]*astropy.units.rad, 
        dec=decs[ind]*astropy.units.rad
    )])
    source_coords.append(new_coords)

for ind in range(len(ras)):
    output_filename = f"/lustre/rbyrne/decorr_sims/source_sim_za{int(zenith_angles[ind])}_az{int(azimuths[ind])}.uvfits"
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
    output_uv.write_uvfits(output_filename)