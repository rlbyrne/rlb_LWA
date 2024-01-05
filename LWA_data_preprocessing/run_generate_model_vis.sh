script_path=/home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis.py
obsname=20230819_093023_73MHz
input_obs=/data03/rbyrne/20230819/${obsname}.ms
#beam_file=/home/rbyrne/rlb_LWA/LWAbeam_2015.fits
output_path=/data03/rbyrne/20230819/simulation_outputs
catalog_path=/fast/rbyrne/skymodels
#mpirun -n 20 python ${script_path} ${catalog_path}/cyg_cas.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_cyg_cas_sim.uvfits
#mpirun -n 20 python ${script_path} ${catalog_path}/Gasperin2020_cyg_cas.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_deGasperin_cyg_cas_sim.uvfits
#mpirun -n 20 python ${script_path} ${catalog_path}/Gasperin2020_sources.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_deGasperin_sources_sim.uvfits
#mpirun -n 20 python ${script_path} ${catalog_path}/FullVLSSCatalog.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_VLSS_sim.uvfits
mpirun -n 15 python /home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis_diffuse.py ${catalog_path}/ovro_lwa_sky_map_73.152MHz_equatorial.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_mmode_sim.uvfits

# Test Nivedita's beam
#beam_file=/data03/rbyrne/LWA_10to100.beamfits
#mpirun -n 20 python ${script_path} ${catalog_path}/Gasperin2020_cyg_cas.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_deGasperin_cyg_cas_sim_NMbeam.uvfits
