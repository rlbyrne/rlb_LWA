script_path=/home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis.py
obsname=cal46_time11_conj
input_obs=/data03/rbyrne/20231222/test_pyuvsim_modeling/${obsname}.ms
#beam_file=/home/rbyrne/rlb_LWA/LWAbeam_2015.fits
beam_file=/data03/rbyrne/LWA_10to100.beamfits
output_path=/data03/rbyrne/20231222/test_pyuvsim_modeling
catalog_path=/fast/rbyrne/skymodels
#mpirun -n 20 python ${script_path} ${catalog_path}/cyg_cas.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_cyg_cas_sim.uvfits
#mpirun -n 20 python ${script_path} ${catalog_path}/Gasperin2020_cyg_cas.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_deGasperin_cyg_cas_sim.uvfits
#mpirun -n 20 python ${script_path} ${catalog_path}/Gasperin2020_sources.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_deGasperin_sources_sim.uvfits
#mpirun -n 20 python ${script_path} ${catalog_path}/FullVLSSCatalog.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_VLSS_sim.uvfits
#mpirun -n 20 python ${script_path} ${catalog_path}/Gasperin2020_cyg_cas_48MHz.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_deGasperin_cyg_cas_48MHz_sim.uvfits
mpirun -n 20 python ${script_path} ${catalog_path}/Cas_single_source_48MHz.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_Cas_single_source_48MHz_sim.uvfits

# Run diffuse
#diffuse_script_path=/home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis_diffuse.py
#mpirun -n 15 python ${diffuse_script_path} ${catalog_path}/ovro_lwa_sky_map_73.152MHz.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_mmode_2048_sim.uvfits

# Test Nivedita's beam
#nm_beam_file=/data03/rbyrne/LWA_10to100.beamfits
#mpirun -n 20 python ${script_path} ${catalog_path}/Gasperin2020_cyg_cas.skyh5 ${nm_beam_file} ${input_obs} ${output_path}/${obsname}_deGasperin_cyg_cas_sim_NMbeam.uvfits
