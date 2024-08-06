script_path=/home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis.py
obsname=cal46_time11_conj
input_obs=/data03/rbyrne/20231222/test_pyuvsim_modeling/${obsname}.ms
#beam_file=/home/rbyrne/rlb_LWA/LWAbeam_2015.fits
beam_file=/data03/rbyrne/LWA_10to100.beamfits
output_path=/data03/rbyrne/20231222/test_diffuse_normalization
catalog_path=/fast/rbyrne/skymodels
#mpirun -n 20 python ${script_path} ${catalog_path}/cyg_cas.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_cyg_cas_sim.uvfits
#mpirun -n 20 python ${script_path} ${catalog_path}/Gasperin2020_cyg_cas.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_deGasperin_cyg_cas_sim.uvfits
#mpirun -n 20 python ${script_path} ${catalog_path}/Gasperin2020_sources.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_deGasperin_sources_sim.uvfits
#mpirun -n 20 python ${script_path} ${catalog_path}/FullVLSSCatalog.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_VLSS_sim.uvfits
#mpirun -n 20 python ${script_path} ${catalog_path}/Gasperin2020_sources_48MHz.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_deGasperin_sources_48MHz_sim.uvfits
# mpirun -n 20 python ${script_path} ${catalog_path}/Gasperin2020_sources_plus_48MHz.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_deGasperin_sources_plus_48MHz_sim.uvfits

# Run diffuse
#diffuse_script_path=/home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis_diffuse.py
#mpirun -n 15 python ${diffuse_script_path} ${catalog_path}/ovro_lwa_sky_map_46.992MHz_nside512.skyh5 ${beam_file} ${input_obs} ${output_path}/${obsname}_mmode_46.992MHz_nside512_sim.uvfits

# Run matvis
map_path="/fast/rbyrne/skymodels/ovro_lwa_sky_map_46.992MHz_nside512.skyh5"
beam_path="/data03/rbyrne/LWA_10to100.beamfits"
input_obs="/data03/rbyrne/20231222/compare_lsts/46_time210_conj.ms"
output_uvfits_path="/data03/rbyrne/20231222/compare_lsts/46_time210_mmode_matvis_sim_nside512.uvfits"
python /home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis_matvis.py ${map_path} ${beam_path} ${input_obs} ${output_uvfits_path}

map_path="/fast/rbyrne/skymodels/ovro_lwa_sky_map_46.992MHz_nside512.skyh5"
beam_path="/data03/rbyrne/LWA_10to100.beamfits"
input_obs="/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj.ms"
output_uvfits_path="/data03/rbyrne/20231222/compare_lsts/cal46_time11_mmode_matvis_sim_nside512_testgpu.uvfits"
python /home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis_matvis.py ${map_path} ${beam_path} ${input_obs} ${output_uvfits_path}

# Run source model
map_path="/fast/rbyrne/skymodels/Gasperin2020_sources_plus_48MHz.skyh5"
beam_path="/data03/rbyrne/LWA_10to100.beamfits"
input_obs="/data03/rbyrne/20231222/compare_lsts/46_time210_conj.ms"
output_uvfits_path="/data03/rbyrne/20231222/compare_lsts/46_time210_Gasperin_plus.uvfits"
python /home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis.py ${map_path} ${beam_path} ${input_obs} ${output_uvfits_path}

# Run matvis, testing GPUs
map_path="/fast/rbyrne/skymodels/ovro_lwa_sky_map_46.992MHz_nside512.skyh5"
beam_path="/data03/rbyrne/LWA_10to100.beamfits"
input_obs="/data03/rbyrne/20231222/compare_lsts/46_time210_conj.ms"
output_uvfits_path="/data03/rbyrne/20231222/compare_lsts/46_time210_mmode_matvis_sim_nside512_gpu.uvfits"
python /home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis_matvis.py ${map_path} ${beam_path} ${input_obs} ${output_uvfits_path}


###### Run source models for Gregg's data 7/17/24 ######
script_path="/home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis.py"
catalog_path="/fast/rbyrne/skymodels/Gasperin2020_point_sources_plus_48MHz.skyh5"
beam_file="/lustre/rbyrne/LWA_10to100_MROsoil_efields.fits"
input_obs="/lustre/rbyrne/2024-03-02/46_time1.ms"
output_file="/lustre/rbyrne/2024-03-02/calibration_models/46_time1_deGasperin_point_sources.ms"
mpirun -n 10 python ${script_path} ${catalog_path} ${beam_file} ${input_obs} ${output_file}

###### Run source models with extended components for Gregg's data 8/5/24 ######
script_path="/home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis_fftvis.py"
catalog_path="/fast/rbyrne/skymodels/Gasperin2020_point_sources_plus_48MHz.skyh5"
beam_file="/lustre/rbyrne/LWA_10to100_MROsoil_efields.fits"
input_obs="/lustre/rbyrne/2024-03-02/46_time1.ms"
output_file="/lustre/rbyrne/2024-03-02/calibration_models/46_time1_deGasperin_point_sources.ms"
python ${script_path} ${catalog_path} ${beam_file} ${input_obs} ${output_file}
