script_path=/home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis.py
obsname=20230819_093023_73MHz
input_obs=/data03/rbyrne/20230819/${obsname}.ms
beam_file=/home/rbyrne/rlb_LWA/LWAbeam_2015.fits
catalog_file=/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5
output_path=/data03/rbyrne/20230819/simulation_outputs
mpirun -n 20 python ${script_path} ${catalog_file} ${beam_file} ${input_obs} ${output_path}/${obsname}_cyg_cas_sim.uvfits
