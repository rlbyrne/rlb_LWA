import os
import numpy as np
import pyradiosky


def extended_source_and_diffuse_sims_Oct2():
    source_skymodel = "/lustre/rbyrne/skymodels/Gasperin2020_sources_plus_64.skyh5"
    diffuse_skymodel = (
        "/lustre/rbyrne/skymodels/ovro_lwa_sky_map_36-73MHz_nside512.skyh5"
    )
    beam = "/lustre/rbyrne/LWA_10to100_MROsoil_efields.fits"
    subbands = ["41", "46", "50", "55", "59", "64", "69", "73", "78", "82"]
    #use_time_offsets = np.arange(500, 600)
    use_time_offsets = np.arange(600, 700)

    for time_offset in use_time_offsets:
        for use_subband in subbands:
            reference_file = f"/lustre/rbyrne/2024-03-03/reference_data_for_sims/20240303_093000_{use_subband}MHz.ms"
            time_string = str(93000 + 10 * time_offset).zfill(6)

            output_file = f"/lustre/rbyrne/simulation_outputs/20240303_{time_string}_{use_subband}MHz_source_sim.uvfits"
            if os.path.isfile(output_file):
                print(f"File {output_file} exists. Skipping.")
            else:
                os.system(
                    f"sbatch --nice=1000 /home/rbyrne/rlb_LWA/LWA_data_preprocessing/run_simulation_slurm.sh '{source_skymodel}' '{beam}' '{reference_file}' '{output_file}' {time_offset}"
                )

            output_file = f"/lustre/rbyrne/simulation_outputs/20240303_{time_string}_{use_subband}MHz_diffuse_sim.uvfits"
            if os.path.isfile(output_file):
                print(f"File {output_file} exists. Skipping.")
            else:
                os.system(
                    f"sbatch --nice=1000 /home/rbyrne/rlb_LWA/LWA_data_preprocessing/run_simulation_slurm.sh '{diffuse_skymodel}' '{beam}' '{reference_file}' '{output_file}' {time_offset}"
                )


def cygA_only_sims_Oct2():

    beam = "/lustre/rbyrne/LWA_10to100_MROsoil_efields.fits"
    use_filenames = [
        "18",
        "23",
        "27",
        "36",
        "41",
        "46",
        "50",
        "55",
        "59",
        "64",
        "73",
        "78",
        "82",
    ]

    for file_name in use_filenames:
        datafile = f"/lustre/gh/2024-03-02/calibration/ruby/{file_name}.ms"
        source_skymodel = (
            "/lustre/rbyrne/skymodels/Gasperin2020_CygA_point_source_48MHz.skyh5"
        )
        output_file = f"/lustre/rbyrne/2024-03-02/ruby/calibration_models/{file_name}_cygA_point_source.ms"
        if os.path.isfile(output_file):
            print(f"File {output_file} exists. Skipping.")
        else:
            os.system(
                f"sbatch /home/rbyrne/rlb_LWA/LWA_data_preprocessing/run_simulation_slurm.sh '{source_skymodel}' '{beam}' '{datafile}' '{output_file}' 0"
            )


def format_data_Oct9():

    freq_bands = ["41", "46", "50", "55", "59", "64", "69", "73", "78", "82"]
    freq_bands = freq_bands[2:]
    for use_band in freq_bands:
        os.system(
            f"sbatch /home/rbyrne/rlb_LWA/LWA_data_preprocessing/run_format_data_slurm.sh '{use_band}'"
        )


def calibrate_data_Oct18():

    freq_bands = [
        #"41",
        #"46",
        #"50",
        #"55",
        #"59",
        "64",
        "69",
        "73",
        "78",
        "82",
    ]
    for use_band in freq_bands:
        os.system(
            f"sbatch /home/rbyrne/rlb_LWA/LWA_data_preprocessing/run_newcal_slurm.sh '{use_band}'"
        )

def create_ddcal_sims_Nov11():

    source_skymodel = "/lustre/rbyrne/skymodels/Gasperin2020_sources_plus_64.skyh5"
    diffuse_skymodel = "/lustre/rbyrne/skymodels/ovro_lwa_sky_map_36-73MHz_nside512.skyh5"

    skymodel_names = [
        "Cas",
        "Cyg",
        "Vir",
    ]

    beam = "/lustre/rbyrne/LWA_10to100_MROsoil_efields.fits"
    use_subband = "73"
    time_offset = 0

    reference_file = f"/lustre/rbyrne/2024-03-03/20240303_133205_73MHz.ms"

    # Run diffuse
    output_file_diffuse = f"/lustre/rbyrne/2024-03-03/20240303_133205_73MHz_model_diffuse.ms"
    #os.system(
    #    f"sbatch /home/rbyrne/rlb_LWA/LWA_data_preprocessing/run_simulation_slurm.sh '{diffuse_skymodel}' '{beam}' '{reference_file}' '{output_file_diffuse}' {time_offset}"
    #)

    for use_name in skymodel_names:
        output_cat_name = f"/lustre/rbyrne/skymodels/Gasperin2020_sources_{use_name}_64.skyh5"
        output_file_source_sim = f"/lustre/rbyrne/2024-03-03/20240303_133205_73MHz_model_{use_name}.ms"
        cat = pyradiosky.SkyModel()
        cat.read(source_skymodel)
        comp_inds = np.where([name.startswith(use_name) for name in cat.name])[0]
        cat.select(component_inds=comp_inds, inplace=True)
        cat.write_skyh5(output_cat_name)
        os.system(
            f"sbatch /home/rbyrne/rlb_LWA/LWA_data_preprocessing/run_simulation_slurm.sh '{output_cat_name}' '{beam}' '{reference_file}' '{output_file_source_sim}' {time_offset}"
        )

if __name__ == "__main__":

    extended_source_and_diffuse_sims_Oct2()
