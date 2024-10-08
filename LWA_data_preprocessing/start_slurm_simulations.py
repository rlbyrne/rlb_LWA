import os
import numpy as np


def extended_source_and_diffuse_sims_Oct2():
    source_skymodel = "/lustre/rbyrne/skymodels/Gasperin2020_sources_plus_64.skyh5"
    diffuse_skymodel = (
        "/lustre/rbyrne/skymodels/ovro_lwa_sky_map_36-73MHz_nside512.skyh5"
    )
    beam = "/lustre/rbyrne/LWA_10to100_MROsoil_efields.fits"
    subbands = ["41", "46", "50", "55", "59", "64", "69", "73", "78", "82"]
    use_time_offsets = np.arange(12, 24)

    for time_offset in use_time_offsets:
        for use_subband in subbands:
            reference_file = f"/lustre/rbyrne/2024-03-03/reference_data_for_sims/20240303_093000_{use_subband}MHz.ms"
            time_string = str(93000 + 10 * time_offset).zfill(6)

            output_file = f"/lustre/rbyrne/simulation_outputs/20240303_{time_string}_{use_subband}MHz_source_sim.uvfits"
            if os.path.isfile(output_file):
                print(f"File {output_file} exists. Skipping.")
            else:
                os.system(
                    f"sbatch /home/rbyrne/rlb_LWA/LWA_data_preprocessing/run_simulation_slurm.sh '{source_skymodel}' '{beam}' '{reference_file}' '{output_file}' {time_offset}"
                )

            output_file = f"/lustre/rbyrne/simulation_outputs/20240303_{time_string}_{use_subband}MHz_diffuse_sim.uvfits"
            if os.path.isfile(output_file):
                print(f"File {output_file} exists. Skipping.")
            else:
                os.system(
                    f"sbatch /home/rbyrne/rlb_LWA/LWA_data_preprocessing/run_simulation_slurm.sh '{diffuse_skymodel}' '{beam}' '{reference_file}' '{output_file}' {time_offset}"
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


if __name__ == "__main__":

    extended_source_and_diffuse_sims_Oct2()
