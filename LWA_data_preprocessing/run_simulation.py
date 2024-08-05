from generate_model_vis_fftvis import run_fftvis_diffuse_sim


def run_compact_source_sims_Aug5():

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

        run_fftvis_diffuse_sim.delay(
            map_path=f"/lustre/rbyrne/2024-03-02/calibration_models/Gasperin2020_sources_plus_{file_name}.skyh5",
            beam_path="/lustre/rbyrne/LWA_10to100_MROsoil_efields.fits",
            input_data_path=f"/lustre/gh/2024-03-02/calibration/ruby/{file_name}.ms",
            output_uvfits_path=f"/lustre/rbyrne/2024-03-02/calibration_models/{file_name}_deGasperin_sources.uvfits",
        )


if __name__ == "__main__":
    run_compact_source_sims_Aug5()
