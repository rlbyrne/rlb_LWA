# from generate_model_vis_fftvis import run_fftvis_diffuse_sim
from run_lwa_jobs_celery import run_simulation_celery, test_function


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
        run_simulation_celery.delay(
            f"/lustre/rbyrne/skymodels/Gasperin2020_sources_plus_{file_name}.skyh5",
            "/lustre/rbyrne/LWA_10to100_MROsoil_efields.fits",
            f"/lustre/gh/2024-03-02/calibration/ruby/{file_name}.ms",
            f"/lustre/rbyrne/2024-03-02/calibration_models/{file_name}_deGasperin_sources.uvfits",
        )


if __name__ == "__main__":
    run_compact_source_sims_Aug5()
