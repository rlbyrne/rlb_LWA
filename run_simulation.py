from run_lwa_jobs_celery import (
    run_simulation_celery,
    run_calibration_celery,
    test_function,
)


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


def run_calibration_Aug7():

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
        # "82",
    ]

    for file_name in use_filenames:

        datafile = f"/lustre/gh/2024-03-02/calibration/ruby/{file_name}.ms"
        model_file = f"/lustre/rbyrne/2024-03-02/calibration_models/{file_name}_deGasperin_sources.ms"

        run_calibration_celery(
            datafile,
            model_file,
            "DATA",
            "DATA",
            True,
            10,
            125,
            False,
            f"/lustre/rbyrne/2024-03-02/calibration_outputs/{file_name}_extended_sources_cal_log.txt",
            f"/lustre/rbyrne/2024-03-02/calibration_outputs/{file_name}_extended_sources.calfits",
            f"/lustre/rbyrne/2024-03-02/calibration_outputs/{file_name}_calibrated_extended_sources.ms",
        )


if __name__ == "__main__":
    run_compact_source_sims_Aug5()
