from celery import Celery
import time
import sys

sys.path.append("/home/rbyrne/rlb_LWA/LWA_data_preprocessing")
from generate_model_vis_fftvis import run_fftvis_diffuse_sim

app = Celery(
    "cosmo",
    broker="pyamqp://rbyrne:rbyrne@rabbitmq.calim.mcs.pvt:5672/rbyrne",
    backend="redis://10.41.0.85:6379/0",
    # include=["run_simulation_celery"],
)

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=7200,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=20,
    task_serializer="json",
    # task_serializer="pickle",
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
)


@app.task
def run_simulation_celery(
    map_path,
    beam_path,
    input_data_path,
    output_uvfits_path,
):
    run_fftvis_diffuse_sim(
        map_path=map_path,
        beam_path=beam_path,
        input_data_path=input_data_path,
        output_uvfits_path=output_uvfits_path,
        log_path=None,
    )


@app.task
def test_function(filename, number):
    f = open(filename, "a")
    f.write(f"started job {number}\n")
    f.close()
    time.sleep(60)
    f = open(filename, "a")
    f.write(f"ended job {number}\n")
    f.close()


if __name__ == "__main__":
    app.start()
