from celery import Celery
import time


app = Celery(
    "cosmo",
    # broker="pyamqp://pipe:pipe@rabbitmq.calim.mcs.pvt:5672/rbyrne",
    broker="pyamqp://rbyrne:rbyrne@rabbitmq.calim.mcs.pvt:5672/rbyrne",
    backend="redis://10.41.0.85:6379/0",
)

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=7200,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=20,
    task_serializer="json",
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
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
