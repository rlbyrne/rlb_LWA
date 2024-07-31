import multiprocessing
import psutil
import time


def get_usage(proc):

    use_interval = 10
    pid = proc.pid

    p = psutil.Process(pid)
    while True:
        cpu_utilization = p.cpu_percent(interval=use_interval)
        memory = p.memory_info()[3] / 1000000000
        children = p.children(recursive=True)
        for child in children:
            cpu_utilization += child.cpu_percent(interval=use_interval)
            memory += child.memory_info()[3] / 1000000000
        print(f"CPU utilization (%): {cpu_utilization}")
        print(f"Memory utilization (GB): {memory}")
        time.sleep(use_interval)


def prnt_squ():

    i = 0
    while i < 20:
        print(i)
        time.sleep(2)
        i += 1


if __name__ == "__main__":

    proc1 = multiprocessing.Process(target=prnt_squ)
    proc1.start()

    proc2 = multiprocessing.Process(
        target=get_usage,
        args=[
            proc1,
        ],
    )
    proc2.start()

    proc1.join()
    proc2.terminate()
    proc2.join()
