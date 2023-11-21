import numpy as np
import glob
import pyuvdata
import os
import sys
import argparse
import math
import multiprocessing as mp
import casacore.tables as tbl
import aoflagger as aof

# Adapted from Nivedita Mahesh

parser = argparse.ArgumentParser()
parser.add_argument("-file_path", type=str)
parser.add_argument("-strategy_path", type=str)
parser.add_argument("-read_prog", type=int)
parser.add_argument("-out_dir", type=str)
parser.add_argument("-time_chunk", type=int)
parser.add_argument("-threads", type=int)
args = parser.parse_args()

if (args.file_path).endswith(".ms"):  # Single file
    data_chunks = 1
    ndata_chunks = 1
    fulldayrun_path = [args.file_path]
else:
    fulldayrun_path = sorted(glob.glob(args.file_path + "/*"))

    total_t = len(fulldayrun_path) * 10 / (3600)
    print("No.of hours of data:", total_t)

    data_chunks = int(args.time_chunk * 60 / 10)
    ndata_chunks = len(fulldayrun_path) / data_chunks
    ## Set up AO flagger and image set buffer

flagger = aof.AOFlagger()
nch = 192
datasets = data_chunks
data1 = flagger.make_image_set(nch, datasets, 8)
strategy_aa = flagger.load_strategy_file(args.strategy_path)


def do_aoflagging(base_list, count):

    for i in range(4):
        data1.set_image_buffer(2 * i, np.real(base_list[:, :, i]))
        data1.set_image_buffer(2 * i + 1, np.imag(base_list[:, :, i]))
    flags_aa1 = strategy_aa.run(data1)

    if count % 10000 == 0:
        print("No:of baselines processed by AOFlagger-", count)
    return flags_aa1.get_buffer()


for time in range(math.ceil(ndata_chunks)):

    pool = mp.Pool(processes=args.threads)
    async_res = [
        pool.apply_async(read_data_from_ms, (fulldayrun_path[i], args.read_prog, i))
        for i in range(time * data_chunks, (time + 1) * data_chunks)
    ]
    dataset = np.array([file_o.get() for file_o in async_res])
    # print(args.time_chunk, "mins of data read")
    pool = mp.Pool(processes=args.threads)

    baseline_res = [
        pool.apply_async(do_aoflagging, (dataset[:, k, :, :], k))
        for k in range(np.shape(dataset)[1])
    ]
    final_flag = np.array([result_base.get() for result_base in baseline_res])

    # print(
    #    str((time + 1) * args.time_chunk),
    #    "/",
    #    str(math.ceil(ndata_chunks) * args.time_chunk),
    #    "min have been processed & saved",
    # )

    flags_dim = np.repeat(final_flag[:, :, :, np.newaxis], 4, axis=3)
    flag_copy = np.zeros((62128, 1, 192, 4))

    for i in range(data_chunks):

        uvd = pyuvdata.UVData()
        uvd.read_ms(fulldayrun_path[time * data_chunks + i], run_check=False)
        uvd.reorder_pols(order="CASA", run_check=False)
        flag_copy[:, 0, :, :] = flags_dim[:, i, :, :]
        uvd.data_array[np.where(flag_copy)] = 0.0
        uvd.write_ms(
            args.out_dir + org_msfiles[time * data_chunks + i].split("/")[-1],
            run_check=False,
        )

plot = 0
if plot == 1:
    masked_array = np.ma.array(b1_data_set69[j, :, :, 0])
    masked_array[flagvalues_aa1 == 1] = np.ma.masked
