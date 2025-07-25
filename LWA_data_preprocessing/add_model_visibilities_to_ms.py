from casatools import table
import casatasks
import os
import sys
import datetime
import argparse


def add_model_visibilities_to_ms(
    data_file,
    model_file,
):
    
    tb = table()

    # Open the model file and read its DATA column
    tb.open(model_file)
    model_data = tb.getcol('DATA')
    tb.close()

    # Open the data file and write to its MODEL_DATA column
    casatasks.clearcal(vis=data_file, addmodel=True)  # Create MODEL_DATA column
    tb.open(data_file, nomodify=False)
    tb.putcol('MODEL_DATA', model_data)
    tb.close()

def casa_calibrate(
    ms_path, 
    caltable=None, 
    min_cal_baseline_lambda=10,
    max_cal_baseline_lambda=125,
):

    if caltable is None:
        caltable = f"{ms_path.removesuffix(".ms")}.bcal"
    casatasks.bandpass(
        ms_path,
        caltable,
        uvrange=f"{min_cal_baseline_lambda}~{max_cal_baseline_lambda}lambda",
        fillgaps=1,
    )


if __name__ == "__main__":

    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--data_file",
        nargs=1,
        type=str,
    )
    CLI.add_argument(
        "--model_file",
        nargs=1,
        type=str,
    )
    CLI.add_argument(
        "--min_cal_baseline_lambda",
        nargs=1,
        type=int,
        default=None,
    )
    CLI.add_argument(
        "--max_cal_baseline_lambda",
        nargs=1,
        type=int,
        default=None,
    )

    # parse the command line
    args = CLI.parse_args()

    concatenate_files(
        args.path_in,
        args.path_out[0],
    )
    
    add_model_visibilities_to_ms(
        "/lustre/21cmpipe/2025-05-08/20250508_160736-160926_41MHz.ms",
        "/lustre/21cmpipe/2025-05-08/20250508_160736-160926_41MHz_source_sim.ms",
    )