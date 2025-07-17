import casatasks
import os
import sys
import datetime
import argparse


def find_and_concat_files():

    use_freq_bands = [
        "41",
        "46",
        "50",
        "55",
        "59",
        "64",
        "69",
        "73",
        "78",
        "82",
    ]
    delta_time = datetime.timedelta(minutes=2)
    start_time = datetime.datetime(2025, 5, 5, 12, 56, 9)
    time_steps = 2

    for time_step in range(time_steps):

        use_start_time = start_time + time_step * delta_time
        end_time = start_time + (time_step + 1) * delta_time
        min_time_str = use_start_time.strftime("%H%M%S")
        max_time_str = end_time.strftime("%H%M%S")
        year = use_start_time.strftime("%Y")
        month = use_start_time.strftime("%m")
        day = use_start_time.strftime("%d")

        for freq_band in use_freq_bands:

            datadir = (
                f"/lustre/pipeline/calibration/{freq_band}MHz/{year}-{month}-{day}/12"
            )
            # datadir = f"/lustre/pipeline/slow/{freq_band}MHz/{year}-{month}-{day}/12"
            copied_data_dir = f"/lustre/rbyrne/{year}-{month}-{day}"

            if not os.path.isdir(
                copied_data_dir
            ):  # Make target directory if it does not exist
                os.mkdir(copied_data_dir)

            all_files = os.listdir(datadir)
            use_files = [
                filename
                for filename in all_files
                if filename.startswith(f"{year}{month}{day}")
                and filename.endswith(".ms")
            ]
            print(use_files)
            print(min_time_str)
            print(max_time_str)
            use_files = [
                filename
                for filename in use_files
                if (int(filename.split("_")[1]) >= int(min_time_str))
                and int(filename.split("_")[1]) < int(max_time_str)
            ]
            if len(use_files) != 12:
                print("ERROR: Number of files found is not 12.")
                sys.exit()

            use_files.sort()
            output_filename = f"{year}{month}{day}_{use_files[0].split('_')[1]}-{use_files[-1].split('_')[1]}_{freq_band}MHz.ms"

            if not os.path.isdir(f"{copied_data_dir}/{output_filename}"):
                # Copy files
                for filename in use_files:
                    if not os.path.isfile(f"{copied_data_dir}/{filename}"):
                        print(f"Copying file {filename}")
                        os.system(
                            f"cp -r {datadir}/{filename} {copied_data_dir}/{filename}"
                        )
                use_files_full_paths = [
                    f"{copied_data_dir}/{filename}" for filename in use_files
                ]

            concatenate_files(
                use_files_full_paths,
                f"{copied_data_dir}/{output_filename}",
                run_aoflagger=True,
            )


def concatenate_files(
    use_files_full_paths,
    output_filename,
    run_aoflagger=True,
):

    # Concatenate files
    casatasks.virtualconcat(
        vis=[filename for filename in use_files_full_paths], concatvis=output_filename
    )

    if run_aoflagger:
        os.system(f"aoflagger {output_filename}")


if __name__ == "__main__":

    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--path_in",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
    )
    CLI.add_argument(
        "--path_out",
        nargs=1,
        type=str,  # any type/callable can be used here
    )

    # parse the command line
    args = CLI.parse_args()

    concatenate_files(
        args.path_in,
        args.path_out[0],
    )
