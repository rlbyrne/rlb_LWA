import os
import numpy as np
import pyuvdata


def copy_data():

    for subband in [
        "13MHz",
        "18MHz",
        "23MHz",
        "27MHz",
        "32MHz",
        "36MHz",
        "41MHz",
        "46MHz",
        "50MHz",
        "55MHz",
        "59MHz",
        "64MHz",
        "69MHz",
        "73MHz",
        "78MHz",
        "82MHz",
    ]:
        if not os.path.isdir(
            f"/lustre/21cmpipe/solar_panel_test/{subband}/2025-08-19/04"
        ):
            os.makedirs(
                f"/lustre/21cmpipe/solar_panel_test/{subband}/2025-08-19/04",
                exist_ok=True,
            )
        os.system(
            f"cp -r /lustre/pipeline/slow/{subband}/2025-08-19/04 /lustre/21cmpipe/solar_panel_test/{subband}/2025-08-19"
        )
        os.system(
            f"cp -r /lustre/pipeline/slow/{subband}/2025-08-19/05 /lustre/21cmpipe/solar_panel_test/{subband}/2025-08-19"
        )


def extract_autos():

    date_str = "2025-08-19"
    hours = ["04", "05"]
    subbands = [
        "13MHz",
        "18MHz",
        "23MHz",
        "27MHz",
        "32MHz",
        "36MHz",
        "41MHz",
        "46MHz",
        "50MHz",
        "55MHz",
        "59MHz",
        "64MHz",
        "69MHz",
        "73MHz",
        "78MHz",
        "82MHz",
    ]
    filenames = [
        os.listdir(
            f"/lustre/21cmpipe/solar_panel_test/{subbands[0]}/{date_str}/{hour}"
        )[:]
        for hour in hours
    ]
    filetimes = np.array(
        [name[0:15] for filenames_sublist in filenames for name in filenames_sublist]
    )

    for time_str in filetimes:
        first_file = True
        for subband in subbands:
            new_uv = pyuvdata.UVData()
            new_uv.read_ms(
                f"/lustre/21cmpipe/solar_panel_test/{subband}/{date_str}/{time_str[9:11]}/{time_str}_{subband}.ms"
            )
            new_uv.select(ant_str="auto")
            new_uv.select(
                antenna_names=[
                    "LWA364",
                    "LWA362",
                    "LWA363",
                    "LWA356",
                    "LWA355",
                    "LWA263",
                    "LWA302",
                ],
            )
            if first_file:
                uv = new_uv
                first_file = False
            else:
                uv += new_uv
        uv.write_uvfits(
            f"/lustre/21cmpipe/solar_panel_test/{time_str}_autos_selected.uvfits"
        )


if __name__ == "__main__":
    extract_autos()
