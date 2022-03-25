import plot_fits
import numpy as np


def calculate_source_values_Mar15():

    path = "/Users/ruby/Astro/polarized_source_sims_Feb2022/fhd_rlb_polarized_source_sim_optimal_weighting_Mar2021/output_data"
    filename_stokes_i = "polarized_source_MWA_sim_results_optimal_Dirty_I.fits"
    filename_stokes_q = "polarized_source_MWA_sim_results_optimal_Dirty_Q.fits"
    filename_stokes_u = "polarized_source_MWA_sim_results_optimal_Dirty_U.fits"
    filename_stokes_v = "polarized_source_MWA_sim_results_optimal_Dirty_V.fits"

    stokes_i = plot_fits.load_fits(f"{path}/{filename_stokes_i}")
    stokes_q = plot_fits.load_fits(f"{path}/{filename_stokes_q}")
    stokes_u = plot_fits.load_fits(f"{path}/{filename_stokes_u}")
    stokes_v = plot_fits.load_fits(f"{path}/{filename_stokes_v}")

    image_span = 100
    x_center = 1150
    y_center = 1151
    stokes_i.crop_image(
        new_x_range=[x_center - image_span / 2, x_center + image_span / 2],
        new_y_range=[y_center - image_span / 2, y_center + image_span / 2],
        inplace=True,
    )
    stokes_q.crop_image(
        new_x_range=[x_center - image_span / 2, x_center + image_span / 2],
        new_y_range=[y_center - image_span / 2, y_center + image_span / 2],
        inplace=True,
    )
    stokes_u.crop_image(
        new_x_range=[x_center - image_span / 2, x_center + image_span / 2],
        new_y_range=[y_center - image_span / 2, y_center + image_span / 2],
        inplace=True,
    )
    stokes_v.crop_image(
        new_x_range=[x_center - image_span / 2, x_center + image_span / 2],
        new_y_range=[y_center - image_span / 2, y_center + image_span / 2],
        inplace=True,
    )
    max_val = np.max(stokes_i.signal_arr)
    max_coords = np.where(stokes_i.signal_arr == max_val)
    print(f"Stokes I intensity: {stokes_i.signal_arr[max_coords]}")
    print(f"Stokes Q intensity: {stokes_q.signal_arr[max_coords]}")
    print(f"Stokes U intensity: {stokes_u.signal_arr[max_coords]}")
    print(f"Stokes V intensity: {stokes_v.signal_arr[max_coords]}")
    stokes_q.plot(
        # x_pixel_extent=[x_center - image_span / 2, x_center + image_span / 2],
        # y_pixel_extent=[y_center - image_span / 2, y_center + image_span / 2],
        signal_extent=[-7e5, 7e5],
    )


def plot_images_Mar17():

    path = "/Users/ruby/Astro/polarized_source_sims_Feb2022/fhd_rlb_polarized_source_sim_optimal_weighting_Mar2021/output_data"
    stokes_signal_extent = [-1e5, 3.5e5]
    stokes = ["I", "Q", "U", "V"]
    for stokes_name in stokes:
        filename = f"polarized_source_MWA_sim_results_optimal_Dirty_{stokes_name}.fits"
        plot_fits.plot_fits_file(
            f"{path}/{filename}",
            signal_extent=stokes_signal_extent,
            save_filename = f"/Users/ruby/Documents/2022 Winter/Polarimetry paper review/plots/Stokes{stokes_name}.png",
            title = f"Stokes {stokes_name}"
        )

    instr_signal_extents = [[-1e4, 3.5e4], [-1e4, 3.5e4], [-.5e3, 1.75e3], [-.5e3/120, 1.75e3/120]]
    instr_names = ["XX", "YY", "XY_real", "XY_imaginary"]
    titles = ["pp", "qq", "pq, Real Part", "pq, Imaginary Part"]
    for ind, instr_name in enumerate(instr_names):
        filename = f"polarized_source_MWA_sim_results_optimal_Dirty_{instr_name}.fits"
        plot_fits.plot_fits_file(
            f"{path}/{filename}",
            signal_extent=instr_signal_extents[ind],
            save_filename = f"/Users/ruby/Documents/2022 Winter/Polarimetry paper review/plots/Instr_{instr_name}.png",
            title = titles[ind]
        )


def plot_skyh5_test_Mar23():

    path = "/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_diffuse_skyh5_Mar2022/output_data"
    stokes = ["I", "Q", "U", "V"]
    for stokes_name in stokes:
        filename = f"1061316296_optimal_Model_{stokes_name}.fits"
        plot_fits.plot_fits_file(
            f"{path}/{filename}",
            save_filename = f"/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_diffuse_skyh5_Mar2022/1061316296_optimal_Model_{stokes_name}.png",
            title = f"Stokes {stokes_name}"
        )



if __name__ == "__main__":
    plot_skyh5_test_Mar23()
