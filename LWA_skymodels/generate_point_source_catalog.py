import numpy as np
import pyradiosky
from astropy.units import Quantity
from astropy.coordinates import Latitude, Longitude
from Gasperin2020_source_models_to_skyh5 import concatenate_catalogs


def create_point_source_catalog(input_catalog):

    sources = [
        source_name
        for source_name in list(set(input_catalog.extended_model_group))
        if len(source_name) > 0
    ]
    catalog_list = []
    for source_name in sources:
        source_inds = np.where(input_catalog.extended_model_group == source_name)[0]
        if len(source_inds) > 0:
            component_flux = input_catalog.stokes[0, 0, source_inds]
            weighted_mean_ra = np.sum(
                component_flux.value * input_catalog.ra[source_inds]
            ) / np.sum(component_flux.value)
            weighted_mean_dec = np.sum(
                component_flux.value * input_catalog.dec[source_inds]
            ) / np.sum(component_flux.value)
            weighted_mean_spectral_index = np.sum(
                component_flux.value * input_catalog.spectral_index[source_inds]
            ) / np.sum(component_flux.value)
            catalog = pyradiosky.SkyModel(
                name=input_catalog.name[source_inds[0]],
                extended_model_group=source_name,
                ra=Longitude(weighted_mean_ra),
                dec=Latitude(weighted_mean_dec),
                stokes=np.sum(input_catalog.stokes[:, :, source_inds], axis=2),
                spectral_type="spectral_index",
                reference_frequency=np.mean(
                    input_catalog.reference_frequency[source_inds]
                ),
                spectral_index=np.array([weighted_mean_spectral_index]),
                frame="icrs",
            )
            catalog_list.append(catalog)
    output_catalog = concatenate_catalogs(catalog_list)
    return output_catalog


if __name__ == "__main__":
    input_catalog = pyradiosky.SkyModel()
    input_catalog.read("/fast/rbyrne/skymodels/Gasperin2020_sources_plus_48MHz.skyh5")
    catalog = create_point_source_catalog(input_catalog)
    catalog.write_skyh5(
        "/fast/rbyrne/skymodels/Gasperin2020_point_sources_plus_48MHz.skyh5",
        clobber=True,
    )
