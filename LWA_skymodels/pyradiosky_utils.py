import numpy as np
import pyradiosky
import sys
import matplotlib.pyplot as plt
from astropy.units import Quantity


def flux_select(skymodel, min_flux=None, max_flux=None, inplace=True):

    if min_flux is None:
        use_inds = np.arange(skymodel.Ncomponents)
    else:
        use_inds = np.where(skymodel.stokes[0, :, :].value >= min_flux)[1]
        use_inds = np.unique(use_inds)

    if max_flux is not None:
        use_inds_max_cut = np.where(skymodel.stokes[0, :, :].value <= max_flux)[1]
        use_inds = np.intersect1d(use_inds, use_inds_max_cut)

    if inplace:
        skymodel.select(component_inds=use_inds, inplace=True)
    else:
        skymodel_new = skymodel.select(component_inds=use_inds, inplace=False)
        return skymodel_new


def get_named_sources():

    named_sources = {
        "Crab": {"ra": 83.6331, "dec": 22.0145},
        "PicA": {"ra": 79.9572, "dec": -45.7788},
        "HydA": {"ra": 139.524, "dec": -12.0956},
        "CenA": {"ra": 201.365, "dec": -43.0192},
        "HerA": {"ra": 252.784, "dec": 4.9925},
        "VirA": {"ra": 187.706, "dec": 12.3911},
        "CygA": {"ra": 299.868, "dec": 40.7339},
        "CasA": {"ra": 350.858, "dec": 58.8},
        "3C161": {"ra": 96.7921, "dec": -5.88472},
        "3C353": {"ra": 260.117, "dec": -0.979722},
        "3C409": {"ra": 303.615, "dec": 23.5814},
        "3C444": {"ra": 333.607, "dec": -17.0267},
        "ForA": {"ra": 50.6738, "dec": -37.2083},
        "HerA": {"ra": 252.793, "dec": 4.99806},
        "NGC0253": {"ra": 11.8991, "dec": -25.2886},
        "PicA": {"ra": 79.9541, "dec": -45.7649},
        "VirA": {"ra": 187.706, "dec": 12.3786},
        "PKS0349-27": {"ra": 57.8988, "dec": -27.7431},
        "PKS0442-28": {"ra": 71.1571, "dec": -28.1653},
        "PKS2153-69": {"ra": 329.275, "dec": -69.6900},
        "PKS2331-41": {"ra": 353.609, "dec": -41.4233},
        "PKS2356-61": {"ra": 359.768, "dec": -60.9164},
        "PKSJ0130-2610": {"ra": 22.6158, "dec": -26.1656},
    }

    return named_sources


def plot_skymodel(
    skymodel,
    ra_range=None,
    dec_range=None,
    min_flux=None,
    label_sources=True,
    ra_cut_val=0.0,
    pol="I",
    save_filename=None,
):

    pol_ind = np.where(np.array(["I", "Q", "U", "V"]) == pol)[0]
    if len(pol_ind) == 0:
        print("ERROR: Unknown pol. Exiting.")
        sys.exit(1)

    if min_flux is not None:
        use_skymodel = flux_select(skymodel, min_flux=min_flux, inplace=False)
    else:
        use_skymodel = skymodel

    source_fluxes = use_skymodel.stokes[pol_ind, :, :].squeeze()
    flux_plot_max = np.max(source_fluxes)
    source_markersizes = []
    markersize_range = [1.0, 500.0]
    for use_flux in source_fluxes:
        if use_flux >= flux_plot_max:
            use_flux = flux_plot_max
        source_markersizes.append(
            use_flux / flux_plot_max * (markersize_range[1] - markersize_range[0])
            + markersize_range[0]
        )

    source_ras = use_skymodel.lon
    source_decs = use_skymodel.lat

    if ra_range is None:
        ra_min = min(source_ras)
        ra_max = max(source_ras)
        ra_range = [
            ra_min - (ra_max - ra_min) / 10.0,
            ra_max + (ra_max - ra_min) / 10.0,
        ]
    if dec_range is None:
        dec_min = min(source_decs)
        dec_max = max(source_decs)
        dec_range = [
            dec_min - (dec_max - dec_min) / 10.0,
            dec_max + (dec_max - dec_min) / 10.0,
        ]

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.scatter(
        source_ras,
        source_decs,
        s=source_markersizes,
        facecolors="blue",
        edgecolors="none",
        alpha=0.3,
    )

    if label_sources:
        named_sources = get_named_sources()
        label_source_names = list(named_sources.keys())
        named_source_ras = [named_sources[name]["ra"] for name in label_source_names]
        named_source_decs = [named_sources[name]["dec"] for name in label_source_names]
        plt.scatter(
            named_source_ras,
            named_source_decs,
            marker="o",
            s=markersize_range[1],
            facecolors="none",
            edgecolors="red",
            linewidth=1,
        )
        for i, name in enumerate(label_source_names):
            plt.annotate(
                name, (named_source_ras[i], named_source_decs[i]), fontsize=10.0
            )

    plt.xlim(ra_range[1].value, ra_range[0].value)
    plt.ylim(dec_range[0].value, dec_range[1].value)
    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")

    if save_filename is None:
        plt.show()
    else:
        plt.savefig(save_filename, dpi=200)
    plt.close()
