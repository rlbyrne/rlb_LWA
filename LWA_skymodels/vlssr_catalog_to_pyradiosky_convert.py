import numpy as np
import pyradiosky
import astropy.units as units
from astropy.units import Quantity
from astropy.coordinates import Latitude, Longitude
import matplotlib.pyplot as plt


# Catalog is FullVLSSCatalog.text from https://www.cv.nrao.edu/vlss/CATALOG/
# Reference: Lane et al. 2014, DOI:10.1093/mnras/stu256

catalog_savepath = "/Users/ruby/Astro/FullVLSSCatalog.skyh5"
plot_catalog = False

catalogfile = open("/Users/ruby/Astro/FullVLSSCatalog.text")
catalogdata = catalogfile.readlines()
catalogfile.close()

start_line = 17
cat_RA_h = []
cat_RA_m = []
cat_RA_s = []
cat_Dec_d = []
cat_Dec_m = []
cat_Dec_s = []
flux = []

line = start_line
while line < len(catalogdata) - 1:
    if catalogdata[line][0:6].strip() == "NVSS":  # Remove page breaks
        line += 3
    line_data = catalogdata[line]
    cat_RA_h.append(line_data[0:2])
    cat_RA_m.append(line_data[2:5])
    cat_RA_s.append(line_data[5:11])
    cat_Dec_d.append(line_data[11:15])
    cat_Dec_m.append(line_data[15:18])
    cat_Dec_s.append(line_data[18:23])
    flux.append(line_data[29:36])
    line += 2

cat_RA_h = np.array([int(val) for val in cat_RA_h])
cat_RA_m = np.array([int(val) for val in cat_RA_m])
cat_RA_s = np.array([float(val) for val in cat_RA_s])
cat_Dec_d_int = np.array([int(val) for val in cat_Dec_d])
cat_Dec_m = np.array([int(val) for val in cat_Dec_m])
cat_Dec_s = np.array([float(val) for val in cat_Dec_s])
cat_RA = cat_RA_h + cat_RA_m / 60.0 + cat_RA_s / (60.0**2.0)
cat_RA = cat_RA * 15.0  # Convert to degrees
cat_Dec = np.abs(cat_Dec_d_int) + cat_Dec_m / 60.0 + cat_Dec_s / (60.0**2.0)
for dec_ind in range(len(cat_Dec)):
    if cat_Dec_d[dec_ind][1:2] == "-":
        cat_Dec[dec_ind] *= -1
flux = np.array([float(val) for val in flux])

if plot_catalog:
    ra_range = None
    dec_range = None
    label_sources = True
    label_source_names = None
    ra_cut_val = 0.0

    source_fluxes = list(flux)
    flux_plot_max = max(source_fluxes)
    source_markersizes = []
    markersize_range = [1.0, 500.0]
    for use_flux in source_fluxes:
        if use_flux >= flux_plot_max:
            use_flux = flux_plot_max
        source_markersizes.append(
            use_flux / flux_plot_max * (markersize_range[1] - markersize_range[0])
            + markersize_range[0]
        )

    source_ras = cat_RA
    source_decs = cat_Dec

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
    )

    if label_sources:
        if label_source_names is None:
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

    plt.xlim(ra_range[1], ra_range[0])
    plt.ylim(dec_range[0], dec_range[1])
    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    plt.show()

nsources = len(cat_RA)
nfreqs = 1
cat_stokes = Quantity(np.zeros((4, nfreqs, nsources), dtype=float), "Jy")
cat_stokes[0, 0, :] = flux * units.Jy
cat_spectral_index = np.full(nsources, -0.8)
cat_name = [
    f"VLSSr_sourcecatalog_{str(source_ind+1).zfill(6)}"
    for source_ind in range(nsources)
]

vlssr_catalog = pyradiosky.SkyModel(
    name=cat_name,
    ra=Longitude(cat_RA, units.deg),
    dec=Latitude(cat_Dec, units.deg),
    stokes=cat_stokes,
    spectral_type="spectral_index",
    reference_frequency=Quantity(np.full(nsources, 73.8 * 1e6), "hertz"),
    spectral_index=cat_spectral_index,
)
if not vlssr_catalog.check():
    print("ERROR: Catalog check failed.")
    sys.exit(1)

vlssr_catalog.write_skyh5(catalog_savepath)
