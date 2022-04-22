from casacore import tables
import pyradiosky
import pyuvdata
import numpy as np


def catalog_ms_to_pyradiosky(cat_ms_path):
# Unfinished function
# Unclear how to get absolute fluxes and positions of sources from the ms

    cattable = tables.table(cat_ms_path)
    fluxcol = cattable.getcol("Flux")
    reffreq = cattable.getcol("Reference_Frequency")
    specindcol = cattable.getcol("Spectral_Parameters")
    cat_name = cattable.getcol("Label")

    nsources = np.shape(fluxcol)[0]
    nfreqs = 1

    cat_stokes = Quantity(np.zeros((4, nfreqs, nsources), dtype=float), "Jy")
    cat_stokes[:, 0, :] = np.reshape(np.transpose(np.real(fluxcol)), (4, 1, nsources)) * units.Jy
    cat_spectral_index = np.squeeze(specindcol)

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
        reference_frequency=Quantity(reffreq, "hertz"),
        spectral_index=cat_spectral_index,
    )


def cal_ms_to_uvcal(
    cal_ms_path,
    data_ms_path=None,  # Path to ms file; defaults to cal_ms_path directory
):

    if data_ms_path is None:
        cal_ms_path_split = cal_ms_path.split(".")
        data_ms_path = ".".join(cal_ms_path_split[:-1]) + ".ms"

    uvd = pyuvdata.UVData()
    uvd.read_ms(data_ms_path)
    uvd.instrument = "OVRO-LWA"
    uvd.telescope_name = "OVRO-LWA"
    uvd.set_telescope_params()
    uvd.x_orientation = "east"  # Assumption, need to check

    caltable = tables.table(cal_ms_path)
    flagcol = caltable.getcol("FLAG")
    antcol = caltable.getcol("ANTENNA1")
    refant = caltable.getcol("ANTENNA2")[0]
    gains = caltable.getcol("CPARAM")

    # Get antenna names
    anttable = tables.table(f"{data_ms_path}/ANTENNA")
    antnames = anttable.getcol("NAME")
    refant_name = antnames[refant]
    antnames = np.array([antnames[antind] for antind in antcol])
    ant_array = np.array(
        [np.where(np.array(uvd.antenna_names) == name)[0][0] for name in antnames]
    )

    uvcal = pyuvdata.UVCal()
    uvcal.telescope_name = "OVRO-LWA"
    uvcal.set_telescope_params()
    uvcal.cal_style = "sky"
    uvcal.gain_convention = "multiply"
    uvcal.Nspws = 1
    uvcal.ref_antenna_name = refant_name
    uvcal.Nants_data = np.shape(gains)[0]
    uvcal.Nants_telescope = np.shape(gains)[0]
    uvcal.Nfreqs = np.shape(gains)[1]
    uvcal.Njones = np.shape(gains)[2]
    uvcal.Ntimes = 1
    uvcal.ant_array = ant_array
    uvcal.antenna_names = uvd.antenna_names
    uvcal.antenna_numbers = uvd.antenna_numbers
    uvcal.channel_width = uvd.channel_width
    uvcal.freq_array = uvd.freq_array
    uvcal.integration_time = np.mean(uvd.integration_time) * uvd.Ntimes
    uvcal.time_array = np.array([np.mean(uvd.time_array)])
    uvcal.antenna_positions = uvd.antenna_positions
    uvcal.gain_array = np.reshape(
        gains, (uvcal.Nants_data, uvcal.Nspws, uvcal.Nfreqs, uvcal.Ntimes, uvcal.Njones)
    )
    uvcal.flag_array = np.reshape(
        flagcol,
        (uvcal.Nants_data, uvcal.Nspws, uvcal.Nfreqs, uvcal.Ntimes, uvcal.Njones),
    )

    # Spoof these required parameters
    uvcal.spw_array = np.array([0])
    uvcal.sky_catalog = "cyg_cas_simple"
    uvcal.sky_field = "Cyg-Cas"
    uvcal.history = f"From {data_ms_path}"
    uvcal.jones_array = np.array([-5, -6])
    uvcal.x_orientation = "east"
    uvcal.quality_array = np.ones(
        (uvcal.Nants_data, uvcal.Nspws, uvcal.Nfreqs, uvcal.Ntimes, uvcal.Njones),
        dtype=float,
    )

    uvcal.check()

    return uvcal


if __name__ == "__main__":
    uvcal = cal_ms_to_uvcal(
        "/Users/ruby/Astro/LWA_data/LWA_data_20220307/20220307_175923_61MHz.bcal"
    )
