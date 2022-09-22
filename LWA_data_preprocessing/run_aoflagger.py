import numpy as np
import aoflagger
import pyuvdata


def tutorial():

    nch = 256
    ntimes = 1000
    count = 50       # number of trials in the false-positives test

    flagger = aoflagger.AOFlagger()
    path = flagger.find_strategy_file(aoflagger.TelescopeId.Generic)
    strategy = flagger.load_strategy_file(path)
    data = flagger.make_image_set(ntimes, nch, 8)

    ratiosum = 0.0
    ratiosumsq = 0.0
    for repeat in range(count):
        for imgindex in range(8):
            # Initialize data with random numbers
            values = np.random.normal(0, 1, [nch, ntimes])
            data.set_image_buffer(imgindex, values)

        flags = strategy.run(data)
        flagvalues = flags.get_buffer()
        ratio = float(sum(sum(flagvalues))) / (nch*ntimes)
        ratiosum += ratio
        ratiosumsq += ratio*ratio

    print("Percentage flags (false-positive rate) on Gaussian data: " +
        str(ratiosum * 100.0 / count) + "% +/- " +
        str(np.sqrt(
            (ratiosumsq/count - ratiosum*ratiosum / (count*count) )
            ) * 100.0) )


def flag_ms_file():

    ms_file = "/home/rbyrne/20220812_000158_84MHz.ms"
    uvd = pyuvdata.UVData()
    uvd.read_ms(ms_file, data_column="DATA")
    uvd.instrument = "OVRO-LWA"
    uvd.telescope_name = "OVRO-LWA"
    uvd.set_telescope_params()
    uvd.check()

    flagger = aoflagger.AOFlagger()
    path = flagger.find_strategy_file(aoflagger.TelescopeId.Generic)
    strategy = flagger.load_strategy_file(path)
    flag_data = flagger.make_image_set(uvd.Ntimes, uvd.Nfreqs, uvd.Nbls*uvd.Npols)

    bl_pol_ind = 0
    for pol in uvd.polarization_array():
        for bl in np.unique(uvd.baseline_array):
            blt_inds = np.where(uvd.baseline_array == bl)[0]
            uvd_copy = uvd.select(polarizations=pol, blt_inds=blt_inds, inplace=False)
            uvd_copy.reorder_freqs(channel_order="freq")
            flag_data.set_image_buffer(bl_pol_ind, uvd_copy.data_array)
            bl_pol_ind += 1

    flags = strategy.run(flag_data)
    flagvalues = flags.get_buffer()
    print(np.shape(flagvalues))
    print(np.sum(flagvalues))


if __name__=="__main__":
    flag_ms_file()
