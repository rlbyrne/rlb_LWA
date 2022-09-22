import numpy as np
import aoflagger


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
        print(np.shape(flags))
        flagvalues = flags.get_buffer()
        ratio = float(sum(sum(flagvalues))) / (nch*ntimes)
        ratiosum += ratio
        ratiosumsq += ratio*ratio

    print("Percentage flags (false-positive rate) on Gaussian data: " +
        str(ratiosum * 100.0 / count) + "% +/- " +
        str(np.sqrt(
            (ratiosumsq/count - ratiosum*ratiosum / (count*count) )
            ) * 100.0) )

if __name__=="__main__":
    tutorial()
