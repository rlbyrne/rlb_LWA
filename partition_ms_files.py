# Run in CASA with command execfile("partition_ms_files.py")

pathname = "/lustre/rbyrne/2024-03-02/10"
for filename in ["55", "59", "64", "73", "78", "82"]:
    partition(
        f"{pathname}/{filename}.ms",
        outputvis=f"{pathname}/{filename}_10_001.ms",
        scan="1~5",
    )

datafile_list = [
    f"{pathname}/{filename}_10.001.ms"
    for filename in ["41", "46", "50", "55", "59", "64", "73", "78", "82"]
]
concat(vis=datafile_list, concatvis=f"{pathname}/41-82_10_001.ms")
