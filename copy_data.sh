for time in 08 09 10 11 12
do
    mkdir /lustre/rbyrne/2024-03-02/${time}
    for freq in 18 23 27 36 41 46 50 55 59 64 73 78 82
    do
        cp -r /lustre/gh/2024-03-02/${time}/${freq}.ms /lustre/rbyrne/2024-03-02/${time}
    done
done