NODES=(01 01 02 02 03 03 04 04 05 05 06 06 07 07 08 08)
FREQS=(15 20 24 29 34 38 43 47 52 57 61 66 70 75 80 84)
for TIMESTAMP in 000008 000018 000028 000038 000048 000058 000108 000118 000128 000138 000148 000158
do
  for i in {0..15}
  do
    scp -r calim${NODES[i]}:/data${NODES[i]}/slow/20220812_${TIMESTAMP}_${FREQS[i]}MHz.ms /Users/ruby/Astro/lwa_data_20220812/
    pid=$!
    wait $pid
    scp -r /Users/ruby/Astro/lwa_data_20220812/20220812_${TIMESTAMP}_${FREQS[i]}MHz.ms rbyrne@wario.caltech.edu:/safepool/rbyrne/lwa_data/
    pid=$!
    wait $pid
    rm -r /Users/ruby/Astro/lwa_data_20220812/20220812_${TIMESTAMP}_${FREQS[i]}MHz.ms
    pid=$!
    wait $pid
  done
done
