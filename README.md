# ECE270 Final Project

To run on CSIF Machines, clone repository and from within the NeuronGPU directory:

```
./configure --with-mpi=no --prefix=/home/USERMAME/install
make
make install
python3 python/Potjans_2014/run_microcircuit.py
```

Reconfigure if you have issues, sometimes it has a bug with neurongpulid.la can't figure out why.

Note that the overall runtime is slow, a lot occurs to build and initialize the network, but the simulation time is what matters.

With larger GroupResolutions, the warning `Maximum number of spikes in spike buffer exceeded for spike buffer` may be observed.
This is because certain neurons start to fire at too high a rate as the simulation degrades in accuracy.