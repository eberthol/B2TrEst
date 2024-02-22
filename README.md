# B2TrEst

This package provides a way to estimate the number of hits deposited by a charged track passing through the Central Drift Chamber (CDC) of the Belle II detector.

From the 3-momenta, charged, and production vertex of a given track, the software computes the trajectory of the particle in the magnetic field of the detector. Then, using a simplified version of the CDC, it evaluates the number of CDC hits aossiciated to this track.

It is possible to configure the CDC model. [IMPROVE]

# Installation

This package uses comon python librairies suchs as numpy, pandas, scipy and matplotlib.

MAKE THIS CLEAR! AND ADD LINKS
If you want to use the script with rootfiles, you also need a way to read these files into pandas dataframes. For that you can use uproot [link] or ROOT.

# Usage

The file ``track_propagation_in_CDC.py`` contains all the classes and methods that are used to compute the trajectory of charged particles and the number of hits in the CDC.



- expected format of input [dict.] + CDC model
- outputs an obect -> can get NCDChits from there

- can also plot some stuff

# Quick start (notebook)

```simple_example.ipynb```

# Recommendations for running on a large number of tracks

- can be slow -> better to run on small number of events at a time (a thousand entries)
- [maybe provide some basic script]
