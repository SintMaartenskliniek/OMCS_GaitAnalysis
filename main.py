# -*- coding: utf-8 -*-
"""
Main script to read and analyse optical motion capture data (VICON based)
Possibility to include analog (forceplate) data

Version - Author:
    17-05-2023: C.J. Ensink - c.ensink@maartenskliniek.nl
"""

# Import dependencies
from readmarkerdata import readmarkerdata
from gaiteventdetection import gaiteventdetection
from gaitcharacteristics import spatiotemporals, propulsion

# Example GRAIL (treadmill) trial
# Set datapath
datapath = 'data/exampleGRAIL.c3d'
markerdata, fs_markerdata, analogdata, fs_analogdata = readmarkerdata(datapath, analogdata=True)
gaitevents = gaiteventdetection(markerdata, fs_markerdata, algorithmtype='velocity', trialtype='treadmill', debugplot=True)
spatiotemporals = spatiotemporals(markerdata, gaitevents)
# In case propulsion needs to be bodyweight normalized, provide bodyweight as keyword argument
bodyweight = 67 # In kg
trial = datapath # For title above debugplot
gaitevents, spatiotemporals, analogdata = propulsion(gaitevents, spatiotemporals, analogdata, bodyweight=bodyweight, debugplot=True, plot_title=trial)

# Example overground trial
datapath = 'data/exampleOverground.c3d'
markerdata, fs_markerdata, analogdata, fs_analogdata = readmarkerdata(datapath, analogdata=True)
gait_events = gaiteventdetection(markerdata, fs_markerdata, algorithmtype='velocity', trialtype='overground', debugplot=True)
spatiotemporals = spatiotemporals(markerdata, gait_events, trialtype ='overground')

