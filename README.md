# OMCS based gait analysis
Gait analysis based on optical motion capture, including the option for forceplate data analysis.


**Main script:**
- *main.py*


**Required scripts:**
- *readmarkerdata.py*: imports markerdata from  c3d file and if present also forceplate data
- *gaiteventdetection.py*: calculates gait events (initial contact, terminal contact)
- *gaitcharacteristics.py*: calculates gait characteristics (gait speed, gait cycle duration, stance time, swing time, stride length, AP-propulsion)


**Example data:**
- *exampleGRAIL.c3d*: example data of walking on the GRIAL (treadmill including forceplates)
- *exampleOvergorund.c3d*: example data of walking overground
