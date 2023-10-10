"""

File containing two functions to calculate gait characteristics from marker and force plate data.
    1) def spatiotemporals(markerdata, sample_frequency)
    2) def propulsion(propulsion(gaitevents, spatiotemporals, analogdata, bodyweight)

                      
1) def spatiotemporals(markerdata, sample_frequency)
Function to calculate spatiotemporal files from marker data.
Can be applied to both treadmill and overground data.

INPUT
spatiotemporals(markerdata, sample_frequency)
    markerdata:        dictionary of labeled marker data 
    gaitevents:        dictionary of gait events

OPTIONAL INPUT
spatiotemporals(**kwargs)
    sample_frequency:  sample frequency of marker data; defaults to 100
    trialtype:         string; can be 'treadmill' or 'overground'; defaults to 'treadmill'
    debugplot:         bool; can be 'True' or 'False'; defaults to 'False'

OUTPUT
    spatiotemporals:   dictionary with spatiotemporal parameters



2) def propulsion(propulsion(gaitevents, spatiotemporals, analogdata, bodyweight)
Function to calculate generated forward propulsion from force plate (analog) data.

INPUT
propulsion(gaitevents, spatiotemporals, analogdata, bodyweight)
    gaitevents:        dictionary of gait events
    spatiotemporals:   dictionary with spatiotemporal parameters
    analogdata:        dictionary with analog (force plate) data

OPTIONAL INPUT
propulsion(**kwargs)
    bodyweight:        bodyweight (kg) of the person performing the walking trial; defaults to 1 kg
    debugplot:         bool; can be 'True' or 'False'; defaults to 'False'
    plot_title:        string; title of the debug plot
    fs_analogdata:     integer; defaults to 1000 Hz
    fs_markerdata:     integer; defaults to 100 Hz
    
OUTPUT
    gaitevents:        dictionary; added index numbers of start and stop of each propulsion and braking impulse, and index numbers of peak propulsion and braking instances
    spatiotemporals:   dictionary; added generated forward propulsion for each propulsion and braking impulse, and peak propulsion and braking values
    analogdata:        dictionary; added filtered and resampled analog data (2nd order, lowpass Butterworth filter, resampled to 100 Hz)

Copyright (c):
    2023, Carmen Ensink, Sint Maartenskliniek,
    c.ensink@maartenskliniek.nl

    
"""

# Import dependencies
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def spatiotemporals(markerdata, gaitevents, **kwargs):
    
    # Set defaults
    sample_frequency = 100
    trialtype = 'treadmill' 
    debugplot = 'False'
    th_artefact_time = 1.8 # In seconds; assume artefact if swing/stance time duration is >1.8 seconds
    th_artefact_length_min = 200 # In mm; assume artefact if stridelenght <200mm
    th_artefact_length_max = 1800 # In mm; assume artefact if stridelenght >1800mm
    spatiotemporals = dict()
    
    # Check optional input arguments
    for key, value in kwargs.items():
        if key == 'sample_frequency':
            sample_frequency = value
        if key == 'trialtype':
            trialtype = value
        if key == 'debugplot':
            debugplot = value
    
    # Gait events
    ICL = gaitevents['Index numbers initial contact left']
    ICR = gaitevents['Index numbers initial contact right']
    TCL = gaitevents['Index numbers terminal contact left']
    TCR = gaitevents['Index numbers terminal contact right']
    
    
    # Filter ankle marker data
    # Set second-order low-pass butterworth filter;
    N = 2 # Order of the butterworth filter
    filter_type = 'lowpass' # Type of the filter
    fc1 = 5  # Cut-off frequency of the first low pass filter
    wn1 = fc1 / (sample_frequency / 2) # Normalize the frequency
    B1, A1 = signal.butter(N, wn1, filter_type) # First low-pass filterdesign
    
    vlank = dict()
    vrank = dict()
    # Apply low pass filter on data and calculate velocity
    vlank['vlankx'] = signal.filtfilt(B1, A1, markerdata['LANK'][:,0])
    vlank['vlanky'] = signal.filtfilt(B1, A1, markerdata['LANK'][:,1])
    vlank['vlankz'] = signal.filtfilt(B1, A1, markerdata['LANK'][:,2])
    
    vrank['vrankx'] = signal.filtfilt(B1, A1, markerdata['RANK'][:,0])
    vrank['vranky'] = signal.filtfilt(B1, A1, markerdata['RANK'][:,1])
    vrank['vrankz'] = signal.filtfilt(B1, A1, markerdata['RANK'][:,2])
    
    
    
    
    
    match trialtype:
        case 'treadmill':
            
            # Stance time = time from initial contact of one foot till terminal contact of the same foot
            # left side
            StTL = np.array([])
            for i in range(0,len(ICL)):
                firstTC = np.argwhere(TCL > ICL[i])
                if len(firstTC) > 0:
                    firstTC = np.argwhere(TCL > ICL[i])[0]
                    if TCL[firstTC] - ICL[i] > th_artefact_time*sample_frequency:
                        StTL = np.append(StTL, np.nan)
                    else:
                        StTL = np.append(StTL, TCL[firstTC] - ICL[i])
            # Right side
            StTR = np.array([])
            for i in range(0,len(ICR)):
                firstTC = np.argwhere(TCR > ICR[i])
                if len(firstTC) > 0:
                    firstTC = np.argwhere(TCR > ICR[i])[0]
                    if TCR[firstTC] - ICR[i] > th_artefact_time*sample_frequency:
                        StTR = np.append(StTR, np.nan)
                    else:
                        StTR = np.append(StTR, TCR[firstTC] - ICR[i])
            
            
            # Swing time = time from terminal contact of one foot till initial contact of the same foot
            # left side
            SwTL = np.array([])
            for i in range(0,len(TCL)):
                firstIC = np.argwhere(ICL > TCL[i])
                if len(firstIC) > 0:
                    firstIC = np.argwhere(ICL > TCL[i])[0]
                    if ICL[firstIC] - TCL[i] > th_artefact_time*sample_frequency:
                        SwTL = np.append(SwTL, np.nan)
                    else:
                        SwTL = np.append(SwTL, ICL[firstIC] - TCL[i])
            # Right side
            SwTR = np.array([])
            for i in range(0,len(TCR)):
                firstIC = np.argwhere(ICR > TCR[i])
                if len(firstIC) > 0:
                    firstIC = np.argwhere(ICR > TCR[i])[0]
                    if ICR[firstIC] - TCR[i] > th_artefact_time*sample_frequency:
                        SwTR = np.append(SwTR, np.nan)
                    else:
                        SwTR = np.append(SwTR, ICR[firstIC] - TCR[i])
            
            
            # Velocity profile over the full trial for all flat-foot phases
            # Time interval for foot flat phase
            valsleft = np.array(range(np.round(0.30*np.nanmean(StTL)).astype(int) , np.round(0.50*np.nanmean(StTL)).astype(int)))
            valsright = np.array(range(np.round(0.30*np.nanmean(StTR)).astype(int) , np.round(0.50*np.nanmean(StTR)).astype(int)))
            # velocity = differentiated markerdatavicon['ANK'], at the assumed foot flat phase.
            # this is assumed to be the velocity of the treadmill and per definition the walkingspeed
            velocity_left = np.zeros(len(markerdata['LANK']))*np.nan
            for i in range(0, len(ICL)-1):
                velocity_left[ICL[i] + valsleft[0:-1]] = np.diff(markerdata['LANK'][ICL[i]+valsleft,1])*sample_frequency
            velocity_right = np.zeros(len(markerdata['RANK']))*np.nan
            for i in range(0, len(ICR)-1):
                velocity_right[ICR[i] + valsright[0:-1]] = np.diff(markerdata['RANK'][ICR[i]+valsright,1])*sample_frequency
            # Convert mm/s to m/s
            velocity_left = velocity_left/1000
            velocity_right = velocity_right/1000
            
            
            # Filter velocity of ankle marker data
            # Set second-order low-pass butterworth filter;
            fc2 = 8 # Cut-off frequency of the second low pass filter
            wn2 = fc2 / (sample_frequency / 2) # Normalize the frequency
            B2, A2 = signal.butter(N, wn2, filter_type) # Second low-pass filterdesign
            # Left side
            vlank['vlankx2'] = signal.filtfilt(B2, A2, np.diff(vlank['vlankx']))*sample_frequency
            vlank['vlanky2'] = signal.filtfilt(B2, A2, np.diff(vlank['vlanky']))*sample_frequency
            vlank['vlankz2'] = signal.filtfilt(B2, A2, np.diff(vlank['vlankz']))*sample_frequency
            # Right side
            vrank['vrankx2'] = signal.filtfilt(B2, A2, np.diff(vrank['vrankx']))*sample_frequency
            vrank['vranky2'] = signal.filtfilt(B2, A2, np.diff(vrank['vranky']))*sample_frequency
            vrank['vrankz2'] = signal.filtfilt(B2, A2, np.diff(vrank['vrankz']))*sample_frequency
            # Back to one matrix
            vlank['vlank'] = np.swapaxes(np.vstack((vlank['vlankx2'],vlank['vlanky2'], vlank['vlankz2'])), 1, 0)
            vrank['vrank'] = np.swapaxes(np.vstack((vrank['vrankx2'],vrank['vranky2'], vrank['vrankz2'])), 1, 0)
            vlank = vlank['vlank']
            vrank = vrank['vrank']
            
            # Stride length
            # Position difference between two heel strikes of one foot + time difference*velocity of the treadmill
            # Left side
            stridelengths_left = np.zeros((len(ICL),3))*np.nan
            treadmill_left = np.zeros(len(ICL))*np.nan
            foot_left = np.zeros((len(ICL),3))*np.nan
            for i in range(1, len(ICL)):
                start_stride = ICL[ICL<ICL[i]][-1]
                stop_stride = ICL[i]
                duration_stride = stop_stride-start_stride
                
                start_swing = TCL[TCL<ICL[i]][-1]
                duration_swing = stop_stride-start_swing
                start_ff = int(start_swing+0.1*duration_swing)
                stop_ff = int(start_swing+0.6*duration_swing)
                # duration_ff = stop_ff-start_ff
                treadmill_left[i] = (duration_stride/sample_frequency) * np.nanmean(vrank[start_ff : stop_ff,1]) # assumed during left swing the right foot is at the treadmill
                
                foot_left[i,:] = markerdata['LHEE'][stop_stride] - markerdata['LHEE'][start_stride]  
                stridelengths_left[i,0] = start_swing 
                stridelengths_left[i,1] = stop_stride 
                stridelengths_left[i,2] = treadmill_left[i] - foot_left[i,1]
            
            # Artefact rejection routine; if stridelength deviates >60% of median stridelength or is >th_artefact_length cm or has a presumed swing time of >th_artefact_time seconds
            mslL = np.nanmedian(stridelengths_left[:,2])
            for i in range(0, len(stridelengths_left)):
                if stridelengths_left[i,2] > 1.6*mslL or stridelengths_left[i,2] < th_artefact_length_min or stridelengths_left[i,1]-stridelengths_left[i,0] > th_artefact_time*np.median(SwTL):
                    stridelengths_left[i,:] = np.nan
            stridelengths_left = stridelengths_left[~np.isnan(stridelengths_left).any(axis=1), :]
            
            # Right side
            stridelengths_right = np.zeros((len(ICR),3))*np.nan
            treadmill_right = np.zeros(len(ICR))*np.nan
            foot_right = np.zeros((len(ICR),3))*np.nan
            for i in range(1, len(ICR)):
                start_stride = ICR[ICR<ICR[i]][-1]
                stop_stride = ICR[i]
                duration_stride = stop_stride-start_stride
                
                start_swing = TCR[TCR<ICR[i]][-1]
                duration_swing = stop_stride-start_swing
                start_ff = int(start_swing+0.1*duration_swing)
                stop_ff = int(start_swing+0.6*duration_swing)
                # duration_ff = stop_ff-start_ff
                treadmill_right[i] = (duration_stride/sample_frequency) * np.nanmean(vlank[start_ff : stop_ff,1]) # assumed during left swing the right foot is at the treadmill
                
                foot_right[i,:] = markerdata['RHEE'][stop_stride] - markerdata['RHEE'][start_stride]
                stridelengths_right[i,0] = start_swing
                stridelengths_right[i,1] = stop_stride
                stridelengths_right[i,2] = treadmill_right[i] - foot_right[i,1]
            
            # Artefact rejection routine; if stridelength deviates >60% of median stridelength or is >th_artefact_length cm or has a presumed swing time of >th_artefact_time seconds
            mslR = np.nanmedian(stridelengths_right[:,2])
            for i in range(0, len(stridelengths_right)):
                if stridelengths_right[i,2] > 1.6*mslR or stridelengths_right[i,2] < th_artefact_length_min or stridelengths_right[i,1]-stridelengths_right[i,0] > th_artefact_time*np.median(SwTR): 
                    stridelengths_right[i,:] = np.nan
            stridelengths_right = stridelengths_right[~np.isnan(stridelengths_right).any(axis=1), :]
            
            
            # Gait cycle duration = time from heel strike till heel strike of the same foot
            # Left side
            GCDL = np.zeros((len(stridelengths_left),3))*np.nan
            GCDL[:,0] = stridelengths_left[:,0]
            GCDL[:,1] = stridelengths_left[:,1]
            GCDL[1:,2] = np.diff(stridelengths_left[:,1])
            # Artefact rejection routine; deem artefact if gait cycle duration >3 seconds, or gait cycle duration <0.3 seconds, or gait cycle duration deviates >50% of median gait cycle duration
            GCDL[GCDL[:,2]>3*sample_frequency,:] = np.nan
            GCDL[GCDL[:,2]<0.3*sample_frequency,:] = np.nan
            GCDL[GCDL[:,2]>1.5*np.nanmedian(GCDL[:,2]),:] = np.nan
            
            # Right side
            GCDR = np.zeros((len(stridelengths_right),3))*np.nan
            GCDR[:,0] = stridelengths_right[:,0]
            GCDR[:,1] = stridelengths_right[:,1]
            GCDR[1:,2] = np.diff(stridelengths_right[:,1])
            # Artefact rejection routine; deem artefact if gait cycle duration >3 seconds, or gait cycle duration <0.3 seconds, or gait cycle duration deviates >50% of median gait cycle duration
            GCDR[GCDR[:,2]>3*sample_frequency,:] = np.nan
            GCDR[GCDR[:,2]<0.3*sample_frequency,:] = np.nan    
            GCDR[GCDR[:,2]>1.5*np.nanmedian(GCDR[:,2]),:] = np.nan
            
            
            # Step lengths and stepwidths
            # Position difference between feet at heel strike
            steplengths_left = np.zeros((len(ICL),2))*np.nan
            stepwidths_left = np.zeros((len(ICL),2))*np.nan
            foot_left = np.zeros((len(ICL),3))*np.nan
            for i in range(0, len(ICL)):
                foot_left[i,:] = markerdata['LHEE'][ICL[i]] - markerdata['RHEE'][ICL[i]] # forward swing is in opposite direction of treadmill
                steplengths_left[i,0] = ICL[i]
                stepwidths_left[i,0] = ICL[i]
                steplengths_left[i,1] = np.abs(foot_left[i,1])
                stepwidths_left[i,1] = np.abs(foot_left[i,0])
            
            steplengths_right = np.zeros((len(ICR),2))*np.nan
            stepwidths_right = np.zeros((len(ICR),2))*np.nan
            foot_right = np.zeros((len(ICR),3))*np.nan
            for i in range(0, len(ICR)):
                foot_right[i,:] = markerdata['RHEE'][ICR[i]] - markerdata['LHEE'][ICR[i]] # forward swing is in opposite direction of treadmill
                steplengths_right[i,0] = ICR[i]
                stepwidths_right[i,0] = ICR[i]
                steplengths_right[i,1] = np.abs(foot_right[i,1])
                stepwidths_right[i,1] = np.abs(foot_right[i,0])
            
        case 'overground':
            
            # Stride length
            # Position difference between two heel strikes of one foot
            # Left side
            stridelengths_left = np.zeros((len(ICL),4))*np.nan
            foot_left = np.zeros((len(ICL),3))*np.nan
            for i in range(1, len(ICL)):
                start_stride = ICL[ICL<ICL[i]][-1]
                stop_stride = ICL[i]
                duration_stride = stop_stride-start_stride
                start_swing = TCL[TCL<ICL[i]][-1]
                
                foot_left[i,:] = markerdata['LHEE'][stop_stride] - markerdata['LHEE'][start_stride]
                stridelengths_left[i,0] = start_swing
                stridelengths_left[i,1] = stop_stride
                stridelengths_left[i,2] = np.abs(foot_left[i,0])
                stridelengths_left[i,3] = start_stride
            
            uni, idx, counts = np.unique(stridelengths_left[:,3], return_index=True, return_counts=True)
            stridelengths_left = stridelengths_left[idx[counts<2],:]
            uni, idx, counts = np.unique(stridelengths_left[:,0],return_index=True, return_counts=True)
            stridelengths_left = stridelengths_left[idx[counts<2],:]
            
            # Artefact rejection routine; deem artefact if stridelength >th_artefact_length_max, or stridelength<th_artefact_legnth_min
            for i in range(0, len(stridelengths_left)):
                if stridelengths_left[i,2] > th_artefact_length_max or stridelengths_left[i,2] < th_artefact_length_min:
                    stridelengths_left[i,:] = np.nan
            mslL = np.nanmedian(stridelengths_left[:,2])
            for i in range(0, len(stridelengths_left)):
                if stridelengths_left[i,2] > th_artefact_time*mslL:
                    stridelengths_left[i,:] = np.nan
            stridelengths_left = stridelengths_left[~np.isnan(stridelengths_left).any(axis=1), :]
            
            # Right side
            stridelengths_right = np.zeros((len(ICR),4))*np.nan
            foot_right = np.zeros((len(ICR),3))*np.nan
            for i in range(1, len(ICR)):
                start_stride = ICR[ICR<ICR[i]][-1]
                stop_stride = ICR[i]
                duration_stride = stop_stride-start_stride
                start_swing = TCR[TCR<ICR[i]][-1]
                
                foot_right[i,:] = markerdata['RHEE'][stop_stride] - markerdata['RHEE'][start_stride]
                stridelengths_right[i,0] = start_swing
                stridelengths_right[i,1] = stop_stride
                stridelengths_right[i,2] = np.abs(foot_right[i,0])
                stridelengths_right[i,3] = start_stride
            
            uni, idx, counts = np.unique(stridelengths_right[:,3], return_index=True, return_counts=True)
            stridelengths_right = stridelengths_right[idx[counts<2],:]
            uni, idx, counts = np.unique(stridelengths_right[:,0], return_index=True, return_counts=True)
            stridelengths_right = stridelengths_right[idx[counts<2],:]
            
            # Artefact rejection routine; deem artefact if stridelength >th_artefact_length_max, or stridelength<th_artefact_legnth_min
            for i in range(0, len(stridelengths_right)):
                if stridelengths_right[i,2] > th_artefact_length_max or stridelengths_right[i,2] < th_artefact_length_min:
                    stridelengths_right[i,:] = np.nan
            mslR = np.nanmedian(stridelengths_right[:,2])
            for i in range(0, len(stridelengths_right)):
                if stridelengths_right[i,2] > th_artefact_time*mslR:
                    stridelengths_right[i,:] = np.nan
            stridelengths_right = stridelengths_right[~np.isnan(stridelengths_right).any(axis=1), :]
    
            
            # Stance time = time from heel strike of one foot till toe off of the same foot
            StTL = np.zeros((len(stridelengths_left),3))*np.nan
            StTL[:,0] = stridelengths_left[:,0]
            StTL[:,1] = stridelengths_left[:,1]
            for i in range(1, len(stridelengths_left[:,0])):
                StTL[i,2] = stridelengths_left[i,0] - stridelengths_left[i-1,1]
            
            StTR = np.zeros((len(stridelengths_right),3))*np.nan
            StTR[:,0] = stridelengths_right[:,0]
            StTR[:,1] = stridelengths_right[:,1]
            for i in range(1, len(stridelengths_right[:,0])):
                StTR[i,2] = stridelengths_right[i,0] - stridelengths_right[i-1,1]
            
            
            # Swing time = time from toe off of one foot till heel strike of the same foot
            SwTL = np.zeros((len(stridelengths_left),3))*np.nan
            SwTL[:,0] = stridelengths_left[:,0]
            SwTL[:,1] = stridelengths_left[:,1]
            SwTL[:,2] = stridelengths_left[:,1] - stridelengths_left[:,0]
            
            SwTR = np.zeros((len(stridelengths_right),3))*np.nan
            SwTR[:,0] = stridelengths_right[:,0]
            SwTR[:,1] = stridelengths_right[:,1]
            SwTR[:,2] = stridelengths_right[:,1] - stridelengths_right[:,0]
            
            
            # Gait cycle duration = time from heel strike till heel strike of the same foot
            GCDL = np.zeros((len(stridelengths_left),3))*np.nan
            GCDL[:,0] = stridelengths_left[:,3]
            GCDL[:,1] = stridelengths_left[:,1]
            GCDL[:,2] = stridelengths_left[:,1]-stridelengths_left[:,3]
            # Artefact rejection routine; deem artefact if gait cycle duration >3 seconds, or gait cycle duration <0.3 seconds, or gait cycle duration deviates >50% of median gait cycle duration
            GCDL[GCDL[:,2]>3*sample_frequency,:] = np.nan #2.3
            GCDL[GCDL[:,2]<0.3*sample_frequency,:] = np.nan
            GCDL[GCDL[:,2]>1.5*np.nanmedian(GCDL[:,2]),:] = np.nan
            GCDL[GCDL[:,2]==0,:] = np.nan

            GCDR = np.zeros((len(stridelengths_right),3))*np.nan
            GCDR[:,0] = stridelengths_right[:,3]
            GCDR[:,1] = stridelengths_right[:,1]
            GCDR[:,2] = stridelengths_right[:,1]-stridelengths_right[:,3]
            # Artefact rejection routine; deem artefact if gait cycle duration >3 seconds, or gait cycle duration <0.3 seconds, or gait cycle duration deviates >50% of median gait cycle duration
            GCDR[GCDR[:,2]>3*sample_frequency,:] = np.nan #2.3
            GCDR[GCDR[:,2]<0.3*sample_frequency,:] = np.nan
            GCDR[GCDR[:,2]>1.5*np.nanmedian(GCDR[:,2]),:] = np.nan
            GCDR[GCDR[:,2]==0,:] = np.nan
            
            
            # Place filtered ankle marker data back in one matrix
            # Back to one matrix
            vlank['vlank'] = np.swapaxes(np.vstack((vlank['vlankx'],vlank['vlanky'], vlank['vlankz'])), 1, 0)
            vrank['vrank'] = np.swapaxes(np.vstack((vrank['vrankx'],vrank['vranky'], vrank['vrankz'])), 1, 0)
            vlank = vlank['vlank']
            vrank = vrank['vrank']
            
            
            # Velocity profile over the full trial, defined as the differentiated markerdatavicon['ANK']
            velocity_left = np.abs(np.diff(vlank, axis=0))
            velocity_right = np.abs(np.diff(vrank, axis=0))
            # Convert mm/frame to m/s
            velocity_left = velocity_left/1000 * sample_frequency
            velocity_right = velocity_right/1000 * sample_frequency
    
            
            # Step lengths and stepwidths
            # Position difference between heel strikes of one foot and heel strike of other foot
            steplengths_left = np.zeros(len(ICL))*np.nan
            stepwidths_left = np.zeros(len(ICL))*np.nan
            steptime_left = np.zeros(len(ICL))*np.nan
            foot_left = np.zeros((len(ICL),3))*np.nan
            for i in range(0, len(ICL)):
                foot_left[i,:] = np.abs(markerdata['LHEE'][ICL[i]] - markerdata['RHEE'][ICL[i]]) # forward swing is in opposite direction of treadmill
                steplengths_left[i] = foot_left[i,1]
                stepwidths_left[i] = np.abs(foot_left[i,0])
                if i == 0 and ICL[0]<ICR[0]:
                    steptime_left[i] = ICL[i]-TCL[i]
                elif i == 0 and ICL[0]>ICR[0]:
                    steptime_left[i] = ICL[i]-ICR[ICR<ICL[i]][-1]
                if i > 0:
                    if (ICL[i] - ICR[ICR<ICL[i]][-1]) < 2.3*sample_frequency:
                        steptime_left[i] = ICL[i]-ICR[ICR<ICL[i]][-1]
                    else:
                        steptime_left[i] = np.nan
            
            steplengths_right = np.zeros(len(ICR))*np.nan
            stepwidths_right = np.zeros(len(ICR))*np.nan
            steptime_right = np.zeros(len(ICR))*np.nan
            foot_right = np.zeros((len(ICR),3))*np.nan
            for i in range(0, len(ICR)):
                foot_right[i,:] = (-1) * (markerdata['RHEE'][ICR[i]] - markerdata['LHEE'][ICR[i]]) # forward swing is in opposite direction of treadmill
                steplengths_right[i] = foot_right[i,1]
                stepwidths_right[i] = np.abs(foot_right[i,0])
                if i == 0 and ICR[0]<ICL[0]:
                    steptime_right[i] = ICR[i]-TCR[i]
                elif i == 0 and ICR[0]>ICL[0]:
                    steptime_right[i] = ICR[i]-ICL[ICL<ICR[i]][-1]
                if i > 0:
                    if (ICR[i] - ICL[ICL<ICR[i]][-1]) < 2.3*sample_frequency:
                        steptime_right[i] = ICR[i]-ICL[ICL<ICR[i]][-1]
                    else:
                        steptime_right[i] = np.nan
            
    
    # Velocity per stride
    # Left side
    velocity_stridesleft = np.zeros((len(stridelengths_left),3))*np.nan
    velocity_stridesleft[:,0] = stridelengths_left[:,0]
    velocity_stridesleft[:,1] = stridelengths_left[:,1]
    velocity_stridesleft[:,2] = (stridelengths_left[:,2]/1000) / ((GCDL[:,2])/sample_frequency)
    if trialtype == 'overground':
        for i in range(len(GCDL)):
            if np.isnan(GCDL[i,2]) == True:
                velocity_stridesleft[i,2] = np.nan
        stridelengths_left = stridelengths_left[~np.isnan(GCDL).any(axis=1), :]
        velocity_stridesleft = velocity_stridesleft[~np.isnan(GCDL).any(axis=1), :]
        GCDL = GCDL[~np.isnan(GCDL).any(axis=1), :]
    # Right side
    velocity_stridesright = np.zeros((len(stridelengths_right),3))*np.nan
    velocity_stridesright[:,0] = stridelengths_right[:,0]
    velocity_stridesright[:,1] = stridelengths_right[:,1]
    velocity_stridesright[:,2] = (stridelengths_right[:,2]/1000) / ((GCDR[:,2])/sample_frequency)
    if trialtype == 'overground':
        for i in range(len(GCDR)):
            if np.isnan(GCDR[i,2]) == True:
                velocity_stridesright[i,2] = np.nan
        stridelengths_right = stridelengths_right[~np.isnan(GCDR).any(axis=1), :]
        velocity_stridesright = velocity_stridesright[~np.isnan(GCDR).any(axis=1), :]
        GCDR = GCDR[~np.isnan(GCDR).any(axis=1), :]
    
    
    # Fill output dictionary
    spatiotemporals['Velocity left (m/s)'] = velocity_left
    spatiotemporals['Velocity right (m/s)'] = velocity_right
    spatiotemporals['Gait speed left strides (m/s)'] = velocity_stridesleft
    spatiotemporals['Gait speed right strides (m/s)'] = velocity_stridesright
    spatiotemporals['Gait Cycle duration left (s)'] = GCDL
    spatiotemporals['Gait Cycle duration left (s)'][:,2] = spatiotemporals['Gait Cycle duration left (s)'][:,2]/sample_frequency
    spatiotemporals['Gait Cycle duration right (s)'] = GCDR
    spatiotemporals['Gait Cycle duration right (s)'][:,2] = spatiotemporals['Gait Cycle duration right (s)'][:,2]/sample_frequency
    spatiotemporals['Stance time left (s)'] = StTL/sample_frequency
    spatiotemporals['Stance time right (s)'] = StTR/sample_frequency
    spatiotemporals['Swing time left (s)'] = SwTL/sample_frequency
    spatiotemporals['Swing time right (s)'] = SwTR/sample_frequency
    spatiotemporals['Steplength left (mm)'] = steplengths_left
    spatiotemporals['Steplength right (mm)'] = steplengths_right
    spatiotemporals['Stepwidth left (mm)'] = stepwidths_left
    spatiotemporals['Stepwidth right(mm)'] = stepwidths_right
    if trialtype == 'treadmill':
        spatiotemporals['Stridelength left (mm)'] = stridelengths_left
        spatiotemporals['Stridelength right (mm)'] = stridelengths_right
    elif trialtype == 'overground':
        spatiotemporals['Stridelength left (mm)'] = stridelengths_left[:,0:3]
        spatiotemporals['Stridelength right (mm)'] = stridelengths_right[:,0:3]
    
    
    return spatiotemporals





def propulsion(gaitevents, spatiotemporals, analogdata, **kwargs):
    # Function was based on:
    # Deffeyes, J. E., & Peters, D. M. (2021). Time-integrated propulsive and braking impulses do not depend on walking speed. Gait & posture, 88, 258-263.
    # DOI: https://doi.org/10.1016/j.gaitpost.2021.06.012
    
    # Set defaults
    fs_analogdata = 1000 # Sample frequecy of the force plates
    fs_markerdata = 100 # Sample frequency of the marker data
    bodyweight = 1
    th_crossings = 0 # Set threshold_crossings at 0 Newton to identify crossings in force in AP direction
    th_crosssteps = -10 * 0.90 * bodyweight # Set threshold_crosssteps at 10 times 90% of the bodyweight to identify cross steps and deem artefact
    debugplot = False
    title = ' '
    
    # Check optional input arguments
    for key, value in kwargs.items():
        if key == 'fs_analogdata':
            fs_analogdata = value
        if key == 'fs_markerdata':
            fs_markerdata = value
        if key == 'debugplot':
            debugplot = value
        if key == 'plot_title':
            title = value
        if key == 'bodyweight':
            bodyweight = value
    
    
    
    # Filter and resample force plate data
    # Zeni, J. A., Jr, Richards, J. G., & Higginson, J. S. (2008).
    # Two simple methods for determining gait events during treadmill and overground walking using kinematic data.
    # Gait & posture, 27(4), 710–714. https://doi.org/10.1016/j.gaitpost.2007.07.007
    # Fourth-order low-pass butterworth filter;
    # Cut-off frequency: 20Hz
    fc = 20  # Cut-off frequency of the filter
    omega = fc / (fs_analogdata / 2) # Normalize the frequency
    N = 2 # Order of the butterworth filter (scipy.signal.filtfilt is a forwrd-backward linear filter meaning the Nth-order*2 is applied)
    filter_type = 'lowpass' # Type of the filter
    b, a = signal.butter(N, omega, filter_type)
    
    analogdatalabels = list(analogdata.keys())
    for label in analogdatalabels:
        if 'Force.Fy1' in label:
            analogdata['Force Y left filtered'] = signal.filtfilt(b, a, analogdata[label]) # Apply filter
        if 'Force.Fy2' in label:
            analogdata['Force Y right filtered'] = signal.filtfilt(b, a, analogdata[label]) # Apply filter
        if 'Force.Fz1' in label:
            analogdata['Force Z left filtered'] = signal.filtfilt(b, a, analogdata[label]) # Apply filter
        if 'Force.Fz2' in label:
            analogdata['Force Z right filtered'] = signal.filtfilt(b, a, analogdata[label]) # Apply filter

    # Resample force data to 100 Hz (similar to markerdata)
    analogdata['Force Y left filtered'] = signal.resample(analogdata['Force Y left filtered'], int(len(analogdata['Force Y left filtered'])/10))
    analogdata['Force Y right filtered'] = signal.resample(analogdata['Force Y right filtered'], int(len(analogdata['Force Y right filtered'])/10))
    analogdata['Force Z left filtered'] = signal.resample(analogdata['Force Z left filtered'], int(len(analogdata['Force Z left filtered'])/10))
    analogdata['Force Z right filtered'] = signal.resample(analogdata['Force Z right filtered'], int(len(analogdata['Force Z right filtered'])/10))

    # Very low pass filter for first segmentation of stance in braking and propulsion areas
    fc = 5  # Cut-off frequency of the filter
    omega = fc / (fs_analogdata / 2) # Normalize the frequency
    N = 2 # Order of the butterworth filter (scipy.signal.filtfilt is a forwrd-backward linear filter meaning the Nth-order*2 is applied)
    filter_type = 'lowpass' # Type of the filter
    b, a = signal.butter(N, omega, filter_type)
    
    analogdatalabels = list(analogdata.keys())
    for label in analogdatalabels:
        if 'Force.Fy1' in label:
            force_y_left = signal.filtfilt(b, a, analogdata[label]) # Apply filter
        if 'Force.Fy2' in label:
            force_y_right = signal.filtfilt(b, a, analogdata[label]) # Apply filter
            
    # Resample force data to 100 Hz (similar to markerdata)
    force_y_left = signal.resample(force_y_left, int(len(force_y_left)/10))
    force_y_right = signal.resample(force_y_right, int(len(force_y_right)/10))
    
    # First determine stance phase from IC till TC according to vicon data,
    # Deem cross steps as faulty stance phases to calculate propulsive force,
    # Then find the local minimum,
    # Last find zero crossing around local minima as start and stop of propulsion.
    
    # Left side
    spatiotemporals['Stance left index numbers'] = np.array([], dtype=int)
    gaitevents['Propulsion left start'] = np.array([], dtype=int)
    gaitevents['Propulsion left stop'] = np.array([], dtype=int)
    gaitevents['Braking left start'] = np.array([], dtype=int)
    gaitevents['Braking left stop'] = np.array([], dtype=int)
    
    for i in range(0, len(gaitevents['Index numbers initial contact left'])):
        try:
            start = gaitevents['Index numbers initial contact left'][i] # start of stance phase
            stop = gaitevents['Index numbers terminal contact left'][ gaitevents['Index numbers terminal contact left'] > gaitevents['Index numbers initial contact left'][i] ][0] # end of stance phase
            # Identify crossstep: force in Z direction should cross 90% of the bodyweight, force in Z direction of the contralateral side should reach almost 0 at some point during the stance, force in Z direction should at some point before heel-strike and after toe-off reach almost zero
            if np.min(analogdata['Force Z left filtered'][start:stop]) < th_crosssteps and np.any(analogdata['Force Z left filtered'][start:stop] > -1) and analogdata['Force Z left filtered'][start-10] > -10 and analogdata['Force Z left filtered'][stop+10] > -10: # If not cross step: continue
                # Stance phase with correction for cross steps
                spatiotemporals['Stance left index numbers'] = np.append(spatiotemporals['Stance left index numbers'], np.arange(start, stop, step=1)) # save the index numbers of the stance phase
                
                # Find local maximum peak in strongly filtered Y force (= braking force)
                maxpeaks = signal.find_peaks(force_y_left[start+5:stop-10])[0] + start+5
                if len(maxpeaks)>0:
                    localmax = np.argmax(force_y_left[maxpeaks])
                    localmax = int(maxpeaks[localmax])
                    if force_y_left[localmax] < th_crossings: # all data is negative and thus propulsion, (no braking force was generated)
                        localmax = False
                else: # no braking peaks
                    localmax = False
                
                # Find local minimum peak in strongly filtered Y force (= forward force) after the maximum braking force
                minpeaks = signal.find_peaks(-force_y_left[start+10:stop-5])[0] + start+10
                if type(localmax) == int:
                    minpeaks = minpeaks[minpeaks>localmax]
                elif localmax == False:
                    minpeaks = minpeaks[minpeaks>start]
                if len(minpeaks)>0:
                    localmin = np.argmin(force_y_left[minpeaks])
                    localmin = int(minpeaks[localmin])
                    if force_y_left[localmin] > th_crossings: # all data is positive and thus braking, (no propulsive forcef was generated)
                            localmin = False
                else: # no propulsion peaks
                    localmin = False

                
                # Find approximate braking to propulsion point at first positive to negative zero crossing in highly filtered signal
                if type(localmin) == int and type(localmax) == int: # both braking and propulsion
                    braking_to_propulsion = np.argwhere(force_y_left[localmax:localmin] < th_crossings) +localmax
                    if len(braking_to_propulsion) > 0:
                        braking_to_propulsion = int(braking_to_propulsion[0])
                    else:
                        braking_to_propulsion = False # local minimum and local maxium were found, but data not smaller than 0 > only braking
                elif localmin == False or localmax == False: # no braking-to-propulsion transition
                    braking_to_propulsion = False
                   
                              
                # Find actual braking-to-propulsion point based on 20Hz filtered signal
                if type(braking_to_propulsion) == int:
                    signs = np.sign(analogdata['Force Y left filtered'][int(braking_to_propulsion-10) : int(braking_to_propulsion+10)])
                    crossings = np.argwhere(np.diff(signs)<-1) + int(braking_to_propulsion-10) # positive to negative direction
                    true_braking_to_propulsion = int(crossings[np.argmin(np.abs(crossings-braking_to_propulsion))])
                    if true_braking_to_propulsion < start:
                        if np.nanmean(analogdata['Force Y left filtered'][start:stop]) < 0: # only propulsion
                            true_braking_to_propulsion = start # assume no braking, only propulsion during this stance phase
                        if np.nanmean(analogdata['Force Y left filtered'][start:stop]) > 0: # only braking
                            true_braking_to_propulsion = stop # assume no braking, only propulsion during this stance phase
                    gaitevents['Braking left stop'] = np.append(gaitevents['Braking left stop'], true_braking_to_propulsion)
                    gaitevents['Propulsion left start'] = np.append(gaitevents['Propulsion left start'], true_braking_to_propulsion)
                
                elif type(braking_to_propulsion) == bool: # no braking-to-propulsion transition
                    if type(localmin) == int and localmax == False: # No braking
                        gaitevents['Braking left stop'] = np.append(gaitevents['Braking left stop'], start)
                        gaitevents['Propulsion left start'] = np.append(gaitevents['Propulsion left start'], start)
                    elif localmin == False and type(localmax) == int: # No propulsion
                        gaitevents['Braking left stop'] = np.append(gaitevents['Braking left stop'], stop)
                        gaitevents['Propulsion left start'] = np.append(gaitevents['Propulsion left start'], stop)
                    elif type(localmin) == int and type(localmax) == int:
                        gaitevents['Braking left stop'] = np.append(gaitevents['Braking left stop'], stop)
                        gaitevents['Propulsion left start'] = np.append(gaitevents['Propulsion left start'], stop)
                        
                # Find approximate start of braking at "almost zero-crossing" in highly filtered signal    
                if type(localmax) == int:
                    signs = np.sign(((force_y_left/bodyweight)-0.01)[start-10 : localmax])
                    crossings = np.argwhere(np.diff(signs)>1) + int(start-10)
                    if len(crossings) > 0:
                        start_brake = crossings[-1]
                    else:
                        start_brake = np.argmin(((force_y_left/bodyweight)-0.01)[start : localmax]) + start
                elif type(localmax) == bool:
                    start_brake = int(start)
                    
                # Find actual start of braking at closest zero-crossing in 20 Hz filterd signal around approximate start of the break in negative to positive direction
                if type(localmax) == int:
                    signs = np.sign(analogdata['Force Y left filtered'][start-10 : localmax])
                    crossings = np.argwhere(np.diff(signs)>1) + int(start-10) # negative to positive direction
                    if len(crossings) > 0:
                        true_start_brake = int(crossings[np.argmin(np.abs(crossings-start_brake))])
                    else:
                        true_start_brake = int(start)
                elif type(localmax) == bool:
                    true_start_brake = int(start)
                    
                gaitevents['Braking left start'] = np.append(gaitevents['Braking left start'], true_start_brake)
                
                # Find approximate stop of propulsion at "almost zero-crossing" in highly filtered signal
                if type(localmin) == int:
                    signs = np.sign(((force_y_left/bodyweight)+0.01)[localmin : stop +10])
                    crossings = np.argwhere(np.diff(signs)>1) + localmin # negative to positve direction
                    if len(crossings) > 0:
                        stop_prop = crossings[-1]
                    else:
                        stop_prop = np.argmax(((force_y_left/bodyweight)+0.01)[localmin:stop]) + localmin
                elif type(localmin) == bool:
                    stop_prop = int(stop)
                
                # Find actual stop of propulsion at closest zero-crossing in 20 Hz filterd signal around approximate stop of propulsion in negative to positive direction
                if type(localmin) == int:
                    signs = np.sign(analogdata['Force Y left filtered'][localmin : stop +10])
                    crossings = np.argwhere(np.diff(signs)>1) + localmin # negative to positive direction
                    if len(crossings) > 0:
                        true_stop_prop = int(crossings[np.argmin(np.abs(crossings-stop_prop))])
                    else:
                        true_stop_prop = int(stop)                    
                elif type(localmin) == bool:
                    true_stop_prop = int(stop)

                gaitevents['Propulsion left stop'] = np.append(gaitevents['Propulsion left stop'], true_stop_prop)
                        
        except:
            pass
                
    # Right side
    spatiotemporals['Stance right index numbers'] = np.array([], dtype=int)
    gaitevents['Propulsion right start'] = np.array([], dtype=int)
    gaitevents['Propulsion right stop'] = np.array([], dtype=int)
    gaitevents['Braking right start'] = np.array([], dtype=int)
    gaitevents['Braking right stop'] = np.array([], dtype=int)
    
    for i in range(0, len(gaitevents['Index numbers initial contact right'])):
        try:
            start = gaitevents['Index numbers initial contact right'][i] # start of stance phase
            stop = gaitevents['Index numbers terminal contact right'][ gaitevents['Index numbers terminal contact right'] > gaitevents['Index numbers initial contact right'][i] ][0] # end of stance phase
            # Identify crossstep: force in Z direction should cross 90% of the bodyweight, force in Z direction of the contralateral side should reach almost 0 at some point during the stance, force in Z direction should at some point before heel-strike and after toe-off reach almost zero
            if np.min(analogdata['Force Z right filtered'][start:stop]) < th_crosssteps and np.any(analogdata['Force Z right filtered'][start:stop] > -1) and analogdata['Force Z right filtered'][start-10] > -10 and analogdata['Force Z right filtered'][stop+10] > -10: # If not cross step: continue
                # Stance phase with correction for cross steps
                spatiotemporals['Stance right index numbers'] = np.append(spatiotemporals['Stance right index numbers'], np.arange(start, stop, step=1)) # save the index numbers of the stance phase
                
                # Find local maximum peak in strongly filtered Y force (= braking force)
                maxpeaks = signal.find_peaks(force_y_right[start+5:stop-10])[0] + start+5
                if len(maxpeaks)>0:
                    localmax = np.argmax(force_y_right[maxpeaks])
                    localmax = int(maxpeaks[localmax])
                    if force_y_right[localmax] < th_crossings: # all data is negative and thus propulsion, (no braking force was generated)
                        localmax = False
                else: # no braking peaks
                    localmax = False
                
                # Find local minimum peak in strongly filtered Y force (= forward force) after the maximum braking force
                minpeaks = signal.find_peaks(-force_y_right[start+10:stop-5])[0] + start+10
                if type(localmax) == int:
                    minpeaks = minpeaks[minpeaks>localmax]
                elif localmax == False:
                    minpeaks = minpeaks[minpeaks>start]
                if len(minpeaks)>0:
                    localmin = np.argmin(force_y_right[minpeaks])
                    localmin = int(minpeaks[localmin])
                    if force_y_right[localmin] > th_crossings: # all data is positive and thus braking, (no propulsive forcef was generated)
                            localmin = False
                else: # no propulsion peaks
                    localmin = False

                
                # Find approximate braking to propulsion point at first positive to negative zero crossing in highly filtered signal
                if type(localmin) == int and type(localmax) == int: # both braking and propulsion
                    braking_to_propulsion = np.argwhere(force_y_right[localmax:localmin] < th_crossings) +localmax
                    if len(braking_to_propulsion) > 0:
                        braking_to_propulsion = int(braking_to_propulsion[0])
                    else:
                        braking_to_propulsion = False # local minimum and local maxium were found, but data not smaller than 0 > only braking
                elif localmin == False or localmax == False: # no braking-to-propulsion transition
                    braking_to_propulsion = False
                   
                              
                # Find actual braking-to-propulsion point based on 20Hz filtered signal
                if type(braking_to_propulsion) == int:
                    signs = np.sign(analogdata['Force Y right filtered'][int(braking_to_propulsion-10) : int(braking_to_propulsion+10)])
                    crossings = np.argwhere(np.diff(signs)<-1) + int(braking_to_propulsion-10) # positive to negative direction
                    true_braking_to_propulsion = int(crossings[np.argmin(np.abs(crossings-braking_to_propulsion))])
                    if true_braking_to_propulsion < start:
                        if np.nanmean(analogdata['Force Y right filtered'][start:stop]) < 0: # only propulsion
                            true_braking_to_propulsion = start # assume no braking, only propulsion during this stance phase
                        if np.nanmean(analogdata['Force Y right filtered'][start:stop]) > 0: # only braking
                            true_braking_to_propulsion = stop # assume no braking, only propulsion during this stance phase
                    gaitevents['Braking right stop'] = np.append(gaitevents['Braking right stop'], true_braking_to_propulsion)
                    gaitevents['Propulsion right start'] = np.append(gaitevents['Propulsion right start'], true_braking_to_propulsion)
                
                elif type(braking_to_propulsion) == bool: # no braking-to-propulsion transition
                    if type(localmin) == int and localmax == False: # No braking
                        gaitevents['Braking right stop'] = np.append(gaitevents['Braking right stop'], start)
                        gaitevents['Propulsion right start'] = np.append(gaitevents['Propulsion right start'], start)
                    elif localmin == False and type(localmax) == int: # No propulsion
                        gaitevents['Braking right stop'] = np.append(gaitevents['Braking right stop'], stop)
                        gaitevents['Propulsion right start'] = np.append(gaitevents['Propulsion right start'], stop)
                    elif type(localmin) == int and type(localmax) == int:
                        gaitevents['Braking right stop'] = np.append(gaitevents['Braking right stop'], stop)
                        gaitevents['Propulsion right start'] = np.append(gaitevents['Propulsion right start'], stop)
                        
                # Find approximate start of braking at "almost zero-crossing" in highly filtered signal    
                if type(localmax) == int:
                    signs = np.sign(((force_y_right/bodyweight)-0.01)[start-10 : localmax])
                    crossings = np.argwhere(np.diff(signs)>1) + int(start-10)
                    if len(crossings) > 0:
                        start_brake = crossings[-1]
                    else:
                        start_brake = np.argmin(((force_y_right/bodyweight)-0.01)[start : localmax]) + start
                elif type(localmax) == bool:
                    start_brake = int(start)
                    
                # Find actual start of braking at closest zero-crossing in 20 Hz filterd signal around approximate start of the break in negative to positive direction
                if type(localmax) == int:
                    signs = np.sign(analogdata['Force Y right filtered'][start-10 : localmax])
                    crossings = np.argwhere(np.diff(signs)>1) + int(start-10) # negative to positive direction
                    if len(crossings) > 0:
                        true_start_brake = int(crossings[np.argmin(np.abs(crossings-start_brake))])
                    else:
                        true_start_brake = int(start)
                elif type(localmax) == bool:
                    true_start_brake = int(start)
                    
                gaitevents['Braking right start'] = np.append(gaitevents['Braking right start'], true_start_brake)
                
                # Find approximate stop of propulsion at "almost zero-crossing" in highly filtered signal
                if type(localmin) == int:
                    signs = np.sign(((force_y_right/bodyweight)+0.01)[localmin : stop +10])
                    crossings = np.argwhere(np.diff(signs)>1) + localmin # negative to positve direction
                    if len(crossings) > 0:
                        stop_prop = crossings[-1]
                    else:
                        stop_prop = np.argmax(((force_y_right/bodyweight)+0.01)[localmin:stop]) + localmin
                elif type(localmin) == bool:
                    stop_prop = int(stop)
                
                # Find actual stop of propulsion at closest zero-crossing in 20 Hz filterd signal around approximate stop of propulsion in negative to positive direction
                if type(localmin) == int:
                    signs = np.sign(analogdata['Force Y right filtered'][localmin : stop +10])
                    crossings = np.argwhere(np.diff(signs)>1) + localmin # negative to positive direction
                    if len(crossings) > 0:
                        true_stop_prop = int(crossings[np.argmin(np.abs(crossings-stop_prop))])
                    else:
                        true_stop_prop = int(stop)                    
                elif type(localmin) == bool:
                    true_stop_prop = int(stop)

                gaitevents['Propulsion right stop'] = np.append(gaitevents['Propulsion right stop'], true_stop_prop)
                        
        except:
            pass
    
    
    # Remove propulsion start/stop events in first 10 seconds of trial
    gaitevents['Propulsion left start'] = gaitevents['Propulsion left start'][gaitevents['Propulsion left start'] > 10*fs_markerdata]
    try:
        gaitevents['Propulsion left stop'] = gaitevents['Propulsion left stop'][gaitevents['Propulsion left stop'] >= gaitevents['Propulsion left start'][0]]
    except IndexError:
        gaitevents['Propulsion left stop'] = np.array([], dtype=int)
    try:
        gaitevents['Propulsion left start'] = gaitevents['Propulsion left start'][gaitevents['Propulsion left start'] <= gaitevents['Propulsion left stop'][-1]]
    except IndexError:
        gaitevents['Propulsion left start'] = np.array([], dtype=int)
    
    gaitevents['Braking left start'] = gaitevents['Braking left start'][gaitevents['Braking left start'] > 10*fs_markerdata]
    try:
        gaitevents['Braking left stop'] = gaitevents['Braking left stop'][gaitevents['Braking left stop'] >= gaitevents['Braking left start'][0]]
    except IndexError:
        gaitevents['Braking left stop'] = np.array([], dtype=int)
    try:
        gaitevents['Braking left start'] = gaitevents['Braking left start'][gaitevents['Braking left start'] <= gaitevents['Braking left stop'][-1]]
    except IndexError:
        gaitevents['Braking left start'] = np.array([], dtype=int)
    
    gaitevents['Propulsion right start'] = gaitevents['Propulsion right start'][gaitevents['Propulsion right start'] > 10*fs_markerdata]
    try:
        gaitevents['Propulsion right stop'] = gaitevents['Propulsion right stop'][gaitevents['Propulsion right stop'] >= gaitevents['Propulsion right start'][0]]
    except IndexError:
        gaitevents['Propulsion right stop'] = np.array([], dtype=int)
    try:
        gaitevents['Propulsion right start'] = gaitevents['Propulsion right start'][gaitevents['Propulsion right start'] <= gaitevents['Propulsion right stop'][-1]]
    except IndexError:
        gaitevents['Propulsion right start'] = np.array([], dtype=int)
    
    gaitevents['Braking right start'] = gaitevents['Braking right start'][gaitevents['Braking right start'] > 10*fs_markerdata]
    try:
        gaitevents['Braking right stop'] = gaitevents['Braking right stop'][gaitevents['Braking right stop'] >= gaitevents['Braking right start'][0]]
    except IndexError:
        gaitevents['Braking right stop'] = np.array([], dtype=int)
    try:
        gaitevents['Braking right start'] = gaitevents['Braking right start'][gaitevents['Braking right start'] <= gaitevents['Braking right stop'][-1]]
    except IndexError:
        gaitevents['Braking right start'] = np.array([], dtype=int)
    
    
    # Peak breaking and propulsive forces
    gaitevents['Peak propulsion left'] = np.array([], dtype=int)
    for i in range(len(gaitevents['Propulsion left start'])):
        try:
            idxmin = np.argmin(analogdata['Force Y left filtered'] [gaitevents['Propulsion left start'][i] : gaitevents['Propulsion left stop'][i]])
            gaitevents['Peak propulsion left'] = np.append(gaitevents['Peak propulsion left'], gaitevents['Propulsion left start'][i]+idxmin)
        except ValueError:
            pass
    gaitevents['Peak braking left'] = np.array([], dtype=int)
    for i in range(len(gaitevents['Braking left start'])):
        try:
            idxmax = np.argmax(analogdata['Force Y left filtered'] [gaitevents['Braking left start'][i] : gaitevents['Braking left stop'][i]])
            gaitevents['Peak braking left'] = np.append(gaitevents['Peak braking left'], gaitevents['Braking left start'][i]+idxmax)
        except ValueError:
            pass
    gaitevents['Peak propulsion right'] = np.array([], dtype=int)
    for i in range(len(gaitevents['Propulsion right start'])):
        try:
            idxmin = np.argmin(analogdata['Force Y right filtered'] [gaitevents['Propulsion right start'][i] : gaitevents['Propulsion right stop'][i]])
            gaitevents['Peak propulsion right'] = np.append(gaitevents['Peak propulsion right'], gaitevents['Propulsion right start'][i]+idxmin)
        except ValueError:
            pass
    gaitevents['Peak braking right'] = np.array([], dtype=int)
    for i in range(len(gaitevents['Braking right start'])):
        try:
            idxmax = np.argmax(analogdata['Force Y right filtered'] [gaitevents['Braking right start'][i] : gaitevents['Braking right stop'][i]])
            gaitevents['Peak braking right'] = np.append(gaitevents['Peak braking right'], gaitevents['Braking right start'][i]+idxmax)
        except ValueError:
            pass
    
    # Debug plot
    if debugplot == True:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        axs[0].set_title(title, fontsize=20)
        # Left
        axs[0].plot(analogdata['Force Y left filtered']/bodyweight, 'k', label='Force Y left')
        axs[0].plot(force_y_left/bodyweight, 'grey', label='Filtered force Y left')
        # axs[0].plot(analogdata['Force Z left filtered']/bodyweight, 'orange', label='Force Z left')
        axs[0].plot(gaitevents['Index numbers initial contact left'], analogdata['Force Y left filtered'][gaitevents['Index numbers initial contact left']]/bodyweight, 'r.')
        axs[0].plot(gaitevents['Index numbers terminal contact left'], analogdata['Force Y left filtered'][gaitevents['Index numbers terminal contact left']]/bodyweight, 'g.')
        # axs[0].plot(gaitevents['Braking left start'], analogdata['Force Y left filtered'][gaitevents['Braking left start']]/bodyweight, 'kx', label='Braking start')
        axs[0].vlines(x=gaitevents['Braking left start'], ymin=np.min(analogdata['Force Y left filtered']/bodyweight), ymax=np.max(analogdata['Force Y left filtered']/bodyweight), color='red')
        axs[0].vlines(x=gaitevents['Propulsion left start'], ymin=np.min(analogdata['Force Y left filtered']/bodyweight), ymax=np.max(analogdata['Force Y left filtered']/bodyweight), color='grey')
        axs[0].vlines(x=gaitevents['Propulsion left stop'], ymin=np.min(analogdata['Force Y left filtered']/bodyweight), ymax=np.max(analogdata['Force Y left filtered']/bodyweight), color='green')
        # axs[0].plot(gaitevents['Propulsion left stop'], analogdata['Force Y left filtered'][gaitevents['Propulsion left stop']]/bodyweight, 'kx', label='Propulsion stop')
        axs[0].plot(gaitevents['Peak propulsion left'], analogdata['Force Y left filtered'][gaitevents['Peak propulsion left']]/bodyweight, 'gx', label='Propulsion peak')
        axs[0].plot(gaitevents['Peak braking left'], analogdata['Force Y left filtered'][gaitevents['Peak braking left']]/bodyweight, 'rx', label='Braking peak')
        axs[0].hlines(xmin=0, xmax=len(analogdata['Force Y left filtered']), y=0, color='grey')
        
        for i in range(0, len(gaitevents['Propulsion left start'])):
            axs[0].fill_between(x=np.arange(gaitevents['Propulsion left start'][i], gaitevents['Propulsion left stop'][i]), y1=analogdata['Force Y left filtered'][gaitevents['Propulsion left start'][i] : gaitevents['Propulsion left stop'][i]]/bodyweight, y2=0, color='lightgreen')
        for i in range(0, len(gaitevents['Braking left start'])):
            axs[0].fill_between(x=np.arange(gaitevents['Braking left start'][i], gaitevents['Braking left stop'][i]), y1=analogdata['Force Y left filtered'][gaitevents['Braking left start'][i] : gaitevents['Braking left stop'][i]]/bodyweight, y2=0, color='pink')
                
        #Right
        axs[1].plot(analogdata['Force Y right filtered']/bodyweight, 'k', label='Force Y')
        axs[1].plot(force_y_right/bodyweight, 'grey', label='Filtered force Y right')
        # axs[1].plot(analogdata['Force Z right filtered']/bodyweight, 'orange', label='Force Z')
        axs[1].plot(gaitevents['Index numbers initial contact right'], analogdata['Force Y right filtered'][gaitevents['Index numbers initial contact right']]/bodyweight, 'r.', label = 'IC')
        axs[1].plot(gaitevents['Index numbers terminal contact right'], analogdata['Force Y right filtered'][gaitevents['Index numbers terminal contact right']]/bodyweight, 'g.', label = 'TC')
        # axs[1].plot(gaitevents['Propulsion right start'], analogdata['Force Y right filtered'][gaitevents['Propulsion right start']]/bodyweight, 'gv', label='Propulsion start')
        # axs[1].plot(gaitevents['Propulsion right stop'], analogdata['Force Y right filtered'][gaitevents['Propulsion right stop']]/bodyweight, 'rv', label='Propulsion stop')
        axs[1].vlines(x=gaitevents['Braking right start'], ymin=np.min(analogdata['Force Y right filtered']/bodyweight), ymax=np.max(analogdata['Force Y right filtered']/bodyweight), color='red')
        axs[1].vlines(x=gaitevents['Propulsion right start'], ymin=np.min(analogdata['Force Y right filtered']/bodyweight), ymax=np.max(analogdata['Force Y right filtered']/bodyweight), color='grey')
        axs[1].vlines(x=gaitevents['Propulsion right stop'], ymin=np.min(analogdata['Force Y right filtered']/bodyweight), ymax=np.max(analogdata['Force Y right filtered']/bodyweight), color='green')
        axs[1].plot(gaitevents['Peak propulsion right'], analogdata['Force Y right filtered'][gaitevents['Peak propulsion right']]/bodyweight, 'gx', label='Propulsion peak')
        axs[1].plot(gaitevents['Peak braking right'], analogdata['Force Y right filtered'][gaitevents['Peak braking right']]/bodyweight, 'rx', label='Braking peak')
        axs[1].hlines(xmin=0, xmax=len(analogdata['Force Y right filtered']), y=0, color='grey')
        
        for i in range(0, len(gaitevents['Propulsion right start'])):
            axs[1].fill_between(x=np.arange(gaitevents['Propulsion right start'][i], gaitevents['Propulsion right stop'][i]), y1=analogdata['Force Y right filtered'][gaitevents['Propulsion right start'][i] : gaitevents['Propulsion right stop'][i]]/bodyweight, y2=0, color='lightgreen')
        
        for i in range(0, len(gaitevents['Braking right start'])):
            axs[1].fill_between(x=np.arange(gaitevents['Braking right start'][i], gaitevents['Braking right stop'][i]), y1=analogdata['Force Y right filtered'][gaitevents['Braking right start'][i] : gaitevents['Braking right stop'][i]]/bodyweight, y2=0, color='pink')
        axs[1].legend()


        
    # Left side
    # Propultion = area under the negative curve
    spatiotemporals['Propulsion left'] = np.zeros(shape=(len(gaitevents['Propulsion left start']),3)) *np.nan
    for i in range(len(gaitevents['Propulsion left start'])):
        spatiotemporals['Propulsion left'][i,0] = gaitevents['Propulsion left start'][i]
        spatiotemporals['Propulsion left'][i,1] = gaitevents['Propulsion left stop'][i]
        # Compute the area using the composite trapezoidal rule.
        this_propulsion = analogdata['Force Y left filtered'][gaitevents['Propulsion left start'][i]:gaitevents['Propulsion left stop'][i]]
        forward_force = (np.abs(np.trapz(this_propulsion[this_propulsion<0])) *1/fs_markerdata)/bodyweight
        backward_force = (np.abs(np.trapz(this_propulsion[this_propulsion>0])) *1/fs_markerdata)/bodyweight
        spatiotemporals['Propulsion left'][i,2] = forward_force - backward_force
        if spatiotemporals['Propulsion left'][i,2] < 0:
            spatiotemporals['Propulsion left'][i,2]= np.nan
    # Peak propulsion
    spatiotemporals['Peak propulsion left'] = np.zeros(shape=(len(gaitevents['Peak propulsion left']),2)) *np.nan
    for i in range(len(gaitevents['Peak propulsion left'])):
        spatiotemporals['Peak propulsion left'][i,0] = gaitevents['Peak propulsion left'][i]
        spatiotemporals['Peak propulsion left'][i,1] = (analogdata['Force Y left filtered'][gaitevents['Peak propulsion left'][i]])/bodyweight
    # Braking = area under the curve
    spatiotemporals['Braking left'] = np.zeros(shape=(len(gaitevents['Braking left start']),3)) *np.nan
    for i in range(len(gaitevents['Braking left start'])):
        spatiotemporals['Braking left'][i,0] = gaitevents['Braking left start'][i]
        spatiotemporals['Braking left'][i,1] = gaitevents['Braking left stop'][i]
        # Compute the area using the composite trapezoidal rule.
        this_brake = analogdata['Force Y left filtered'][gaitevents['Braking left start'][i]:gaitevents['Braking left stop'][i]]
        forward_force = (np.abs(np.trapz(this_brake[this_brake<0])) *1/fs_markerdata)/bodyweight
        backward_force = (np.abs(np.trapz(this_brake[this_brake>0])) *1/fs_markerdata)/bodyweight
        spatiotemporals['Braking left'][i,2] = backward_force - forward_force
        if spatiotemporals['Braking left'][i,2] < 0:
            spatiotemporals['Braking left'][i,2]= np.nan
    # Peak braking
    spatiotemporals['Peak braking left'] = np.zeros(shape=(len(gaitevents['Peak braking left']),2)) *np.nan
    for i in range(len(gaitevents['Peak braking left'])):
        spatiotemporals['Peak braking left'][i,0] = gaitevents['Peak braking left'][i]
        spatiotemporals['Peak braking left'][i,1] = (analogdata['Force Y left filtered'][gaitevents['Peak braking left'][i]])/bodyweight
    
    # Right side
    # Propultion = area under the negative curve
    spatiotemporals['Propulsion right'] = np.zeros(shape=(len(gaitevents['Propulsion right start']),3)) *np.nan
    for i in range(len(gaitevents['Propulsion right start'])):
        spatiotemporals['Propulsion right'][i,0] = gaitevents['Propulsion right start'][i]
        spatiotemporals['Propulsion right'][i,1] = gaitevents['Propulsion right stop'][i]
        # Compute the area using the composite trapezoidal rule.
        this_propulsion = analogdata['Force Y right filtered'][gaitevents['Propulsion right start'][i]:gaitevents['Propulsion right stop'][i]]
        forward_force = (np.abs(np.trapz(this_propulsion[this_propulsion<0])) *1/fs_markerdata)/bodyweight
        backward_force = (np.abs(np.trapz(this_propulsion[this_propulsion>0])) *1/fs_markerdata)/bodyweight
        spatiotemporals['Propulsion right'][i,2] = forward_force - backward_force
        if spatiotemporals['Propulsion right'][i,2] < 0:
            spatiotemporals['Propulsion right'][i,2]= np.nan
    # Peak propulsion
    spatiotemporals['Peak propulsion right'] = np.zeros(shape=(len(gaitevents['Peak propulsion right']),2)) *np.nan
    for i in range(len(gaitevents['Peak propulsion right'])):
        spatiotemporals['Peak propulsion right'][i,0] = gaitevents['Peak propulsion right'][i]
        spatiotemporals['Peak propulsion right'][i,1] = (analogdata['Force Y right filtered'][gaitevents['Peak propulsion right'][i]])/bodyweight
    # Braking = area under the curve
    spatiotemporals['Braking right'] = np.zeros(shape=(len(gaitevents['Braking right start']),3)) *np.nan
    for i in range(len(gaitevents['Braking right start'])):
        spatiotemporals['Braking right'][i,0] = gaitevents['Braking right start'][i]
        spatiotemporals['Braking right'][i,1] = gaitevents['Braking right stop'][i]
        # Compute the area using the composite trapezoidal rule.
        this_brake = analogdata['Force Y right filtered'][gaitevents['Braking right start'][i]:gaitevents['Braking right stop'][i]]
        forward_force = (np.abs(np.trapz(this_brake[this_brake<0])) *1/fs_markerdata)/bodyweight
        backward_force = (np.abs(np.trapz(this_brake[this_brake>0])) *1/fs_markerdata)/bodyweight
        spatiotemporals['Braking right'][i,2] = backward_force - forward_force
        if spatiotemporals['Braking right'][i,2] < 0:
            spatiotemporals['Braking right'][i,2]= np.nan
    # Peak braking
    spatiotemporals['Peak braking right'] = np.zeros(shape=(len(gaitevents['Peak braking right']),2)) *np.nan
    for i in range(len(gaitevents['Peak braking right'])):
        spatiotemporals['Peak braking right'][i,0] = gaitevents['Peak braking right'][i]
        spatiotemporals['Peak braking right'][i,1] = (analogdata['Force Y right filtered'][gaitevents['Peak braking right'][i]])/bodyweight
    
    
    return gaitevents, spatiotemporals, analogdata