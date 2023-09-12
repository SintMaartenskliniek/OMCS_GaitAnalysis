"""

Function to detect gait events from marker data.
Can be applied to both treadmill and overground data.

Based on:
    Zeni, J. A., Jr, Richards, J. G., & Higginson, J. S. (2008).
    Two simple methods for determining gait events during treadmill and overground walking using kinematic data.
    Gait & posture, 27(4), 710–714. https://doi.org/10.1016/j.gaitpost.2007.07.007
    
INPUT
gaiteventdetection(markerdata, sample_frequency)
    markerdata:        dictionary of labeled marker data 
    sample_frequency:  sample frequency of marker data

OPTIONAL INPUT
gaiteventdetection(**kwargs)
    algorithmtype:     string; can be 'velocity' or 'coordinate'; defaults to 'velocity'
    trialtype:         string; can be 'treadmill' or 'overground'; defaults to 'treadmill'
    debugplot:         bool; can be 'True' or 'False'; defaults to 'False'
    
OUTPUT
    gaitevents:        dictionary with index numbers of initial contact and terminal contact events

Copyright (c):
    2023, Carmen Ensink, Sint Maartenskliniek,
    c.ensink@maartenskliniek.nl

    
"""

# Import dependencies
import numpy as np
from scipy.signal import find_peaks
from scipy import signal
import matplotlib.pyplot as plt


def gaiteventdetection(markerdata, sample_frequency, **kwargs):
    
    # Set defaults
    algorithmtype = 'velocity'
    trialtype = 'treadmill'
    debugplot = False
    gaitevents={}
    
    # Check optional input arguments
    for key, value in kwargs.items():
        if key == 'algorithmtype':
            algorithmtype = value
        if key == 'trialtype':
            trialtype = value
        if key == 'debugplot':
            debugplot = value
    
    # Define sacrum
    if 'LPSI' in markerdata:
        sacrum = (markerdata['LPSI'] + markerdata['RPSI']) / 2 # Middle between Left and Right Posterior Superior Iliac Spine
    # Correct for missing data in either LPSI or RPSI marker data
    for i in range(len(markerdata['LPSI'])):
        if np.all(markerdata['LPSI'][i,:] == [0,0,0]) or np.all(markerdata['RPSI'][i,:] == [0,0,0]):
            sacrum[i,:]=[0,0,0]
            
    # Left foot
    heel_left = markerdata['LHEE']
    toe_left = markerdata['LTOE']
    # Right foot
    heel_right = markerdata['RHEE']
    toe_right = markerdata['RTOE']
    
    # Filter the heel and toe marker data
    # Low pass Butterworth filter, order = 2, cut-off frequecy = 15
    fc = 15  # Cut-off frequency of the filter
    w = fc / (sample_frequency / 2) # Normalize the frequency
    N = 2 # Order of the butterworth filter
    filter_type = 'lowpass' # Type of the filter
    b, a = signal.butter(N, w, filter_type)

    # Apply filter on marker data
    # Left heel marker
    fheel_Lx = signal.filtfilt(b, a, heel_left[:,0])
    fheel_Ly = signal.filtfilt(b, a, heel_left[:,1])
    fheel_Lz = signal.filtfilt(b, a, heel_left[:,2])
    heelL = np.vstack((fheel_Lx, fheel_Ly))
    heelL = np.vstack((heelL, fheel_Lz))
    heel_left = np.transpose(heelL)
    # Correct for missing data
    for i in range(len(markerdata['LHEE'])):
        if np.all(markerdata['LHEE'][i,:] == [0,0,0]):
            heel_left[i,:] = [0,0,0]
    
    # Right heel marker
    fheel_Rx = signal.filtfilt(b, a, heel_right[:,0])
    fheel_Ry = signal.filtfilt(b, a, heel_right[:,1])
    fheel_Rz = signal.filtfilt(b, a, heel_right[:,2])
    heelR = np.vstack((fheel_Rx, fheel_Ry))
    heelR = np.vstack((heelR, fheel_Rz))
    heel_right = np.transpose(heelR)
    # Correct for missing data
    for i in range(len(markerdata['RHEE'])):
        if np.all(markerdata['RHEE'][i,:] == [0,0,0]):
            heel_right[i,:] = [0,0,0]
    
    # Left toe marker
    ftoe_Lx = signal.filtfilt(b, a, toe_left[:,0])
    ftoe_Ly = signal.filtfilt(b, a, toe_left[:,1])
    ftoe_Lz = signal.filtfilt(b, a, toe_left[:,2])
    toeL = np.vstack((ftoe_Lx, ftoe_Ly))
    toeL = np.vstack((toeL, ftoe_Lz))
    toe_left = np.transpose(toeL)
    # Correct for missing data
    for i in range(len(markerdata['LTOE'])):
        if np.all(markerdata['LTOE'][i,:] == [0,0,0]):
            toe_left[i,:] = [0,0,0]
    
    # Right toe marker
    ftoe_Rx = signal.filtfilt(b, a, toe_right[:,0])
    ftoe_Ry = signal.filtfilt(b, a, toe_right[:,1])
    ftoe_Rz = signal.filtfilt(b, a, toe_right[:,2])
    toeR = np.vstack((ftoe_Rx, ftoe_Ry))
    toeR = np.vstack((toeR, ftoe_Rz))
    toe_right = np.transpose(toeR)
    # Correct for missing data
    for i in range(len(markerdata['RTOE'])):
        if np.all(markerdata['RTOE'][i,:] == [0,0,0]):
            toe_right[i,:] = [0,0,0]
            
    match trialtype:
        case 'treadmill':
            APdirection = 1 # Anterior-Posterior direction along the y-axis
            axisdef = -1 # Along negative axis
        case 'overground':
            APdirection = 0 # Anterior-Posterior direction along the x-axis
            axisdef = 1 # Along positive axis
    
    # Velocity of sacrum
    velocity_sacrum = np.append(np.array([0]), np.diff(axisdef*sacrum[:,APdirection]))
    
    # Define thresholds for peak identification
    # Threshold for peak prominence
    thprom = 60 
    # Estimate average gait cycle duration
    ms,_=find_peaks(heel_left[:,2], prominence=thprom) 
    # Threshold for distance between peaks
    thdist = 0.6*np.median(np.diff(ms)) 
    
    
    
    # VELOCITY BASED ALGORITHM
    match algorithmtype:
        case 'velocity':
        
            # To apply to overground data, subtract the anterior-posterior coordinate of the sacral marker from the AP-coordinate of each marker
            if trialtype == 'overground':
                heel_left[:,APdirection] = heel_left[:,APdirection] - sacrum[:,APdirection]
                toe_left[:,APdirection] = toe_left[:,APdirection] - sacrum[:,APdirection]
                heel_right[:,APdirection] = heel_right[:,APdirection] - sacrum[:,APdirection]
                toe_right[:,APdirection] = toe_right[:,APdirection] - sacrum[:,APdirection]
            
            # Calculate markervelocity, to make equal length, initial velocity is set to 0
            velocity_LHEE = np.append(np.array([0]), np.diff(axisdef*heel_left[:,APdirection])) # mm/frame, anterior posterior direction
            velocity_LTOE = np.append(np.array([0]), np.diff(axisdef*toe_left[:,APdirection])) # mm/frame, anterior posterior direction
            velocity_RHEE = np.append(np.array([0]), np.diff(axisdef*heel_right[:,APdirection])) # mm/frame, anterior posterior direction    
            velocity_RTOE = np.append(np.array([0]), np.diff(axisdef*toe_right[:,APdirection])) # mm/frame, anterior posterior direction
        
            # Overground walking trials are walking up-and-down the positive axis, correct for this.
            if trialtype == 'overground':
                # Correct for changing walking direction
                velocity_LHEE[velocity_sacrum < 0] = -1*velocity_LHEE[velocity_sacrum < 0]
                velocity_LTOE[velocity_sacrum < 0] = -1*velocity_LTOE[velocity_sacrum < 0]
                velocity_RHEE[velocity_sacrum < 0] = -1*velocity_RHEE[velocity_sacrum < 0]
                velocity_RTOE[velocity_sacrum < 0] = -1*velocity_RTOE[velocity_sacrum < 0]
                
            # Correct for false high velocity due to missing data
            velocity_LHEE[np.abs(velocity_LHEE)>100] = 0
            velocity_LTOE[np.abs(velocity_LTOE)>100] = 0
            velocity_RHEE[np.abs(velocity_RHEE)>100] = 0
            velocity_RTOE[np.abs(velocity_RTOE)>100] = 0
                            
            # Find sign changes in veloctiy data
            # Left heel
            signLHEE = np.sign(velocity_LHEE)
            signchangeLHEE = np.argwhere(((np.roll(signLHEE, 1) - signLHEE) != 0).astype(int))
            idxICleft = np.array([0], dtype = int)
            for i in range(0, len(signchangeLHEE)):
                if velocity_LHEE[signchangeLHEE[i]] < 0 and heel_left[signchangeLHEE[i],2] < 120:
                    if signchangeLHEE[i] > idxICleft[-1] + thdist:
                        if np.abs(velocity_LHEE[signchangeLHEE[i]]) < np.abs(velocity_LHEE[signchangeLHEE[i]-1]):
                            idxICleft = np.append(idxICleft, signchangeLHEE[i]) # AP component of velocity vector changes from positive to negative
                        else:
                            idxICleft = np.append(idxICleft, signchangeLHEE[i]-1)
            idxICleft = idxICleft[1:]
            
            # Left toe
            signLTOE = np.sign(velocity_LTOE)
            signchangeLTOE = np.argwhere(((np.roll(signLTOE, 1) - signLTOE) != 0).astype(int))
            idxTCleft = np.array([0], dtype = int)
            for i in range(0, len(signchangeLTOE)):
                if velocity_LTOE[signchangeLTOE[i]] > 0 and toe_left[signchangeLTOE[i],2] < 100:
                    if signchangeLTOE[i] > idxTCleft[-1] + thdist:
                        if np.abs(velocity_LTOE[signchangeLTOE[i]]) < np.abs(velocity_LTOE[signchangeLTOE[i]-1]):
                            idxTCleft = np.append(idxTCleft, signchangeLTOE[i]) # AP component of velocity vector changes from negative to positive
                        else:
                            idxTCleft = np.append(idxTCleft, signchangeLTOE[i]-1) # AP component of velocity vector changes from negative to positive
            idxTCleft = idxTCleft[1:] 
            
            # Right heel
            signRHEE = np.sign(velocity_RHEE)
            signchangeRHEE = np.argwhere(((np.roll(signRHEE, 1) - signRHEE) != 0).astype(int))
            idxICright = np.array([0], dtype = int)
            for i in range(0, len(signchangeRHEE)):
                if velocity_RHEE[signchangeRHEE[i]] < 0 and heel_right[signchangeRHEE[i],2] < 120:
                    if signchangeRHEE[i] > idxICright[-1] + thdist:
                        if np.abs(velocity_RHEE[signchangeRHEE[i]]) < np.abs(velocity_RHEE[signchangeRHEE[i]-1]):
                            idxICright = np.append(idxICright, signchangeRHEE[i]) # AP component of velocity vector changes from positive to negative
                        else:
                            idxICright = np.append(idxICright, signchangeRHEE[i]-1) # AP component of velocity vector changes from positive to negative
            idxICright = idxICright[1:]
                    
            # Right toe
            signRTOE = np.sign(velocity_RTOE)
            signchangeRTOE = np.argwhere(((np.roll(signRTOE, 1) - signRTOE) != 0).astype(int))
            idxTCright = np.array([0], dtype = int)
            for i in range(0, len(signchangeRTOE)):
                if velocity_RTOE[signchangeRTOE[i]] > 0 and toe_right[signchangeRTOE[i],2] < 100:
                    if signchangeRTOE[i] > idxTCright[-1] + thdist:
                        if np.abs(velocity_RTOE[signchangeRTOE[i]]) < np.abs(velocity_RTOE[signchangeRTOE[i]-1]):
                            idxTCright = np.append(idxTCright, signchangeRTOE[i])# AP component of velocity vector changes from negative to positive
                        else:
                            idxTCright = np.append(idxTCright, signchangeRTOE[i]-1)# AP component of velocity vector changes from negative to positive
            idxTCright = idxTCright[1:]
        
        
        
        # COORDINATE BASED ALGORITHM   
        case 'coordinate':
            
            # Subract sacrum from heel and toe in Anterior-Posterior direction
            diffHeel_left = heel_left
            diffHeel_left[:,APdirection] = heel_left[:,APdirection] - sacrum[:,APdirection] # Subtract AP coordinate of sacrum from heel
            diffToe_left = toe_left
            diffToe_left[:,APdirection] = toe_left[:,APdirection] - sacrum[:,APdirection] # Subtract AP coordinate of sacrum from toe
            
            diffHeel_right = heel_right
            diffHeel_right[:,APdirection] = heel_right[:,APdirection] - sacrum[:,APdirection] # Subtract AP coordinate of sacrum from heel
            diffToe_right = toe_right
            diffToe_right[:,APdirection] = toe_right[:,APdirection] - sacrum[:,APdirection] # Subtract AP coordinate of sacrum from toe
        
        
            idxICleft, _ = find_peaks(-diffHeel_left[:,1], prominence = thprom) # Find negative peaks (Heel strike)
            idxTCleft, _ = find_peaks(diffToe_left[:,1], prominence = thprom) # Find positive peaks (Toe off)
            
            idxICright, _ = find_peaks(-diffHeel_right[:,1], prominence = thprom) # Find negative peaks (Heel strike)
            idxTCright, _ = find_peaks(diffToe_right[:,1], prominence = thprom) # Find positive peaks (Toe off)
            
            # Deem heel strikes before the first toe off as artefact
            remove = np.array([], dtype = int)
            for i in range(0,len(idxICleft)):
                if idxICleft[i] < idxTCleft[0]:
                    remove = np.append(remove, idxICleft[i])
            idxICleft = idxICleft[~np.in1d(idxICleft,remove)]
            remove = np.array([], dtype = int)
            for i in range(0,len(idxICright)):
                if idxICright[i] < idxTCright[0]:
                    remove = np.append(remove, idxICright[i])
            idxICright = idxICright[~np.in1d(idxICright, remove)]
            # Deem toe off after last heel strike as artefact
            remove = np.array([], dtype = int)
            for i in range(0,len(idxTCleft)):
                if idxTCleft[i] > idxICleft[-1]:
                    remove = np.append(remove, idxTCleft[i])
            idxTCleft = idxTCleft[~np.in1d(idxTCleft, remove)]
            remove = np.array([], dtype = int)
            for i in range(0,len(idxTCright)):
                if idxTCright[i] > idxICright[-1]:
                    remove = np.append(remove, idxTCright[i])
            idxTCright = idxTCright[~np.in1d(idxTCright, remove)]

    
    # Remove all events in first 50 and last 50 samples
    firstn = 50
    lastn = 50
    idxTCleft = idxTCleft[idxTCleft>firstn]
    idxTCleft = idxTCleft[idxTCleft<len(toe_left)-lastn]
    idxICleft = idxICleft[idxICleft>firstn]
    idxICleft = idxICleft[idxICleft<len(heel_left)-lastn]
    idxICleft = idxICleft[idxICleft>idxTCleft[0]]
    idxTCleft = idxTCleft[idxTCleft<idxICleft[-1]]
    idxTCright = idxTCright[idxTCright>firstn]
    idxTCright = idxTCright[idxTCright<len(toe_right)-lastn]
    idxICright = idxICright[idxICright>firstn]
    idxICright = idxICright[idxICright<len(heel_right)-lastn]
    idxICright = idxICright[idxICright>idxTCright[0]]
    idxTCright = idxTCright[idxTCright<idxICright[-1]]
    
    # Remove events in case markerdata is missing at instant of found event
    removeICleft=np.array([])
    for t in range(0,len(idxICleft)):
        if np.all(markerdata['LPSI'][idxICleft[t],:] == [0,0,0]) or np.all(markerdata['RPSI'][idxICleft[t],:] == [0,0,0]) or np.all(markerdata['LHEE'][idxICleft[t],:] == [0,0,0]) or np.all(markerdata['LTOE'][idxICleft[t],:] == [0,0,0]):
            removeICleft = np.append(removeICleft, idxICleft[t])
    for r in range(0,len(removeICleft)):
        idxICleft = np.delete(idxICleft, np.where(idxICleft == removeICleft[r]))
    
    removeICright=np.array([])
    for t in range(0,len(idxICright)):
        if np.all(markerdata['LPSI'][idxICright[t],:] == [0,0,0]) or np.all(markerdata['RPSI'][idxICright[t],:] == [0,0,0]) or np.all(markerdata['RHEE'][idxICright[t],:] == [0,0,0]) or np.all(markerdata['RTOE'][idxICright[t],:] == [0,0,0]):
            removeICright = np.append(removeICright, idxICright[t])
    for r in range(0,len(removeICright)):
        idxICright = np.delete(idxICright, np.where(idxICright == removeICright[r]))
        
    removeTCleft=np.array([])
    for t in range(0,len(idxTCleft)):
        if np.all(markerdata['LPSI'][idxTCleft[t],:] == [0,0,0]) or np.all(markerdata['RPSI'][idxTCleft[t],:] == [0,0,0]) or np.all(markerdata['LTOE'][idxTCleft[t],:] == [0,0,0]) or np.all(markerdata['LHEE'][idxTCleft[t],:] == [0,0,0]):
            removeTCleft = np.append(removeTCleft, idxTCleft[t])
    for r in range(0,len(removeTCleft)):
        idxTCleft = np.delete(idxTCleft, np.where(idxTCleft == removeTCleft[r]))
    
    removeTCright=np.array([])
    for t in range(0,len(idxTCright)):
        if np.all(markerdata['LPSI'][idxTCright[t],:] == [0,0,0]) or np.all(markerdata['RPSI'][idxTCright[t],:] == [0,0,0]) or np.all(markerdata['RTOE'][idxTCright[t],:] == [0,0,0]) or np.all(markerdata['RHEE'][idxTCright[t],:] == [0,0,0]):
            removeTCright = np.append(removeTCright, idxTCright[t])
    for r in range(0,len(removeTCright)):
        idxTCright = np.delete(idxTCright, np.where(idxTCright == removeTCright[r]))
        
    
    
    # Plot a debug figure
    if debugplot == True:
        fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
        
        ax1.plot(velocity_LHEE, label = 'Heel velocity left')
        ax1.plot(idxICleft, velocity_LHEE[idxICleft], 'mv')
        ax1.plot(velocity_LTOE, label = 'Toe velocity left')
        ax1.plot(idxTCleft, velocity_LTOE[idxTCleft], 'b^')
        
        ax2.plot(velocity_RHEE, label = 'Heel velocity right')
        ax2.plot(idxICright, velocity_RHEE[idxICright], 'mv')
        ax2.plot(velocity_RTOE, label = 'Toe velocity right')
        ax2.plot(idxTCright, velocity_RTOE[idxTCright], 'b^')
        
        plt.legend()
    
    
    # Put gait event index numbers in a dictionary output
    gaitevents['Index numbers initial contact left'] = np.sort(idxICleft)
    gaitevents['Index numbers initial contact right'] = np.sort(idxICright)
    gaitevents['Index numbers terminal contact left'] = np.sort(idxTCleft)
    gaitevents['Index numbers terminal contact right'] = np.sort(idxTCright)
    
    return gaitevents
    