"""

Function to read marker data and analog data from .c3d files

INPUT
readmarkerdata (filepath)
    filepath:       Filepath to location of the .c3d file

OUTPUT
    markerdata:     Dictionary with marker data, dict keys are the marker labels
    fs_markerdata:  Sample frequency of the marker data; defaults to 100 Hz
    analogdata:     Dictionary with analog data (force plates), dict keys are the analog labels; defaults to False (not available)
    fs_analogdata:  Sample frequency of the analog data (force plates); defaults to False (not available)
    
Copyright (c):
    2023, Carmen Ensink, Sint Maartenskliniek,
    c.ensink@maartenskliniek.nl
    
    
"""

# Import dependencies
import c3d
import numpy as np


def readmarkerdata (filepath, **kwargs):
    
    # Set defaults
    fs_markerdata = 100 # Sample frequency in Hz
    fs_analogdata = False # Sample frequency in Hz, defaults to False (not available)
    markerlabels = list() # List of marker labels
    analog_available = False # Set availability of analog data channels (force plates) to False (not available)
    analoglabels = list() # List of analog labels
    markerdata_list = list() # List to store markerdata
    analog_data_list = list() # List to store analog data (force plates)
    markerdata = dict() # Output dictionary for marker data
    analogdata = dict() # Output dictionary for analog data (force plates)
    
    # Define reader object with fields stored in the c3d.py file
    reader = c3d.Reader(open(filepath, 'rb'))

    #  Define sample frequency of the optical marker data
    fs_markerdata = reader.point_rate
    
    # Read marker label names from the point_labels field, the order of the labels is in line with the marker position data
    markerlabels=reader.point_labels
    
    # Check if analog data (force plates) is available
    analog_available = reader.analog_used>0
    
    # Read analog label names and sample frequency
    if analog_available == True:
        analog_per_frame=reader.header.analog_per_frame
        fs_analogdata = reader.analog_rate
        analoglabels = list(reader.analog_labels)
    elif analog_available == False:
        analoglabels = 'None available'
    
    # Read the actual data
    for i, points, analog in reader.read_frames():
    #            frames : sequence of (frame number, points, analog)
    #            This method generates a sequence of (frame number, points, analog)
    #            tuples, one tuple per frame. The first element of each tuple is the
    #            frame number. The second is a numpy array of parsed, 5D point data
    #            and the third element of each tuple is a numpy array of analog
    #            values that were recorded during the frame. (Often the analog data
    #            are sampled at a higher frequency than the 3D point data, resulting
    #            in multiple analog frames per frame of point data.)
    #
    #            The first three columns in the returned point data are the (x, y, z)
    #            coordinates of the observed motion capture point. The fourth column
    #            is an estimate of the error for this particular point, and the fifth
    #            column is the number of cameras that observed the point in question.
    #            Both the fourth and fifth values are -1 if the point is considered
    #            to be invalid.
        markerdata_list.append((points[:,0:3]).T)# Read only the x, y, z coordinate and append to the markerdata list

        if analog_available == True:
            analog_data_list.append(analog.T)
        elif analog_available == False:
            continue
                
    # Convert the markerdata_list and analog_data_list to numpy arrays
    marker_data=np.stack(markerdata_list, axis=2)
    if analog_available == True:
        analog_data = np.vstack(analog_data_list)
    elif analog_available == False:
        analog_data = np.array([])
        
    # Store markerdata as dictionary with markerlabels as keys
    for i in range(0, len(markerlabels)):
            marker_x = marker_data[0,i,:]
            marker_y = marker_data[1,i,:]
            marker_z = marker_data[2,i,:]
            markerdata[markerlabels[i].split(' ')[0]] = np.transpose(np.array([marker_x, marker_y,marker_z]))
    
    # Store analogdata as dictionary with analoglabels as keys
    for i in range(0,len(analoglabels)):
        analogdata[analoglabels[i]] = analog_data[:,i]
    
    # Check if start and stop frames were adjusted in data labelling software
    actual_start_frame = reader.first_frame
    actual_stop_frame = reader.last_frame
    
    return markerdata, fs_markerdata, analogdata, fs_analogdata
        
