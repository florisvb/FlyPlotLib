# this file contains handy functions for preprocessing data for plotting, like wrapping and unwrapping angles, and removing discontinuities

import numpy as np
import copy


###################################################################################################
# Remove discontinuities
###################################################################################################
# Thanks to HYRY on stackoverflow.com

def remove_discontinuities(x, y, jump=2):
    """
    Returns an array with np.nan inserted between all discontinuities (defined by the size of jump), this prevents plotting functions from plotting the connecting line, which looks ugly.
    
    x    -- numpy array (1 dim)
    y    -- numpy array with the discontinuities (1 dim)
    jump -- size of discontinuity to break
    
    """

    pos = np.where(np.abs(np.diff(y)) >= jump)[0]+1
    x = np.insert(x, pos, np.nan)
    y = np.insert(y, pos, np.nan)
    
    return x, y
    
###################################################################################################
def normalize(array):
    normed_array = norm_array(array)
    return array / normed_array
def norm_array(array):
    normed_array = np.zeros_like(array)
    for i in range(len(array)):
        normed_array[i,:] = np.linalg.norm(array[i])
    return normed_array[:,0]
def diffa(array):
    d = np.diff(array)
    d = np.hstack( (d[0], d) )
    return d
    

###
def iseven(n):
    if int(n)/2.0 == int(n)/2:
        return True
    else:
        return False 
def isodd(n):
    if int(n)/2.0 == int(n)/2:
        return False
    else:
        return True
        
###
def in_range(v, okrange):
    if v > np.min(okrange) and v < np.max(okrange):
        return True
    else:
        return False 
        
###
def interpolate_nan(Array):
    if True in np.isnan(Array):
        array = copy.copy(Array)
        for i in range(2,len(array)):
            if np.isnan(array[i]).any():
                array[i] = array[i-1]
        return array
    else:
        return Array
    
###
def remove_angular_rollover(A, max_change_acceptable):
    array = copy.copy(A)
    for i, val in enumerate(array):
        if i == 0:
            continue
        diff = array[i] - array[i-1]
        if np.abs(diff) > max_change_acceptable:
            factor = np.round(np.abs(diff)/(np.pi))  
            if iseven(factor):
                array[i] -= factor*np.pi*np.sign(diff)
    if len(A) == 2:
        return array[1]
    else:
        return array
        
###
def fix_angular_rollover(a):   
    A = copy.copy(a)
    if type(A) is list or type(A) is np.ndarray or type(A) is np.array:
        for i, a in enumerate(A):
            while np.abs(A[i]) > np.pi:
                A[i] -= np.sign(A[i])*(2*np.pi)
        return A
    else:
        while np.abs(A) > np.pi:
                A -= np.sign(A)*(2*np.pi)
        return A
        
###
def dist_point_to_line(pt, linept1, linept2, sign=False):
    # from wolfram mathworld
    x1 = linept1[0]
    x2 = linept2[0]
    y1 = linept1[1]
    y2 = linept2[1]
    x0 = pt[0]
    y0 = pt[1]
    
    if sign:
        d = -1*((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1) )  / np.sqrt( (x2-x1)**2+(y2-y1)**2)
    else:
        d = np.abs( (x2-x1)*(y1-y0)-(x1-x0)*(y2-y1) ) / np.sqrt( (x2-x1)**2+(y2-y1)**2 )
    
    return d
        
        
###
def dist_to_curve(pt, xdata, ydata):
    
    #print 'ONLY WORKS WITH VERY HIGH RESOLUTION DATA'
        
    curve = np.hstack((xdata, ydata)).reshape(len(xdata),2)
    ptarr = pt.reshape(1,2)*np.ones_like(curve)
    
    # rough:
    xdist = curve[:,0] - ptarr[:,0]
    ydist = curve[:,1] - ptarr[:,1]
    hdist = np.sqrt(xdist**2 + ydist**2)
    
    # get sign
    type1_y = np.interp(pt[0], xdata, ydata)
    sign = np.sign(type1_y - pt[1])
    
    return sign*hdist
