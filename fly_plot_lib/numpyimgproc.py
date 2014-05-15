import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import scipy.optimize
from scipy.ndimage.measurements import center_of_mass
import scipy.ndimage as ndimage
import scipy.interpolate
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import binary_fill_holes



import copy

inf = np.inf

###############################################################################
# Basic Image Processing
###############################################################################

def in_range(val, rang):
    if val > rang[0] and val < rang[1]:
        return True
    else:
        return False
    
def threshold(img, threshold_lo, threshold_hi=255):
    threshed_lo = img>=threshold_lo
    threshed_hi = img<=threshold_hi
    threshed = threshed_lo*threshed_hi
    return threshed
    
def absdiff(a,b):
    img = np.array(a, dtype=float)
    bkg = np.array(b, dtype=float)
    diff = img-bkg
    absdiff = np.abs(diff)
    return absdiff
    
def compare(a, b=None, method=None):

    if type(a) is list:
        result = a[0]
        for i in range(1,len(a)):
            if method == 'lighten':
                result = np.maximum(result,a[i])
            elif method == 'darken':
                result = np.minimum(result,a[i])
        return result
    elif b is not None:
        if method is 'lighten':
            result = np.maximum(a,b)
        elif method is 'darken':
            result = np.minimum(a,b)
        return result
    else:
        ValueError('please enter a valid a,b pair')
        
def darken(a, b=None):
    result = compare(a,b,method='darken')
    return result
def lighten(a, b=None):
    result = compare(a,b,method='lighten')
    return result
    
def auto_adjust_levels(img):
    img2 = (img-img.min())
    if img2.max() > 0:
        img3 = img2*int(255/float(img2.max()))
    else:
        img3 = img2
    return img3
    

def get_ellipse_longaxis(img):
    center = center_of_blob(img)
    ptsT = np.transpose(np.nonzero(img))
    for pt in ptsT:
        pt -= center
    
def get_ellipse_cov(img, erode=False, recenter=True):
    # Pattern. Recogn. 20, Sept. 1998, pp. 31-40
    # J. Prakash, and K. Rajesh
    # Human Face Detection and Segmentation using Eigenvalues of Covariance Matrix, Hough Transform and Raster Scan Algorithms

    #eroded_img = binary_erosion(img)
    #boundary = img-eroded_img
    
    if img is not None:
    
        if erode is not False:
            try:
                e = 0
                while e < erode:
                    e += 1
                    img = binary_erosion(img)
            except:
                pass
                
        img = binary_fill_holes(img)
                
        if recenter:
            center = center_of_blob(img)
        else:
            center = np.array([0,0])

        if 1:
            ptsT = np.transpose(np.nonzero(img))
            for pt in ptsT:
                pt -= center
            pts = (ptsT).T
            cov = np.cov(pts)
            cov = np.nan_to_num(cov)
            
            e,v = np.linalg.eig(cov)
            
            longaxis = v[:,np.argmax(e)]
            shortaxis = v[:,np.argmin(e)]
            
            
            if len(ptsT) > 2:
                dl = [np.dot(longaxis, ptsT[i]) for i in range(len(ptsT))]
                longaxis_radius = np.max( np.abs(dl) )
                
                ds = [np.dot(shortaxis, ptsT[i]) for i in range(len(ptsT))]
                shortaxis_radius = np.max( np.abs(ds) )
            else:
                longaxis_radius = None
                shortaxis_radius = None
                
        if recenter is False:
            return longaxis, shortaxis, [longaxis_radius, shortaxis_radius]
        else:
            return center, longaxis, shortaxis, [longaxis_radius, shortaxis_radius]
            
    else:
        return [0,0],0
        
def rebin( a, newshape ):
        '''Rebin an array to a new shape.
        '''
        assert len(a.shape) == len(newshape)

        slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
        coordinates = scipy.mgrid[slices]
        indices = coordinates.astype('i')   #choose the biggest smaller integer index
        return a[tuple(indices)]
        
def extract_uimg(img, size, zero):  
    return img[ zero[0]:zero[0]+size[0], zero[1]:zero[1]+size[1] ]
    
def rotate_image(img, rot):
    
    imgrot = np.zeros_like(img)
    
    for r in range(img.shape[0]):
        for w in range(img.shape[1]):
            ptrot = np.dot(rot, np.array([r,w]))
            rrot = ptrot[0]
            wrot = ptrot[1]
            if rrot < 0:
                rrot += img.shape[0]
            if wrot < 0:
                wrot += img.shape[1]
            imgrot[rrot, wrot] = img[r,w]
    return imgrot
    
def find_circle(img, npts=25, nstart=0, navg=20, plot=False):

    filled_img = binary_fill_holes(img)
    dil_img = binary_dilation(filled_img)
    edges = dil_img-filled_img
    pts = np.transpose(np.nonzero(edges))
    
    # select an evenly spaced subset of points (to speed up computation):
    if len(pts) > npts:
        indices = np.linspace(nstart, len(pts)-1, npts)
        indices = [int(indices[i]) for i in range(len(indices))]        
    else:
        indices = np.arange(0, len(pts), 1).tolist()
    pts_subset = pts[indices,:]
    
    len_pts_diff = np.arange(1,len(pts_subset), 1)
    
    pts_diff = np.zeros(np.sum(len_pts_diff))
    pts_diff_arr = np.zeros([np.sum(len_pts_diff), 2])
    
    iarr = 0
    for i in range(len(pts_subset)):
        indices = np.arange(i+1, len(pts_subset), 1)
        pts_diff_arr[iarr:len(indices)+iarr, 1] = indices
        pts_diff_arr[iarr:len(indices)+iarr, 0] = np.ones_like(indices)*i
        
        d_arr = pts_subset[indices.tolist(), :] - pts_subset[i,:]
        d = np.array( [np.linalg.norm(d_arr[n]) for n in range(len(d_arr))] )
        pts_diff[iarr:len(indices)+iarr] = d
        
        iarr += len(indices)
        
    ordered_pairs = np.argsort(pts_diff)[::-1]
    best_pairs = pts_diff_arr[(ordered_pairs[0:navg]).tolist()]
    
    center_arr = np.zeros([len(best_pairs), 2])
    radius_arr = np.zeros([len(best_pairs), 1])
    
    #centers = np.zeros([len(best_pairs)])
    for i, pair in enumerate(best_pairs):
        pt1 = np.array(pts_subset[ pair[0] ], dtype=float)
        pt2 = np.array(pts_subset[ pair[1] ], dtype=float)
    
        pt_diff = pt2 - pt1
        radius_arr[i] = np.linalg.norm( pt_diff ) / 2.
        center_arr[i] = pt1 + pt_diff/2
        
    center = np.mean(center_arr, axis=0)
    radius = np.mean(radius_arr)
    
    if plot:
        fig = plt.figure(None)
        ax = fig.add_axes([.1,.1,.8,.8])
        circle = patches.Circle( center, radius=radius, facecolor='none', edgecolor='green')
        ax.add_artist(circle)
        ax.imshow(edges)
        
    return center, radius
    
    
###############################################################################
# Misc Geometry Functions
###############################################################################

def plot_circle(xmesh, ymesh, center, r):
    center = list(center)
    
    def in_circle(x,y):
        R = np.sqrt((X-center[1])**2 + (Y-center[0])**2)
        Z = R<r
        return Z

    x = np.arange(0, xmesh, 1)
    y = np.arange(0, ymesh, 1)
    X,Y = np.meshgrid(x, y)

    Z = in_circle(X, Y)
    
    return Z
    
    
###############################################################################
# Blob Manipulations
###############################################################################


def find_blobs(img, sizerange=[0,inf], aslist=True, dilate=False, erode=False):

    if dilate:
        for i in range(dilate):
            img = binary_dilation(img)
    if erode:
        for i in range(erode):
            img = binary_erosion(img)  
        
    blobs, nblobs = ndimage.label(img)
    blob_list = []
    if nblobs < 1:
        if aslist is False:
            return np.zeros_like(img)
        else:
            return [np.zeros_like(img)]
    #print 'n blobs: ', nblobs
    # erode filter
    n_filtered = 0
    for n in range(1,nblobs+1):
        blob_size = (blobs==n).sum()
        if not in_range(blob_size, sizerange):
            blobs[blobs==n] = 0
            nblobs -= 1
        else:
            if aslist:
                b = np.array(blobs==n, dtype=np.uint8)
                blob_list.append(b)
            else:
                n_filtered += 1
                blobs[blobs==n] = n_filtered
    
    if aslist is False:
        if nblobs < 1:
            return np.zeros_like(img)
        else:
            blobs = np.array( blobs, dtype=np.uint8)
            return blobs
    else:
        if len(blob_list) < 1:
            blob_list = [np.zeros_like(img)]
        return blob_list
        
def find_blob_nearest_to_point(img, pt):
    if type(img) == list:
        blobs = []
        for im in img:
            blobs.append( find_blobs(im, sizerange=[0,inf], aslist=False, dilate=False, erode=False) )
    else:
        img = np.array(img)
        blobs = find_blobs(img, sizerange=[0,inf], aslist=True, dilate=False, erode=False)
    centers = center_of_blob(blobs)
    errs = np.zeros(len(centers))
    for i, center in enumerate(centers):
        errs[i] = np.linalg.norm(center - pt)    
    nearest_index = np.argmin(errs)
    return blobs[nearest_index]
    
def find_biggest_blob(img):
    blobs, nblobs = ndimage.label(img)
    
    if nblobs < 1:
        return None
    
    if nblobs == 1:
        return blobs

    biggest_blob_size = 0
    biggest_blob = None
    for n in range(1,nblobs+1):
        blob_size = (blobs==n).sum()
        if blob_size > biggest_blob_size:
            biggest_blob = blobs==n
            biggest_blob_size = blob_size
    biggest_blob = np.array(biggest_blob, dtype=np.uint8)
    return biggest_blob 
    
def center_of_blob(img):
    if type(img) is list:
        centers = []
        for blob in img:
            center = np.array([center_of_mass(blob)[i] for i in range(1,len(blob.shape))])
            centers.append(center)
        return centers
    else:
        center = np.array([center_of_mass(img)[i] for i in range(0, len(img.shape))])
        return center
        
###############################################################################
# Background Subtraction
###############################################################################

'''
mask = binary array, 1 and 0
'''

def get_uimg( img_roi, relative_center, uimg_roi_radius ):
    row_lo = np.max( [int(round(relative_center[0]))-uimg_roi_radius, 0] )
    row_hi = np.min( [int(round(relative_center[0]))+uimg_roi_radius, img_roi.shape[0]] )
    col_lo = np.max( [int(round(relative_center[1]))-uimg_roi_radius, 0] )
    col_hi = np.min( [int(round(relative_center[1]))+uimg_roi_radius, img_roi.shape[1]] )
    
    uimg = copy.copy(img_roi[row_lo:row_hi, col_lo:col_hi])
    relative_zero = np.array([row_lo, col_lo])
    
    return uimg, relative_zero

def find_object_with_background_subtraction(img, background, mask=None, guess=None, guess_radius=None, sizerange=[0,inf], thresh=10, uimg_roi_radius=30, return_uimg=True, return_mimg=False):

    if guess is not None:
        if True in np.isnan(np.array(guess)):
            guess = None
            guess_radius = None
        else:
            guess = np.array(np.round(guess), dtype=int)
            original_guess = copy.copy(guess)
    else:
        guess_radius = None
        
    if guess_radius is not None:
        guess_radius = int(guess_radius)
        
    # explore the full image for objects
    if guess_radius is None:
        img_roi = img
        diff = absdiff(img, background)
        if mask is not None:
            diff *= mask
        zero = np.array([0,0])
    
    # explore just the guessed ROI: MUCH MORE EFFICIENT!!
    if guess_radius is not None:
        row_lo = np.max( [guess[0]-guess_radius, 0] )
        row_hi = np.min( [guess[0]+guess_radius, img.shape[0]] )
        col_lo = np.max( [guess[1]-guess_radius, 0] )
        col_hi = np.min( [guess[1]+guess_radius, img.shape[1]] )
        img_roi = img[row_lo:row_hi, col_lo:col_hi]
        background_roi = background[row_lo:row_hi, col_lo:col_hi]
        diff = absdiff(img_roi, background_roi)
        zero = np.array([row_lo, col_lo])
        guess = np.array([ (row_hi-row_lo)/2. , (col_hi-col_lo)/2. ])
        
    thresh_adj = 0
    blob = []
    #print 'blob: '
    while np.sum(blob) <= 0: # use while loop to force finding an object
        diffthresh = threshold(diff, thresh+thresh_adj, threshold_hi=255)*255
        
        # find blobs
        if guess is not None:
            blobs = find_blobs(diffthresh, sizerange=sizerange, aslist=True)
            if len(blobs) > 1:
                blob = find_blob_nearest_to_point(blobs, guess)
                #print '*0'
            else:
                blob = blobs[0]
                #print '*1'
        else:
            blob = find_biggest_blob(diffthresh)
            #print '*2'
            
        thresh_adj -= 1
        if thresh_adj+thresh <= 0: # failed to find anything at all!!
            center = original_guess
            print 'failed to find object!!!'
            print center, zero
            
            if return_uimg:
                uimg, relative_zero = get_uimg( img_roi, center-zero, uimg_roi_radius )
                zero += relative_zero
                center = copy.copy(center)
                zero = copy.copy(zero)
                if return_mimg is False:
                    return center, uimg, zero
                else:
                    mimg = copy.copy(img_roi)
                    return center, uimg, zero, mimg
            else:
                center = copy.copy(center)
                return center    
    
    #print '******', blob.shape
    relative_center = center_of_blob(blob)
    #print relative_center, zero
    center = relative_center + zero
    
    # find a uimg
    if return_uimg:
        uimg, relative_zero = get_uimg( img_roi, relative_center, uimg_roi_radius )
        zero += relative_zero
        center = copy.copy(center)
        zero = copy.copy(zero)
        if return_mimg is False:
            return center, uimg, zero
        else:
            mimg = copy.copy(img_roi)
            return center, uimg, zero, mimg
    else:
        center = copy.copy(center)
        return center
        
        
def find_object(img, background=None, threshrange=[1,254], sizerange=[10,400], dist_thresh=10, erode=False, check_centers=False, autothreshpercentage=None):
    if background is not None:
        diff = absdiff(img, background)
    else:
        diff = img
    #print '**shape diff** ', diff.shape
    #imgadj = auto_adjust_levels(diff)
    img = diff
    
    if autothreshpercentage is not None:
        imgshaped = np.reshape(img, [np.product(img.shape)])
        nthpixel = int(autothreshpercentage*len(imgshaped))
        threshmax = np.sort(imgshaped)[nthpixel]
        threshmin = np.min(imgshaped)-1
        threshrange = [threshmin, threshmax]
    
    body = threshold(img, threshrange[0], threshrange[1])*255
    
    if erode is not False:
        for i in range(erode):
            body = binary_erosion(body)
                    
    if check_centers is False:
        blobs = find_blobs(body, sizerange=sizerange, aslist=False)
    else:
        blobs = find_blobs(body, sizerange=sizerange, aslist=True)
    body = blobs
    
    if check_centers:
        centers = center_of_blob(blobs)
        dist = []
        for center in centers:
            diff = np.linalg.norm( center - np.array(img.shape)/2. )
            dist.append(diff)
        body = np.zeros_like(img)
        for j, d in enumerate(dist):
            if d < dist_thresh:
                body += blobs[j]
                
    if body.max() > 1:
        body /= body.max()
    
    if body is None:
        body = np.zeros_like(img)
    
    body = np.array(body*255, dtype=np.uint8)
    
    return body


def find_ellipse(img, background=None, threshrange=[1,254], sizerange=[10,400], dist_thresh=10, erode=False, check_centers=False, autothreshpercentage=None, show=False):
    
    #print '**img shape** ', img.shape
    body = find_object(img, background=background, threshrange=threshrange, sizerange=sizerange, dist_thresh=dist_thresh, erode=erode, check_centers=check_centers, autothreshpercentage=autothreshpercentage)

    if body.sum() < 1 and check_centers==True:
        body = find_object(img, background=background, threshrange=threshrange, sizerange=sizerange, dist_thresh=dist_thresh, erode=erode, check_centers=check_centers, autothreshpercentage=autothreshpercentage)
        
    body = binary_fill_holes(body)
    
    if body.sum() < 1:
        body[body.shape[0] / 2, body.shape[1] / 2] = 1
    
    center, longaxis, shortaxis, ratio = get_ellipse_cov(body, erode=False, recenter=True)
    
    
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)
        
        circle = patches.Circle((center[1], center[0]), 2, facecolor='white', edgecolor='none')
        ax.add_artist(circle)
        ax.plot([center[1]-longaxis[1]*ratio[0], center[1]+longaxis[1]*ratio[0]], [center[0]-longaxis[0]*ratio[0], center[0]+longaxis[0]*ratio[0]], zorder=10, color='white')
    
    
    return center, longaxis, shortaxis, body, ratio
    
    
