# written by Floris van Breugel, with some help from Andrew Straw and Will Dickson
# dependencies for LaTex rendering: texlive, ghostscript, dvipng, texlive-latex-extra

# general imports
import matplotlib
print matplotlib.__version__
print 'recommended version: 1.1.1 or greater'


###################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# used for colorline
from matplotlib.collections import LineCollection

# used in histogram
from scipy.stats import norm as gaussian_distribution
from scipy import signal

# used in colorbar
import matplotlib.colorbar

# used in scatter
from matplotlib.collections import PatchCollection



# not used
#import scipy.optimize
#import scipy.stats.distributions as distributions

###################################################################################################
# Misc Info
###################################################################################################

# FUNCTIONS contained in this file: 
# adjust_spines
# colorline
# histogram
# histogram2d (heatmap)
# boxplot
# colorbar (scale for colormap stuff), intended for just generating a colorbar for use in illustrator figure assembly


# useful links:
# colormaps: http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps




###################################################################################################
# Adjust Spines (Dickinson style, thanks to Andrew Straw)
###################################################################################################

# NOTE: smart_bounds is disabled (commented out) in this function. It only works in matplotlib v >1.
# to fix this issue, try manually setting your tick marks (see example below) 
def adjust_spines(ax,spines, spine_locations={}, smart_bounds=True, xticks=None, yticks=None):
    if type(spines) is not list:
        spines = [spines]
        
    # get ticks
    if xticks is None:
        xticks = ax.get_xticks()
    if yticks is None:
        yticks = ax.get_yticks()
        
    spine_locations_dict = {'top': 10, 'right': 10, 'left': 10, 'bottom': 10}
    for key in spine_locations.keys():
        spine_locations_dict[key] = spine_locations[key]
        
    if 'none' in spines:
        for loc, spine in ax.spines.iteritems():
            spine.set_color('none') # don't draw spine
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        return
    
    for loc, spine in ax.spines.iteritems():
        if loc in spines:
            spine.set_position(('outward',spine_locations_dict[loc])) # outward by x points
            spine.set_color('black')
        else:
            spine.set_color('none') # don't draw spine
            
    # smart bounds, if possible
    if int(matplotlib.__version__[0]) > 0 and smart_bounds: 
        for loc, spine in ax.spines.items():
            if loc in ['left', 'right']:
                ticks = yticks
            if loc in ['top', 'bottom']:
                ticks = xticks
            spine.set_bounds(ticks[0], ticks[-1])

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    if 'top' in spines:
        ax.xaxis.set_ticks_position('top')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])    
    
    if 'left' in spines or 'right' in spines:
        ax.set_yticks(yticks)
    if 'top' in spines or 'bottom' in spines:
        ax.set_xticks(xticks)
    
    for line in ax.get_xticklines() + ax.get_yticklines():
        #line.set_markersize(6)
        line.set_markeredgewidth(1)
                
###################################################################################################
# Colorline
###################################################################################################

# plot a line in x and y with changing colors defined by z, and optionally changing linewidths defined by linewidth
def colorline(ax, x,y,z,linewidth=1, colormap='jet', norm=None, zorder=1, alpha=1, linestyle='solid'):
        cmap = plt.get_cmap(colormap)
        
        if type(linewidth) is list or type(linewidth) is np.array or type(linewidth) is np.ndarray:
            linewidths = linewidth
        else:
            linewidths = np.ones_like(z)*linewidth
        
        if norm is None:
            norm = plt.Normalize(np.min(z), np.max(z))
        else:
            norm = plt.Normalize(norm[0], norm[1])
        
        '''
        if self.hide_colorbar is False:
            if self.cb is None:
                self.cb = matplotlib.colorbar.ColorbarBase(self.ax1, cmap=cmap, norm=norm, orientation='vertical', boundaries=None)
        '''
            
        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be numlines x points per line x 2 (x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create the line collection object, setting the colormapping parameters.
        # Have to set the actual values used for colormapping separately.
        lc = LineCollection(segments, linewidths=linewidths, cmap=cmap, norm=norm, zorder=zorder, alpha=alpha, linestyles=linestyle )
        lc.set_array(z)
        lc.set_linewidth(linewidth)
        
        ax.add_collection(lc)

###################################################################################################
# Colorline with heading
###################################################################################################

def get_wedges_for_heading_plot(x, y, color, orientation, size_radius=0.1, size_angle=20, colormap='jet', colornorm=None, size_radius_range=(0.01,.1), size_radius_norm=None, edgecolor='none', alpha=1, flip=True, deg=True, nskip=0, center_offset_fraction=0.75):
    '''
    Returns a Patch Collection of Wedges, with arbitrary color and orientation
    
    Outputs:
    Patch Collection
    
    Inputs:
    x, y        - x and y positions (np.array or list, each of length N)
    color       - values to color wedges by (np.array or list, length N), OR color string. 
       colormap - specifies colormap to use (string, eg. 'jet')
       norm     - specifies range you'd like to normalize to, 
                  if none, scales to min/max of color array (2-tuple, eg. (0,1) )
    orientation - angles are in degrees, use deg=False to convert radians to degrees
    size_radius - radius of wedge, in same units as x, y. Can be list or np.array, length N, for changing sizes
       size_radius_norm - specifies range you'd like to normalize size_radius to, if size_radius is a list/array
                  should be tuple, eg. (0.01, .1)
    size_angle  - angular extent of wedge, degrees. Can be list or np.array, length N, for changing sizes
    edgecolor   - color for lineedges, string or np.array of length N
    alpha       - transparency (single value, between 0 and 1)
    flip        - flip orientations by 180 degrees, default = True
    nskip       - allows you to skip between points to make the points clearer, nskip=1 skips every other point
    center_offset_fraction  - (float in range (0,1) ) - 0 means (x,y) is at the tip, 1 means (x,y) is at the edge
    '''
    cmap = plt.get_cmap(colormap)
    
    # norms
    if colornorm is None and type(color) is not str:
        colornorm = plt.Normalize(np.min(color), np.max(color))
    elif type(color) is not str:
        colornorm = plt.Normalize(colornorm[0], colornorm[1])
    if size_radius_norm is None:
        size_radius_norm = plt.Normalize(np.min(size_radius), np.max(size_radius), clip=True)
    else:
        size_radius_norm = plt.Normalize(size_radius_norm[0], size_radius_norm[1], clip=True)
        
    indices_to_plot = np.arange(0, len(x), nskip+1)
        
    # fix orientations
    if type(orientation) is list:
        orientation = np.array(orientation)
    if deg is False:
        orientation = orientation*180./np.pi
    if flip:
        orientation += 180
    
    flycons = []
    n = 0
    for i in indices_to_plot:
        # wedge parameters
        if type(size_radius) is list or type(size_radius) is np.array or type(size_radius) is np.ndarray: 
            r = size_radius_norm(size_radius[i])*(size_radius_range[1]-size_radius_range[0]) + size_radius_range[0] 
        else: r = size_radius
        
        if type(size_angle) is list or type(size_angle) is np.array or type(size_angle) is np.ndarray: 
            angle_swept = size_radius[i]
        else: angle_swept = size_radius
        theta1 = orientation[i] - size_angle/2.
        theta2 = orientation[i] + size_angle/2.
        
        center = [x[i], y[i]]
        center[0] -= np.cos(orientation[i]*np.pi/180.)*r*center_offset_fraction
        center[1] -= np.sin(orientation[i]*np.pi/180.)*r*center_offset_fraction
        
        wedge = patches.Wedge(center, r, theta1, theta2)
        flycons.append(wedge)
        
    # add collection and color it
    pc = PatchCollection(flycons, cmap=cmap, norm=colornorm)
    
    # set properties for collection
    pc.set_edgecolors(edgecolor)
    if type(color) is list or type(color) is np.array or type(color) is np.ndarray:
        if type(color) is list:
            color = np.asarray(color)
        pc.set_array(color[indices_to_plot])
    else:
        pc.set_facecolors(color)
    pc.set_alpha(alpha)
    
    return pc

def colorline_with_heading(ax, x, y, color, orientation, size_radius=0.1, size_angle=20, colormap='jet', colornorm=None, size_radius_range=(0.01,.1), size_radius_norm=None, edgecolor='none', alpha=1, flip=True, deg=True, nskip=0, use_center='center', show_centers=True, center_offset_fraction=0.75, center_point_size=2):
    '''
    Plots a trajectory with colored wedge shapes to indicate orientation. 
    See function get_wedges_for_heading_plot for details
    
    Additional options:
    
    show_centers      - (bool) - show a black dot where the actual point is - shows where the center of the wedge is 
    center_point_size - markersize for center, if show_centers
    '''
        
    pc = get_wedges_for_heading_plot(x, y, color, orientation, size_radius=size_radius, size_angle=size_angle, colormap=colormap, colornorm=colornorm, size_radius_range=size_radius_range, size_radius_norm=size_radius_norm, edgecolor=edgecolor, alpha=alpha, flip=flip, deg=deg, nskip=nskip, center_offset_fraction=center_offset_fraction)
        
    ax.add_collection(pc)
    
    if show_centers:
        indices_to_plot = np.arange(0, len(x), nskip+1)
        ax.plot(x[indices_to_plot],y[indices_to_plot],'.', color='black', markersize=center_point_size)
        
###################################################################################################
# Histograms
###################################################################################################
    
# first some helper functions
def custom_hist_rectangles(hist, leftedges, width, bottomedges=None, facecolor='green', edgecolor='none', alpha=1, alignment='vertical'):
    linewidth = 1
    if edgecolor == 'none':
        linewidth = 0 # hack needed to remove edges in matplotlib.version 1.0+
        
    if bottomedges is None:
        bottomedges = np.zeros_like(leftedges)

    if type(width) is not list:
        width = [width for i in range(len(hist))]
    rects = [None for i in range(len(hist))]
    
    if alignment == 'vertical':
        for i in range(len(hist)):
            rects[i] = patches.Rectangle( [leftedges[i], bottomedges[i]], width[i], hist[i], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
    elif alignment == 'horizontal':
        for i in range(len(hist)):
            rects[i] = patches.Rectangle( [bottomedges[i], leftedges[i]], hist[i], width[i], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
    
    return rects

def bootstrap_histogram(xdata, bins, normed=False, n=None, return_raw=False):
    if type(xdata) is not np.ndarray:
        xdata = np.array(xdata)

    if n is None:  
        n = len(xdata)
    else:
        if n > len(xdata):
            n = len(xdata)
    hist_list = np.zeros([n, len(bins)-1])
    
    for i in range(n):
        # Choose #sample_size members of d at random, with replacement
        choices = np.random.random_integers(0, len(xdata)-1, n)
        xsample = xdata[choices]
        hist = np.histogram(xsample, bins, normed=normed)[0].astype(float)
        hist_list[i,:] = hist
        
    hist_mean = np.mean(hist_list, axis=0)
    hist_std = np.std(hist_list, axis=0)
    
    print 'bootstrapped: '
    print
    print hist_mean
    print
    print hist_std
    
    if return_raw:
        return hist_list
    else:
        return hist_mean, hist_std
        
    
def histogram(ax, data_list, bins=10, bin_width_ratio=0.6, colors='green', edgecolor='none', bar_alpha=0.7, curve_fill_alpha=0.4, curve_line_alpha=0.8, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=False, normed_occurences=False, bootstrap_std=False, bootstrap_line_width=0.5, exponential_histogram=False, smoothing_range=None, smoothing_bins_to_exclude=[], binweights=None, n_bootstrap_samples=None, alignment='vertical'):
    '''
    ax          -- matplotlib axis
    data_list   -- list of data collections to histogram - if just one, either give an np.array, or soemthing like [data], where data is a list itself
    '''
    # smoothing_range: tuple or list or sequence, eg. (1,100). Use if you only want to smooth and show smoothing over a specific range
    
    if type(bar_alpha) is not list:
        bar_alpha = [bar_alpha for i in range(len(colors))]
    
    n_bars = float(len(data_list))
    if type(bins) is int:
    
        mia = np.array([np.min(d) for d in data_list])
        maa = np.array([np.max(d) for d in data_list])
        
        bins = np.linspace(np.min(mia), np.max(maa), bins, endpoint=True)
        
    if type(colors) is not list:
        colors = [colors]
    if len(colors) != n_bars:
        colors = [colors[0] for i in range(n_bars)]
        
    bin_centers = np.diff(bins)/2. + bins[0:-1]
    bin_width = np.mean(np.diff(bins))
    bin_width_buff = (1-bin_width_ratio)*bin_width/2.
    bar_width = (bin_width-2*bin_width_buff)/n_bars
    
    butter_b, butter_a = signal.butter(curve_butter_filter[0], curve_butter_filter[1])
    
    data_hist_list = []
    if return_vals:
        data_curve_list = []
        data_hist_std_list = []
        
    # first get max number of occurences
    max_occur = []
    for i, data in enumerate(data_list):
        data_hist = np.histogram(data, bins=bins, normed=normed, weights=None)[0].astype(float)
        max_occur.append(np.max(data_hist))
    max_occur = np.max(np.array(max_occur))
        
    for i, data in enumerate(data_list):
        
        if bootstrap_std:
            data_hist, data_hist_std = bootstrap_histogram(data, bins=bins, normed=normed, n=n_bootstrap_samples)
        else:
            data_hist = np.histogram(data, bins=bins, normed=normed)[0].astype(float)
            
        if binweights is not None:
            data_hist *= binweights[i]
            if normed:
                data_hist /= np.sum(binweights[i])
            
        if exponential_histogram:
            data_hist = np.log(data_hist+1)
            
        if normed_occurences is not False:
            if normed_occurences == 'total':
                data_hist /= max_occur 
                if bootstrap_std:
                    data_hist_std /= max_occur
            else:
                div = float(np.max(data_hist))
                print div
                data_hist /= div 
                if bootstrap_std:
                    data_hist_std /= div
                    
        
        rects = custom_hist_rectangles(data_hist, bins[0:-1]+bar_width*i+bin_width_buff, width=bar_width, facecolor=colors[i], edgecolor=edgecolor, alpha=bar_alpha[i], alignment=alignment)
        if bootstrap_std:
            for j, s in enumerate(data_hist_std):
                x = bins[j]+bar_width*i+bin_width_buff + bar_width/2.
                #ax.plot([x,x], [data_hist[j], data_hist[j]+data_hist_std[j]], alpha=1, color='w')
                #ax.plot(np.array([x,x]), np.array([data_hist[j], data_hist[j]+data_hist_std[j]]), alpha=bar_alpha, color=colors[i], linewidth=bootstrap_line_width)
                ax.vlines(x, data_hist[j], data_hist[j]+data_hist_std[j], color=colors[i], linewidth=bootstrap_line_width)
                
                #ax.plot([x-bar_width/3., x+bar_width/3.], [data_hist[j]+data_hist_std[j],data_hist[j]+data_hist_std[j]], alpha=1, color='w')
                #ax.plot([x-bar_width/3., x+bar_width/3.], [data_hist[j]+data_hist_std[j],data_hist[j]+data_hist_std[j]], alpha=bar_alpha, color=colors[i])
        for rect in rects:
            rect.set_zorder(1)
            ax.add_artist(rect)
        
                
        if show_smoothed:
            if smoothing_range is not None: # in case you only want to smooth and show smoothing over a select range.
                indices_in_smoothing_range = np.where( (bin_centers>smoothing_range[0])*(bin_centers<smoothing_range[-1]) )[0].tolist()
            else:
                indices_in_smoothing_range = [bc for bc in range(len(bin_centers))]
                
            for b in smoothing_bins_to_exclude:
                try:
                    indices_in_smoothing_range.remove(b)
                except ValueError:
                    print 'bin center not in indices list: ', b
                    print 'indices list: ', indices_in_smoothing_range
                
            data_hist_filtered = signal.filtfilt(butter_b, butter_a, data_hist[indices_in_smoothing_range])
            interped_bin_centers = np.linspace(bin_centers[indices_in_smoothing_range[0]]-bin_width/2., bin_centers[indices_in_smoothing_range[-1]]+bin_width/2., 100, endpoint=True)
            v = 100 / float(len(bin_centers))
            
            if alignment == 'vertical':
                interped_data_hist_filtered = np.interp(interped_bin_centers, bin_centers[indices_in_smoothing_range], data_hist_filtered)
                interped_data_hist_filtered2 = signal.filtfilt(butter_b/v, butter_a/v, interped_data_hist_filtered)
                #ax.plot(bin_centers, data_hist_filtered, color=facecolor[i])
                if curve_fill_alpha > 0:
                    ax.fill_between(interped_bin_centers, interped_data_hist_filtered2, np.zeros_like(interped_data_hist_filtered2), color=colors[i], alpha=curve_fill_alpha, zorder=-100, edgecolor='none')
                if curve_line_alpha:
                    ax.plot(interped_bin_centers, interped_data_hist_filtered2, color=colors[i], alpha=curve_line_alpha)
        
            if alignment == 'horizontal':
                interped_data_hist_filtered = np.interp(interped_bin_centers, bin_centers[indices_in_smoothing_range], data_hist_filtered)
                interped_data_hist_filtered2 = signal.filtfilt(butter_b/v, butter_a/v, interped_data_hist_filtered)
                #ax.plot(bin_centers, data_hist_filtered, color=facecolor[i])
                if curve_fill_alpha > 0:
                    ax.fill_betweenx(interped_bin_centers, interped_data_hist_filtered2, np.zeros_like(interped_data_hist_filtered2), color=colors[i], alpha=curve_fill_alpha, zorder=-100, edgecolor='none')
                if curve_line_alpha:
                    ax.plot(interped_data_hist_filtered2, interped_bin_centers, color=colors[i], alpha=curve_line_alpha)
        
        
        
        data_hist_list.append(data_hist)
        if return_vals:
            if bootstrap_std:
                data_hist_std_list.append(data_hist_std)
            
            if show_smoothed:
                data_curve_list.append([interped_bin_centers, interped_data_hist_filtered2])
                
    mins_of_data = [np.min(data) for data in data_list]
    maxs_of_data = [np.max(data) for data in data_list]
    
    mins_of_hist = [np.min(hist) for hist in data_hist_list]
    maxs_of_hist = [np.max(hist) for hist in data_hist_list]
    
    if alignment == 'vertical':
        ax.set_xlim(np.min(mins_of_data), np.max(maxs_of_data)) 
        ax.set_ylim(0, np.max(maxs_of_hist))
    elif alignment == 'horizontal':
        ax.set_ylim(np.min(mins_of_data), np.max(maxs_of_data)) 
        ax.set_xlim(0, np.max(maxs_of_hist))
                
    if return_vals and bootstrap_std is False:
        return bins, data_hist_list, data_curve_list
    elif return_vals and bootstrap_std is True:
        return bins, data_hist_list, data_hist_std_list, data_curve_list
    
    
###########

def histogram_stack(ax, data_list, bins=10, bin_width_ratio=0.8, colors='green', edgecolor='none', normed=True):
    '''
    ax          -- matplotlib axis
    data_list   -- list of data collections to histogram - if just one, either give an np.array, or soemthing like [data], where data is a list itself
    normed - normalizes the SUM of all the stacked histograms
    '''
    # smoothing_range: tuple or list or sequence, eg. (1,100). Use if you only want to smooth and show smoothing over a specific range
    
    n_bars = float(len(data_list))
    if type(bins) is int:
        mia = np.array([np.min(d) for d in data_list])
        maa = np.array([np.max(d) for d in data_list])
        bins = np.linspace(np.min(mia), np.max(maa), bins, endpoint=True)
        
    if type(colors) is not list:
        colors = [colors]
    if len(colors) != n_bars:
        colors = [colors[0] for i in range(n_bars)]
        
    bin_centers = np.diff(bins)/2. + bins[0:-1]
    bin_width = np.mean(np.diff(bins))
    bin_width_buff = (1-bin_width_ratio)*bin_width/2.
    bar_width = (bin_width-bin_width_buff)
    
    data_hist_list = []
        
    all_data = []
    for data in data_list:
        all_data.extend(data.tolist())
    all_data_hist = np.histogram(all_data, bins=bins, normed=False)[0].astype(float) 
    all_data_hist_normed = np.histogram(all_data, bins=bins, normed=True)[0].astype(float) 
    binweights_for_normalizing = all_data_hist_normed / all_data_hist
        
    prev_data_hist = np.zeros_like(all_data_hist)
    for i, data in enumerate(data_list):
        
        data_hist = np.histogram(data, bins=bins, normed=False)[0].astype(float)
        
        if normed:
            data_hist *= binweights_for_normalizing
            
        rects = custom_hist_rectangles(data_hist, bins[0:-1]+bin_width_buff, bottomedges=prev_data_hist, width=bar_width, facecolor=colors[i], edgecolor=edgecolor, alpha=1)
        prev_data_hist += data_hist

        for rect in rects:
            rect.set_zorder(1)
            ax.add_artist(rect)
        
        data_hist_list.append(data_hist)
        
    
    ax.set_xlim(bins[0], bins[-1])
    if normed:
        ax.set_ylim(0, np.max(all_data_hist_normed)+.1*np.max(all_data_hist_normed))
    else:
        ax.set_ylim(0, np.max(all_data_hist)+.1*np.max(all_data_hist))
        

###################################################################################################
# Boxplots
###################################################################################################
    
def boxplot(ax, x_data, y_data_list, nbins=50, colormap='YlOrRd', colorlinewidth=2, boxwidth=1, boxlinecolor='black', classic_linecolor='gray', usebins=None, boxlinewidth=0.5, outlier_limit=0.01, norm=None, use_distribution_for_linewidth=False, min_colorlinewidth=1, show_outliers=True, show_whiskers=True, logcolorscale=False, orientation='vertical'):    
    # if colormap is None: show a line instead of the 1D histogram (ie. a normal boxplot)
    # use_distribution_for_linewidth will adjust the linewidth according to the histogram of the distribution
    # classic_linecolor: sets the color of the vertical line that shows the extent of the data, if colormap=None
    # outlier limit: decimal % of top and bottom data that is defined as outliers (0.01 = top 1% and bottom 1% are defined as outliers)
    # show_whiskers: toggle the whiskers, if colormap is None

    if usebins is None: 
        usebins = nbins
        # usebins lets you assign the bins manually, but it's the same range for each x_coordinate

    for i, y_data in enumerate(y_data_list):
        #print len(y_data)

        # calc boxplot statistics
        median = np.median(y_data)
        ind = np.where(y_data<=median)[0].tolist()
        first_quartile = np.median(y_data[ind])
        ind = np.where(y_data>=median)[0].tolist()
        last_quartile = np.median(y_data[ind])
        #print first_quartile, median, last_quartile
        
        # find outliers
        ind_sorted = np.argsort(y_data)
        bottom_limit = int(len(ind_sorted)*(outlier_limit))
        top_limit = int(len(ind_sorted)*(1-outlier_limit))
        indices_inrange = ind_sorted[bottom_limit:top_limit]
        outliers = ind_sorted[0:bottom_limit].tolist() + ind_sorted[top_limit:len(ind_sorted)-1].tolist()
        y_data_inrange = y_data[indices_inrange]
        y_data_outliers = y_data[outliers]
        x = x_data[i]
        
    
        # plot colorline
        if colormap is not None:
            hist, bins = np.histogram(y_data_inrange, usebins, normed=True)
            hist = hist.astype(float)
            #hist /= np.max(hist)
            x_arr = np.ones_like(bins)*x
            
            if logcolorscale:
                hist = np.log(hist+1)
                
            if use_distribution_for_linewidth:
                colorlinewidth = hist*colorlinewidth + min_colorlinewidth
                
            if orientation == 'vertical':
                colorline(ax, x_arr, bins, hist, colormap=colormap, norm=norm, linewidth=colorlinewidth) # the norm defaults make it so that at each x-coordinate the colormap/linewidth will be scaled to show the full color range. If you want to control the color range for all x-coordinate distributions so that they are the same, set the norm limits when calling boxplot(). 
            elif orientation == 'horizontal':
                colorline(ax, bins, x_arr, hist, colormap=colormap, norm=norm, linewidth=colorlinewidth)
                
        elif show_whiskers:
            if orientation == 'vertical':
                ax.vlines(x, last_quartile, np.max(y_data_inrange), color=classic_linecolor, linestyle=('-'), linewidth=boxlinewidth/2.)
                ax.vlines(x, np.min(y_data_inrange), first_quartile, color=classic_linecolor, linestyle=('-'), linewidth=boxlinewidth/2.)
                ax.hlines([np.min(y_data_inrange), np.max(y_data_inrange)], x-boxwidth/4., x+boxwidth/4., color=classic_linecolor, linewidth=boxlinewidth/2.)
            elif orientation == 'horizontal':
                ax.hlines(x, last_quartile, np.max(y_data_inrange), color=classic_linecolor, linestyle=('-'), linewidth=boxlinewidth/2.)
                ax.hlines(x, np.min(y_data_inrange), first_quartile, color=classic_linecolor, linestyle=('-'), linewidth=boxlinewidth/2.)
                ax.vlines([np.min(y_data_inrange), np.max(y_data_inrange)], x-boxwidth/4., x+boxwidth/4., color=classic_linecolor, linewidth=boxlinewidth/2.)
        
        
        # plot boxplot
        if orientation == 'vertical':
            ax.hlines(median, x-boxwidth/2., x+boxwidth/2., color=boxlinecolor, linewidth=boxlinewidth)
            ax.hlines([first_quartile, last_quartile], x-boxwidth/2., x+boxwidth/2., color=boxlinecolor, linewidth=boxlinewidth/2.)
            ax.vlines([x-boxwidth/2., x+boxwidth/2.], first_quartile, last_quartile, color=boxlinecolor, linewidth=boxlinewidth/2.)
        elif orientation == 'horizontal':
            ax.vlines(median, x-boxwidth/2., x+boxwidth/2., color=boxlinecolor, linewidth=boxlinewidth)
            ax.vlines([first_quartile, last_quartile], x-boxwidth/2., x+boxwidth/2., color=boxlinecolor, linewidth=boxlinewidth/2.)
            ax.hlines([x-boxwidth/2., x+boxwidth/2.], first_quartile, last_quartile, color=boxlinecolor, linewidth=boxlinewidth/2.)
        
        # plot outliers
        if show_outliers:
            if outlier_limit > 0:
                x_arr_outliers = x*np.ones_like(y_data_outliers)
                if orientation == 'vertical':
                    ax.plot(x_arr_outliers, y_data_outliers, '.', markerfacecolor='gray', markeredgecolor='none', markersize=1)
                elif orientation == 'horizontal':
                    ax.plot(y_data_outliers, x_arr_outliers, '.', markerfacecolor='gray', markeredgecolor='none', markersize=1)
###################################################################################################
# 2D "heatmap" Histogram
###################################################################################################

def histogram2d(ax, x, y, bins=100, normed=False, histrange=None, weights=None, logcolorscale=False, colormap='jet', interpolation='nearest', colornorm=None, xextent=None, yextent=None, norm_rows=False, norm_columns=False):
    # the following paramters get passed straight to numpy.histogram2d
    # x, y, bins, normed, histrange, weights
    
    # from numpy.histogram2d:
    '''
    Parameters
    ----------
    x : array_like, shape(N,)
      A sequence of values to be histogrammed along the first dimension.
    y : array_like, shape(M,)
      A sequence of values to be histogrammed along the second dimension.
    bins : int or [int, int] or array-like or [array, array], optional
      The bin specification:
    
        * the number of bins for the two dimensions (nx=ny=bins),
        * the number of bins in each dimension (nx, ny = bins),
        * the bin edges for the two dimensions (x_edges=y_edges=bins),
        * the bin edges in each dimension (x_edges, y_edges = bins).
    
    range : array_like, shape(2,2), optional
      The leftmost and rightmost edges of the bins along each dimension
      (if not specified explicitly in the `bins` parameters):
      [[xmin, xmax], [ymin, ymax]]. All values outside of this range will be
      considered outliers and not tallied in the histogram.
    normed : boolean, optional
      If False, returns the number of samples in each bin. If True, returns
      the bin density, ie, the bin count divided by the bin area.
    weights : array-like, shape(N,), optional
      An array of values `w_i` weighing each sample `(x_i, y_i)`. Weights are
      normalized to 1 if normed is True. If normed is False, the values of the
      returned histogram are equal to the sum of the weights belonging to the
      samples falling into each bin.
    '''
    
    hist,x,y = np.histogram2d(x, y, bins, normed=normed, range=histrange, weights=weights)
    
    if logcolorscale:
        hist = np.log(hist+1) # the plus one solves bin=0 issues
        
    if xextent is None:
        xextent = [x[0], x[-1]]
    if yextent is None:
        yextent = [y[0], y[-1]]
    
    img = hist.T
    
    if norm_rows:
        for r in range(img.shape[0]):
            mi = np.min(img[r,:])
            img[r,:] -= mi
            ma = np.max(img[r,:])
            if ma != 0:
                img[r,:] /= ma
            print mi, ma, np.min(img[r,:]), np.max(img[r,:])
    
    if norm_columns:
        for c in range(img.shape[1]):
            mi = np.min(img[:,c])
            ma = np.max(img[:,c])
            img[:,c] -= mi
            img[:,c] /= ma
        
    
    if colornorm is not None:
        colornorm = matplotlib.colors.Normalize(colornorm[0], colornorm[1])
    else:
        colornorm = matplotlib.colors.Normalize(np.min(np.min(img)), np.max(np.max(img)))
        print 'color norm: ', np.min(np.min(img)), np.max(np.max(img))
        
    
    # make the heatmap
    cmap = plt.get_cmap(colormap)
    ax.imshow(  img, 
                cmap=cmap,
                extent=(xextent[0], xextent[1], yextent[0], yextent[1]), 
                origin='lower', 
                interpolation=interpolation,
                norm=colornorm)
    ax.set_aspect('auto')
    
###################################################################################################
# Colorbar
###################################################################################################

def colorbar(ax=None, ticks=None, ticklabels=None, colormap='jet', aspect=20, orientation='vertical', filename=None, flipspine=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if ticks is None:
        ticks = np.linspace(-1,1,5,endpoint=True)
    
    ax.set_aspect('equal')
    
    # horizontal
    if orientation == 'horizontal':
        xlim = (ticks[0],ticks[-1])
        yrange = (ticks[-1]-ticks[0])/float(aspect)
        ylim = (0, yrange)
        grad = np.linspace(ticks[0], ticks[-1], 500, endpoint=True)
        im = np.vstack((grad,grad))
        if not flipspine:
            adjust_spines(ax,['bottom'], xticks=ticks)
        else:
            adjust_spines(ax,['top'], xticks=ticks)
        if ticklabels is not None:
            ax.set_xticklabels(ticklabels)
    
    # vertical
    if orientation == 'vertical':
        ylim = (ticks[0],ticks[-1])
        xrange = (ticks[-1]-ticks[0])/float(aspect)
        xlim = (0, xrange)
        grad = np.linspace(ticks[0], ticks[-1], 500, endpoint=True)
        im = np.vstack((grad,grad)).T
        if not flipspine:
            adjust_spines(ax,['right'], yticks=ticks)
        else:
            adjust_spines(ax,['left'], yticks=ticks)
        if ticklabels is not None:
            ax.set_yticklabels(ticklabels)

    # make image
    cmap = plt.get_cmap(colormap)
    ax.imshow(  im, 
                cmap=cmap,
                extent=(xlim[0], xlim[-1], ylim[0], ylim[-1]), 
                origin='lower', 
                interpolation='bicubic')
                
    if filename is not None:
        fig.savefig(filename, format='pdf')
    
###################################################################################################
# Scatter Plot (with PatchCollections of circles) : more control than plotting with 'dotted' style with "plot"
###################################################################################################

def get_circles_for_scatter(x, y, color='black', edgecolor='none', colormap='jet', radius=0.01, colornorm=None, alpha=1, radiusnorm=None, maxradius=1, minradius=0):
    cmap = plt.get_cmap(colormap)
    if colornorm is not None:
        colornorm = plt.Normalize(colornorm[0], colornorm[1], clip=True)
    
    # setup normalizing for radius scale factor (if used)
    if type(radius) is list or type(radius) is np.array or type(radius) is np.ndarray:
        if radiusnorm is None:
            radiusnorm = matplotlib.colors.Normalize(np.min(radius), np.max(radius), clip=True)
        else:
            radiusnorm = matplotlib.colors.Normalize(radiusnorm[0], radiusnorm[1], clip=True)

    # make circles
    points = np.array([x, y]).T
    circles = [None for i in range(len(x))]
    for i, pt in enumerate(points):    
        if type(radius) is list or type(radius) is np.array or type(radius) is np.ndarray:
            r = radiusnorm(radius[i])*(maxradius-minradius) + minradius
        else:
            r = radius
        circles[i] = patches.Circle( pt, radius=r )

    # make a collection of those circles    
    cc = PatchCollection(circles, cmap=cmap, norm=colornorm) # potentially useful option: match_original=True
    
    # set properties for collection
    cc.set_edgecolors(edgecolor)
    if type(color) is list or type(color) is np.array or type(color) is np.ndarray:
        cc.set_array(color)
    else:
        cc.set_facecolors(color)
    cc.set_alpha(alpha)
    
    return cc
    
def get_ellipses_for_scatter(ax, x, y, color='black', edgecolor='none', colormap='jet', radius=0.01, colornorm=None, alpha=1, radiusnorm=None, maxradius=1, minradius=0):
    
    # get ellipse size to make it a circle given axes
    x0, y0 = ax.transAxes.transform((ax.get_ylim()[0],ax.get_xlim()[0]))
    x1, y1 = ax.transAxes.transform((ax.get_ylim()[1],ax.get_xlim()[1]))
    dx = x1-x0
    dy = y1-y0
    maxd = max(dx,dy)
    
    cmap = plt.get_cmap(colormap)
    if colornorm is not None:
        colornorm = plt.Normalize(colornorm[0], colornorm[1], clip=True)
    
    # setup normalizing for radius scale factor (if used)
    if type(radius) is list or type(radius) is np.array or type(radius) is np.ndarray:
        if radiusnorm is None:
            radiusnorm = matplotlib.colors.Normalize(np.min(radius), np.max(radius), clip=True)
        else:
            radiusnorm = matplotlib.colors.Normalize(radiusnorm[0], radiusnorm[1], clip=True)

    # make circles
    points = np.array([x, y]).T
    ellipses = [None for i in range(len(x))]
    for i, pt in enumerate(points):    
        if type(radius) is list or type(radius) is np.array or type(radius) is np.ndarray:
            r = radiusnorm(radius[i])*(maxradius-minradius) + minradius
        else:
            r = radius
        width = r*2*maxd/dx
        height = r*2*maxd/dy
        ellipses[i] = patches.Ellipse( pt, width, height)

    # make a collection of those circles    
    cc = PatchCollection(ellipses, cmap=cmap, norm=colornorm) # potentially useful option: match_original=True
    
    # set properties for collection
    cc.set_edgecolors(edgecolor)
    if type(color) is list or type(color) is np.array or type(color) is np.ndarray:
        cc.set_array(color)
    else:
        cc.set_facecolors(color)
    cc.set_alpha(alpha)
    
    return cc

def scatter(ax, x, y, color='black', colormap='jet', edgecolor='none', radius=0.01, colornorm=None, alpha=1, radiusnorm=None, maxradius=1, minradius=0, xlim=None, ylim=None, use_ellipses=True): 
    '''
    Make a colored scatter plot
    
    NOTE: the scatter points will only be circles if you use_ellipses=True, and if you do not change xlim/ylim or the relative size of the axes after the function has been called.
    
    x           -- np.array
    y           -- np.array
    color       -- matplotlib color (eg. string), or np.array of values
    colormap    -- matplotlib coloramp name (eg. 'jet')
    edgecolor   -- matplotlib color for edges (eg. string) - default is 'none', which means no edge
    radius      -- radius of circles to plot - in units of the axes - either a float, or np.array of floats
    colornorm   -- min and max you would like colors in the color array normalized to, eg [0,1], default is to scale to min/max of color array
    alpha       -- transparancy, float btwn 0 and 1
    radiusnorm  -- min/max you would like radius array to be normalized to
    maxradius   -- max radius size you would like
    minradius   -- min radius size you would like
    xlim/ylim   -- x and y limits of axes, default is scaled to min/max of x and y
    use_ellipses-- adjust scatter point so that they are circles, even if aspect is not equal. Only works if you do not change xlim/ylim or axes shape after calling this function
    
    '''
    # color can be array-like, or a matplotlib color 
    # I can't figure out how to control alpha through the individual circle patches.. it seems to get overwritten by the collection. low priority!
    
    if xlim is None:
        xlim = [np.min(x), np.max(x)]
    if ylim is None:
        ylim = [np.min(y), np.max(y)]
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if use_ellipses:
        cc = get_ellipses_for_scatter(ax, x, y, color=color, edgecolor=edgecolor, colormap=colormap, radius=radius, colornorm=colornorm, alpha=alpha, radiusnorm=radiusnorm, maxradius=maxradius, minradius=minradius)

    else:
        cc = get_circles_for_scatter(x, y, color=color, edgecolor=edgecolor, colormap=colormap, radius=radius, colornorm=colornorm, alpha=alpha, radiusnorm=radiusnorm, maxradius=maxradius, minradius=minradius)

    # add collection to axis    
    ax.add_collection(cc)  
    
###################################################################################################
# Run examples: lets you see all the example plots!
###################################################################################################
def run_examples():
    adjust_spines_example_with_custom_ticks()
    colorline_example()
    colorline_with_heading_example()
    histogram_example()
    boxplot_example()
    boxplot_classic_example()
    histogram2d_example()
    colorbar_example()
    scatter_example()

if __name__ == '__main__':
    run_examples()



