import fly_plot_lib
fly_plot_lib.set_params.pdf()
fpl = fly_plot_lib.plot
fpl_text = fly_plot_lib.text

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from scipy.stats import norm as gaussian_distribution

####################################################################################
def text_wrapping_example(save=False):
    text = r"I am a super long string, with my top left corner in the middle of the figure. The text got auto-wrapped so it fits inside this cool little box that has a width and height equal to 40$\%$ of the figure size. How cool is that? With tex set to True in the rcparams (a default for the pdf saving in fly plot lib) you can make \textbf{bold text} and \emph{italic} too. Note the ``r'' before the text string that tells matplotlib to use raw text, which ignores things like ``\textbackslash t''. Unfortunately there is some weird behavior with text wrapping and bold and italic stuff. Also, the characters used to tell latex to do stuff get used in the wrapping calculations. Deal with it, I guess."
    
    
    fig = plt.figure()
    
    left = 0.5
    top = 0.5
    width = 0.4
    height = 0.4
    
    fpl_text.text_box(fig, left, top, width, height, text)
    #fig.text(0, 0.5, text)
    plt.draw()
    
    if save:
        fig.savefig('figures/text_wrapping_example.pdf', format='pdf')    

####################################################################################
def adjust_spines_example(save=False):
    
    x = np.linspace(0,100,100)
    y = x**2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y)
    fpl.adjust_spines(ax, ['left', 'bottom'])
    
    if save:
        fig.savefig('figures/adjust_spines_example.pdf', format='pdf')

####################################################################################
def adjust_spines_example_with_custom_ticks(save=False):

    x = np.linspace(0,100,100)
    y = x**2
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y)
    
    # set limits
    ax.set_xlim(0,100)
    ax.set_ylim(0,20000)
    
    # set custom ticks and tick labels
    xticks = [0, 10, 25, 50, 71, 100] # custom ticks, should be a list
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, smart_bounds=True)
    
    ax.set_xlabel('x axis, custom ticks\ncoooool!')
    
    if save:
        fig.savefig('figures/adjust_spines_custom_ticks_example.pdf', format='pdf')

####################################################################################
def colorline_example(save=False):
    
    def tent(x):
        """
        A simple tent map
        """
        if x < 0.5:
            return x
        else:
            return -1.0*x + 1
    
    pi = np.pi
    t = np.linspace(0, 1, 200)
    y = np.sin(2*pi*t)
    z = np.array([tent(x) for x in t]) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # standard colorline
    fpl.colorline(ax,t,y,z)
    
    # colorline with changing widths, shifted in x
    fpl.colorline(ax,t+0.5,y,z,linewidth=z*5)
    
    # colorline with points, shifted in x
    fpl.colorline(ax,t+1,y,z, linestyle='dotted')
    
    # set the axis to appropriate limits
    fpl.adjust_spines(ax, ['left', 'bottom'])
    ax.set_xlim(0,2)
    ax.set_ylim(0,1.5)
    
    if save:
        fig.savefig('figures/colorline_example.pdf', format='pdf')

####################################################################################
def colorline_with_heading_example(save=False):
    
    pi = np.pi
    t = np.linspace(0, np.pi*2, 200)
    y = np.sin(t)
    x = np.cos(t)
    color = x
    orientation = t
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # standard colorline
    fpl.colorline_with_heading(ax,x,y,color,orientation, show_centers=True, nskip=4, center_offset_fraction=0.75, deg=False)
    
    # set the axis to appropriate limits
    fpl.adjust_spines(ax, ['left', 'bottom'])
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_aspect('equal')
    
    if save:
        fig.savefig('figures/colorline_with_orientation_example.pdf', format='pdf')
        
####################################################################################
def colorline_with_heading_and_radius_example(save=False):
    
    pi = np.pi
    t = np.linspace(0, np.pi*2, 200)
    y = np.sin(t)
    x = np.cos(t)
    color = x
    orientation = t
    size_radius = y
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # standard colorline
    fpl.colorline_with_heading(ax,x,y,color,orientation, size_radius=y, size_radius_range=(0.05, .15), show_centers=True, nskip=4, center_offset_fraction=0.75, deg=False)
    
    # set the axis to appropriate limits
    fpl.adjust_spines(ax, ['left', 'bottom'])
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_aspect('equal')
    
    if save:
        fig.savefig('figures/colorline_with_orientation_and_radius_example.pdf', format='pdf')

####################################################################################
def histogram_example(save=False):
    
    # generate a list of various y data, from three random gaussian distributions
    y_data_list = []
    for i in range(3):
        mean = np.random.random()*10
        std = 3
        ndatapoints = 500
        y_data = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)
        y_data_list.append(y_data)
        
    nbins = 40 # note: if show_smoothed=True with default butter filter, nbins needs to be > ~15 
    bins = np.linspace(-10,30,nbins)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fpl.histogram(ax, y_data_list, bins=bins, bin_width_ratio=0.8, colors=['green', 'black', 'orange'], edgecolor='none', bar_alpha=1, curve_fill_alpha=0.4, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=False, exponential_histogram=False)
    
    fpl.adjust_spines(ax, ['left', 'bottom'])
    
    if save:
        fig.savefig('figures/histogram_example.pdf', format='pdf')
        
####################################################################################
def histogram_horizontal_example(save=False):
    
    # generate a list of various y data, from three random gaussian distributions
    y_data_list = []
    for i in range(3):
        mean = np.random.random()*10
        std = 3
        ndatapoints = 500
        y_data = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)
        y_data_list.append(y_data)
        
    nbins = 40 # note: if show_smoothed=True with default butter filter, nbins needs to be > ~15 
    bins = np.linspace(-10,30,nbins)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fpl.histogram(ax, y_data_list, bins=bins, bin_width_ratio=0.8, colors=['green', 'black', 'orange'], edgecolor='none', bar_alpha=1, curve_fill_alpha=0.4, curve_line_alpha=1, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=False, exponential_histogram=False, alignment='horizontal')
    
    fpl.adjust_spines(ax, ['left', 'bottom'])
    
    if save:
        fig.savefig('figures/histogram_horizontal_example.pdf', format='pdf')
        
####################################################################################
def histogram_stack_example(save=False):
    
    # generate a list of various y data, from three random gaussian distributions
    y_data_list = []
    for i in range(3):
        mean = np.random.random()*10
        std = 3
        ndatapoints = 500
        y_data = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)
        y_data_list.append(y_data)
        
    nbins = 40 # note: if show_smoothed=True with default butter filter, nbins needs to be > ~15 
    bins = np.linspace(-10,30,nbins)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fpl.histogram_stack(ax, y_data_list, bins=bins, bin_width_ratio=0.8, colors=['green', 'black', 'orange'], edgecolor='none', normed=True)
    
    fpl.adjust_spines(ax, ['left', 'bottom'])
    
    if save:
        fig.savefig('figures/histogram_stack_example.pdf', format='pdf')    


####################################################################################
def boxplot_example(save=False):
    # box plot with colorline as histogram

    # generate a list of various y data, from three random gaussian distributions
    x_data = np.linspace(0,20,5)
    y_data_list = []
    for i in range(len(x_data)):
        mean = np.random.random()*10
        std = 3
        ndatapoints = 500
        y_data = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)
        y_data_list.append(y_data)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fpl.boxplot(ax, x_data, y_data_list)
    fpl.adjust_spines(ax, ['left', 'bottom'])
    
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    
    if save:
        fig.savefig('figures/boxplot_example.pdf', format='pdf')  
    
####################################################################################
def boxplot_classic_example(save=False):
    # classic boxplot look (no colorlines)
        
    # generate a list of various y data, from three random gaussian distributions
    x_data = np.linspace(0,20,5)
    y_data_list = []
    for i in range(len(x_data)):
        mean = np.random.random()*10
        std = 3
        ndatapoints = 500
        y_data = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)
        y_data_list.append(y_data)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fpl.boxplot(ax, x_data, y_data_list, colormap=None, boxwidth=1, boxlinewidth=0.5, outlier_limit=0.01, show_outliers=True)
    fpl.adjust_spines(ax, ['left', 'bottom'])
    
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    
    if save:
        fig.savefig('figures/boxplot_classic_example.pdf', format='pdf')  

####################################################################################
def histogram2d_example(save=False):  

    # make some random data
    mean = np.random.random()*10
    std = 3
    ndatapoints = 50000
    x = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)
    y = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fpl.histogram2d(ax, x, y, bins=100)
    
    fpl.adjust_spines(ax, ['left', 'bottom'])
    
    if save:
        fig.savefig('figures/histogram2d_example.pdf', format='pdf')

####################################################################################
def colorbar_example(save=True):
    fpl.colorbar(filename='figures/colorbar_example.pdf')

####################################################################################
def scatter_example(save=False):
    
    x = np.random.random(100)
    y = np.random.random(100)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # show a few different scatter examples
    fpl.scatter(ax, x, y, color=x*10, use_ellipses=False) # with color scale
    fpl.scatter(ax, x+1, y+1, color='black', use_ellipses=False) # set fixed color
    fpl.scatter(ax, x+1, y, color='blue', radius=0.05, alpha=0.2, use_ellipses=False) # set some parameters for all circles 
    fpl.scatter(ax, x, y+1, color='green', radius=x, alpha=0.6, radiusnorm=(0.2, 0.8), minradius=0.01, maxradius=0.05, use_ellipses=False) # let radius vary with some array 
    
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.set_aspect('equal')
    fpl.adjust_spines(ax, ['left', 'bottom'])
    
    if save:
        fig.savefig('figures/scatter_example.pdf', format='pdf')
        
def scatter_example_with_ellipses(save=False):
    
    x = np.random.random(100)
    y = np.random.random(100)*10
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # show a few different scatter examples
    fpl.scatter(ax, x, y, color=x*10) # with color scale
    
    fpl.adjust_spines(ax, ['left', 'bottom'])
    
    if save:
        fig.savefig('figures/scatter_example_with_ellipses.pdf', format='pdf')
        
####################################################################################
def example_gridspec(save=False):
    '''
    This is a VERY rough example. Any help on how to actually make this work properly is much appreciated. This is mostly provided for my own reference.
    '''
    figure_padding = 0.25
    subplot_padding = 0.08

    fig = plt.figure(figsize=(8,3.9))

    aspect_ratio = (4+subplot_padding)/(12.+subplot_padding)

    plt.suptitle("GridSpec w/ different subplotpars")

    gs1 = gridspec.GridSpec(2, 2, width_ratios=[10,2])
    gs1.update(left=figure_padding*aspect_ratio, right=1-figure_padding*aspect_ratio, wspace=subplot_padding, hspace=subplot_padding, top=1-figure_padding+subplot_padding, bottom=figure_padding-subplot_padding)
    ax1 = plt.subplot(gs1[0, 0])
    ax2 = plt.subplot(gs1[1, 0])
    ax3 = plt.subplot(gs1[0, 1])
    ax4 = plt.subplot(gs1[1, 1])

    ax1.plot(np.linspace(0,10,10), np.linspace(0,2,10))
    ax2.plot(np.linspace(0,10,10), np.linspace(0,2,10))
    ax3.plot(np.linspace(0,2,10), np.linspace(0,2,10))
    ax4.plot(np.linspace(0,2,10), np.linspace(0,2,10))

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    ax4.set_aspect('equal')

    if 1:
        fpl.adjust_spines(ax1, ['left'], yticks=[0,1,2])
        ax1.set_ylabel('y axis')
        fpl.adjust_spines(ax2, ['left', 'bottom'], xticks=[0,5,10], yticks=[0,1,2])
        ax2.set_ylabel('y axis')
        ax2.set_xlabel('x axis')
        fpl.adjust_spines(ax3, ['right'], yticks=[0,1,2])
        #ax3.set_ylabel('y axis')
        fpl.adjust_spines(ax4, ['right', 'bottom'], xticks=[0,1,2], yticks=[0,1,2])
        ax4.set_xlabel('x axis')
    if 0:
        fpl.adjust_spines(ax1, 'none')
        fpl.adjust_spines(ax2, 'none')
        fpl.adjust_spines(ax3, 'none')
        fpl.adjust_spines(ax4, 'none')

    if save:
        fig.savefig('figures/gridspec_example.pdf', format='pdf')
    
    
####################################################################################
def run_examples(save=True):
    text_wrapping_example(save)
    adjust_spines_example_with_custom_ticks(save)
    colorline_example(save)
    colorline_with_heading_example(save)
    colorline_with_heading_and_radius_example(save)
    histogram_example(save)
    histogram_stack_example(save)
    boxplot_example(save)
    boxplot_classic_example(save)
    histogram2d_example(save)
    colorbar_example(save)
    scatter_example(save)
    example_gridspec(save)

if __name__ == '__main__':
    run_examples()


