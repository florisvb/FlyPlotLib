import fly_plot_lib
fly_plot_lib.set_params.pdf()
fpl = fly_plot_lib.plot

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm as gaussian_distribution

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
    orientation = np.arcsin(y)*180./np.pi
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # standard colorline
    fpl.colorline_with_heading(ax,t,y,z,orientation)
    
    # set the axis to appropriate limits
    fpl.adjust_spines(ax, ['left', 'bottom'])
    ax.set_xlim(0,2)
    ax.set_ylim(0,1.5)
    
    if save:
        fig.savefig('figures/colorline_with_orientation_example.pdf', format='pdf')

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
    fpl.scatter(ax, x, y, color=x*10) # with color scale
    fpl.scatter(ax, x+1, y+1, color='black') # set fixed color
    fpl.scatter(ax, x+1, y, color='blue', radius=0.05, alpha=0.2) # set some parameters for all circles 
    fpl.scatter(ax, x, y+1, color='green', radius=x, alpha=0.6, radiusnorm=(0.2, 0.8), minradius=0.01, maxradius=0.05) # let radius vary with some array 
    
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.set_aspect('equal')
    fpl.adjust_spines(ax, ['left', 'bottom'])
    
    if save:
        fig.savefig('figures/scatter_example.pdf', format='pdf')
    
    
####################################################################################
def run_examples(save=True):

    adjust_spines_example_with_custom_ticks(save)
    colorline_example(save)
    colorline_with_heading_example(save)
    histogram_example(save)
    boxplot_example(save)
    boxplot_classic_example(save)
    histogram2d_example(save)
    colorbar_example(save)
    scatter_example(save)

if __name__ == '__main__':
    run_examples()

