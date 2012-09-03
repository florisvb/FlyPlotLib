import os

import fly_plot_lib.plot as fpl

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

def play_movie(x, y, color='blue', edgecolor='none', orientation=None, save=False, save_movie_path='', nskip=0, artists=[], xlim=None, ylim=None, colornorm=None, colormap='jet', ghost_tail=20, ax=None, wedge_radius=0.01, circle_radius=0.005, deg=False, flip=True):
    '''
    Show an animation of N x,y trajectories with color, orientation, and a tail.
    
    x               -- list or np.array for x position of trajectory, OR, if multiple trajectories, list of lists/np.arrays
    y               -- list or np.array for y position of trajectory, OR, if multiple trajectories, list of lists/np.arrays
    color           -- list, OR list of lists/np.arrays, OR string
        colornorm   -- [min, max] to use for normalizing color
                       If None, uses first trajectory to set colornorm
    orientation     -- list, OR list of lists/np.arrays, OR None. 
                       If None, movie uses circles, else uses oriented wedges from fly_plot_lib.get_wedges...
    artists         -- optional list of matplotlib artists
    xlim            -- xlim for plot, if None, automatically generate limits based on trajectory
    ylim            -- ylim for plot, if None, automatically generate limits based on trajectory
    ghost_tail      -- number of frames of 'tail' to show
    ax              -- matplotlib axis, optional
    wedge_radius    -- size_radius used in fpl.get_wedges
    circle_radius   -- radius used in fpl.get_circles
                       currently changing size is NOT supported for animations. 
    deg             -- (bool) is orientation given in degrees? True means yes. Passed to fpl.get_wedges...
    flip            -- (bool) flip orientation? True means yes. Passed to fpl.get_wedges...
    
    '''
    
    # prep plot
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    anim_params = {'frame': -1*(1+nskip), 'movie_finished': False}
    
    # fix format for single trajectories
    if type(x) is list:
        if type(x) is float or type(x) is int or type(x) is long:
            x = [x]
            y = [y]
            color = [color]
            edgecolor = [edgecolor]
            orientation = [orientation]
                
    if len(color) != len(x):
        only_color = color[0]
        color = [only_color for i in range(len(x))]
        
    if type(edgecolor) is not list:
        edgecolor = [edgecolor]    
    if len(edgecolor) != len(x):
        only_edgecolor = edgecolor[0]
        edgecolor = [only_edgecolor for i in range(len(x))]
        
    if colornorm is None:
        colornorm = [np.min(color), np.max(color)]    
    norm = matplotlib.colors.Normalize(colornorm[0], colornorm[1])
    color_mappable = matplotlib.cm.ScalarMappable(norm, plt.get_cmap('jet'))

    flies = []
    for i, xpos in enumerate(x):
        if orientation[i] is None: # use circles from scatter
            radius = circle_radius
            alpha = 1
            radiusnorm = None
            maxradius = 1
            minradius = 0
            fly = fpl.get_circles_for_scatter(x[i], y[i], color=color[i], colormap=colormap, radius=radius, colornorm=colornorm, alpha=alpha, radiusnorm=radiusnorm, maxradius=maxradius, minradius=minradius, edgecolor=edgecolor)
        elif orientation[i] is not None: # use wedges from get_wedges
            fly = fpl.get_wedges_for_heading_plot(x[i], y[i], color[i], orientation[i], size_radius=wedge_radius, size_angle=20, colormap=colormap, colornorm=colornorm, alpha=1, flip=flip, deg=deg, nskip=0, center_offset_fraction=0.75, edgecolor=edgecolor)
        
        
        flies.append(fly)
    
    # add artists
    for artist in artists[i]:
        ax.add_artist(artist)
    
    for fly in flies:
        ax.add_collection(fly)
    
    def init_plot(): 
        for fly in flies:
            fly.set_color('none')
        return flies
    
    def updatefig(*args):
        #print anim_params['frame']
        
        anim_params['frame'] += 1 + nskip
        frame_end = anim_params['frame'] + ghost_tail
        if frame_end > len(x[0]):
            anim_params['movie_finished'] = True
            frame_end = len(x[0])
        if anim_params['frame'] >= len(x[0])-1:
            anim_params['frame'] = 0
            frame_end =  anim_params['frame'] + ghost_tail
        frames = np.arange(anim_params['frame'], frame_end, 1).tolist()
        
        for i, fly in enumerate(flies):
            if frame_end > len(x[i]):
                colors = ['none' for f in range(len(x[i]))]
            else:
                colors = ['none' for f in range(len(x[i]))]
                for f in frames:
                    try:
                        colors[f] = color_mappable.to_rgba(color[i][f])
                    except:
                        colors[f] = color[i]
                    
            if frame_end > len(x[i]):
                edgecolors = ['none' for f in range(len(x[i]))]
            else:
                edgecolors = ['none' for f in range(len(x[i]))]
                for f in frames:
                    edgecolors[f] = edgecolor[i]
            
            fly.set_edgecolors(edgecolors)
            fly.set_facecolors(colors)
        
        if save and not anim_params['movie_finished']:
            print 'saving frame: ', str(anim_params['frame']), ' -- if the animation you see is strange, do not worry, look at the pngs'
            frame_prefix = '_tmp'
            frame_prefix = os.path.join(save_movie_path, frame_prefix)
            strnum = str(anim_params['frame'])
            num_frame_digits = np.ceil( np.log10(len(x[0]))) + 1
            while len(strnum) < num_frame_digits:
                strnum = '0' + strnum
            frame_name = frame_prefix + '_' + strnum + '_' + '.png'
            fig.savefig(frame_name, format='png')
            
        if save and anim_params['movie_finished']:
            print 
            print 'Movie finished saving! Close the plot screen now.'
            print 'PNGs are at: ', save_movie_path
            print 'To turn the PNGs into a movie, you can run this command from inside the directory with the tmp files: '
            print 'mencoder \'mf://*.png\' -mf type=png:fps=30 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o animation.avi'
        
        return flies

    # generate and set limits:
    if xlim is None:
        xlim = [np.min(x[0]), np.max(x[0])]
    if ylim is None:
        ylim = [np.min(y[0]), np.max(y[0])]
    
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    
    ax.set_aspect('equal')
    #
    
    fpl.adjust_spines(ax, ['left', 'bottom'])
    ax.set_xlabel('x position, m')
    ax.set_ylabel('y position, m')    
    
    ani = animation.FuncAnimation(fig, updatefig, init_func=init_plot, fargs=anim_params, interval=50, blit=True)
    
    plt.show()
