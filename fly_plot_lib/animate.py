import os

import fly_plot_lib.plot as fpl
import fly_plot_lib.text as flytext

import types

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpyimgproc as nim

import numpy as np

import copy

def get_image_file_list(directory):
    cmd = 'ls ' + directory
    ls = os.popen(cmd).read()
    all_filelist = ls.split('\n')
    try:
        all_filelist.remove('')
    except:
        pass
    return all_filelist
    
def get_nth_image_from_directory(n, directory):
    all_filelist = get_image_file_list(directory)
    
    img_to_get = all_filelist[n]
    # add path
    imgfilename = os.path.join(directory, img_to_get)
    img = plt.imread(imgfilename)    
    return img    
    
def get_image_data(images, frame, mono=True, flipimgx=False, flipimgy=False):
    if type(images) is list:
        return images[frame]
    elif type(images) is str: # should be a directory
        img = get_nth_image_from_directory(frame, images)
        if mono:
            if flipimgx:
                if flipimgy:
                    img[::-1,::-1,0]
                else:
                    return img[::-1,:,0]
            elif flipimgy:
                return img[:,::-1,0]
            else:
                return img[:,:,0]
        else:
            return img
    else:
        return images
        

def play_movie(x, y, images=None, extent=None, aspect='equal', color='blue', edgecolor='none', orientation=None, save=False, save_movie_path='', nskip=0, artists=[], xlim=None, ylim=None, colornorm=None, colormap='jet', ghost_tail=20, ax=None, wedge_radius=0.01, circle_radius=0.005, deg=False, flip=False, imagecolormap='jet', mono=True, flipimgx=False, flipimgy=False, strobe=False, sync_frames=None, figsize=(5,3), dpi=72):
    '''
    Show an animation of N x,y trajectories with color, orientation, and a tail. And optionally background images.
    
    x               -- list or np.array for x position of trajectory, OR, if multiple trajectories, list of lists/np.arrays
    y               -- list or np.array for y position of trajectory, OR, if multiple trajectories, list of lists/np.arrays
    images          --      - list of images to set as the background for the animation
                            - or a single static image
                            - or path to a directory with a sequence of pyplot.imread readable images (jpeg, png, etc), numbered in order
                            - or a type with a __call__ attribute (eg. a function or class method). 
                              The function __call__ attribute should take a single input, the frame number, and return the image to be used.
                            
    extent, aspect  --     see matplotlib.pyplot.imshow for details - note "origin" from imshow does not work as expected, use flipimgx and flipimgy instead
    flipimgx        -- flip the image in along the "x" axis (eg. reverse image columns), default: False. 
    flipimgy        -- flip the image in along the "y" axis (eg. reverse image rows), default: False. 
    imagecolormap   -- colormap for images, eg. 'jet' or 'gray'
        mono        -- if the image is, or should be, mono (grayscale) set this to True (default: True). Required for colormaps to work properly
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
    x = copy.copy(x)
    y = copy.copy(y)
    orientation = copy.copy(orientation)
    
    if orientation is None:
        orientation = [None for i in range(len(x))]
        
    plt.close('all')
    
    # prep plot
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
    anim_params = {'frame': -1*(1+nskip)+1, 'movie_finished': False, 'strobe': strobe, 'strobeimg': None, 'frames': []}
    
    # fix format for single trajectories
    if type(x) is list:
        if type(x[0]) is list or type(x[0]) is np.ndarray:
            #print
            #print type(x[0])
            #print str(len(x)) + ' flies!'
            
            if sync_frames is not None:
                largest_sync_frame = np.max(sync_frames)
                first_frames = []
                for i, xi in enumerate(x):
                    padding = [np.nan]*(largest_sync_frame - sync_frames[i])
                    x[i] = np.hstack((padding, x[i]))
                    y[i] = np.hstack((padding, y[i])) 
                    color[i] = np.hstack((padding, color[i])) 
                    if orientation[0] is not None:
                        orientation[i] = np.hstack((padding, orientation[i])) 
                    first_frames.append(len(padding))
            else:
                first_frames = [0]*len(x)
                
            final_frames = []
            longest_trajectory = 0
            for i, xi in enumerate(x):
                if len(xi) > longest_trajectory:
                    longest_trajectory = len(xi)
            for i, xi in enumerate(x):
                final_frames.append(len(xi))
            for xi in x:
                #print len(xi) 
        else:# type(x[0]) is float or type(x[0]) is int or type(x[0]) is long or type(x[0]):
            #print
            #print 'Just one fly!'
            x = [x]
            y = [y]
            color = [color]
            edgecolor = [edgecolor]
            orientation = [orientation]
            first_frames = [0]
            final_frames = [len(x[0])]
    else:
        #print
        #print 'Not sure what format x is!'
    
    anim_params.setdefault('final_frames', final_frames)
    anim_params.setdefault('first_frames', first_frames)
        
    #print 'length color: ', len(color), 'n flies: ', len(x)
        
    if len(color) != len(x):
        only_color = color[0]
        color = [only_color for i in range(len(x))]
        
    if type(edgecolor) is not list:
        edgecolor = [edgecolor]    
    if len(edgecolor) != len(x):
        only_edgecolor = edgecolor[0]
        edgecolor = [only_edgecolor for i in range(len(x))]
        
    if type(color[0]) is not str:
        if colornorm is None:
            colornorm = [np.min(color), np.max(color)]    
        norm = matplotlib.colors.Normalize(colornorm[0], colornorm[1])
        color_mappable = matplotlib.cm.ScalarMappable(norm, plt.get_cmap(colormap))
    else:
        colornorm = None
        colormap= None
        norm = None
        color_mappable = None
        
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
    for artist in artists:
        ax.add_artist(artist)
        
        
    # add images
    if images is not None:
        frame = 0
        if hasattr(images, '__call__'):
            imgdata = images(frame)
            if len(imgdata.shape) > 2:
                imgdata = imgdata[:,:,0]
            if flipimgx:
                if flipimgy:
                    imgdata = imgdata[::-1,::-1]
                else:
                    imgdata = imgdata[::-1,:]
            elif flipimgy:
                imgdata = imgdata[:,::-1]
            else:
                imgdata = imgdata[:,:]
         
        else:   
            imgdata = get_image_data(images, frame, mono=mono, flipimgx=flipimgx, flipimgy=flipimgy)
            
        img = ax.imshow( imgdata, extent=extent, cmap=plt.get_cmap(imagecolormap), zorder=-10)
    
    for fly in flies:
        ax.add_collection(fly)
    
    def init_plot(): 
        for fly in flies:
            fly.set_color('none')
        if images is None:
            return flies
        else:
            animation_objects = [fly for fly in flies]
            animation_objects.insert(0, img)
            return animation_objects
        
    
    def updatefig(*args):
        anim_params['frame'] += 1 + nskip
        if anim_params['frame'] >= len(x[0])-1:
            anim_params['frame'] = 1
            anim_params['movie_finished'] = True
        anim_params['frames'].append(anim_params['frame'])
        #print anim_params['frame']
        for i, fly in enumerate(flies):
            frame_start = anim_params['frame'] - ghost_tail
            frame_end = anim_params['frame']
            if frame_start < anim_params['first_frames'][i]:
                frame_start = anim_params['first_frames'][i]
            if frame_end > anim_params['final_frames'][i]:
                frame_end = anim_params['final_frames'][i]
            if frame_end < frame_start+nskip:
                continue
            frames = np.arange(frame_start, frame_end, nskip+1).tolist()
            colors = ['none' for f in range(len(x[i]))]
            edgecolors = ['none' for f in range(len(x[i]))]
            for f in frames:
                try:
                    colors[f] = color_mappable.to_rgba(color[i][f])
                except:
                    colors[f] = color[i]
                edgecolors[f] = edgecolor[i]
            
            fly.set_edgecolors(edgecolors)
            fly.set_facecolors(colors)
            
        if images is not None:
            if hasattr(images, '__call__'):
                imgdata = images(frames[-1])
                if len(imgdata.shape) > 2:
                    imgdata = imgdata[:,:,0]
                if flipimgx:
                    if flipimgy:
                        imgdata = imgdata[::-1,::-1]
                    else:
                        imgdata = imgdata[::-1,:]
                elif flipimgy:
                    imgdata = imgdata[:,::-1]
                else:
                    imgdata = imgdata[:,:]
             
            else:   
                imgdata = get_image_data(images, frames[-1], mono=mono, flipimgx=flipimgx, flipimgy=flipimgy)
                
            if anim_params['strobe']:
                if anim_params['strobeimg'] is None:
                    anim_params['strobeimg'] = imgdata
                else:
                    anim_params['strobeimg'] = nim.darken([anim_params['strobeimg'], imgdata])
                img.set_array(anim_params['strobeimg'])
            else:
                img.set_array(imgdata)
                
            
        if save and not anim_params['movie_finished']:
            #print 'saving frame: ', str(anim_params['frame']), ' -- if the animation you see is strange, do not worry, look at the pngs'
            frame_prefix = '_tmp'
            frame_prefix = os.path.join(save_movie_path, frame_prefix)
            strnum = str(anim_params['frame'])
            num_frame_digits = np.ceil( np.log10(len(x[0]))) + 1
            while len(strnum) < num_frame_digits:
                strnum = '0' + strnum
            frame_name = frame_prefix + '_' + strnum + '_' + '.png'
            flytext.set_fontsize(fig, 8)
            fig.savefig(frame_name, format='png')
            
        if save and anim_params['movie_finished']:
            #print 
            #print 'Movie finished saving! Close the plot screen now.'
            #print 'PNGs are at: ', save_movie_path
            #print 'To turn the PNGs into a movie, you can run this command from inside the directory with the tmp files: '
            #print 'mencoder \'mf://*.png\' -mf type=png:fps=30 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o animation.avi'
        
        if images is None:
            return flies
        else:
            animation_objects = [fly for fly in flies]
            animation_objects.insert(0, img)
            return animation_objects

    # generate and set limits:
    if xlim is None:
        xlim = [np.min(x[0]), np.max(x[0])]
    if ylim is None:
        ylim = [np.min(y[0]), np.max(y[0])]
    
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    
    ax.set_aspect(aspect)
    #
    
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xlim, yticks=ylim)
    ax.set_xlabel('x position, m')
    ax.set_ylabel('y position, m')    
    
    ani = animation.FuncAnimation(fig, updatefig, init_func=init_plot, fargs=anim_params, interval=50, blit=True)
    
    plt.show()
