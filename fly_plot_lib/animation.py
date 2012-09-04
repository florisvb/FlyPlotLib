import fly_plot_lib.plot as fpl

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

def example_movie_3_axes():
    t = np.linspace(0, 2*np.pi, 200)
    x = 0.5*np.cos(t) + 0.5
    y = 0.1*np.sin(t)
    z = 0.1*np.cos(t)
    color = z
    play_movie_3_axes(x, y, z, color, orientation=None, save=False, save_movie_path='', nskip=4, artists=[[], [], []], colornorm=None, colormap='jet', ghost_tail=20)

def play_movie_3_axes(x, y, z, color, orientation=None, save=False, save_movie_path='', nskip=0, images=[None, None, None], images_extents=[None, None, None], artists=[[], [], []], lims=[None, None, None], colornorm=None, colormap='jet', ghost_tail=20):
    '''
    Show an animation of an x,y,z trajectory with color and a tail. The axes are referred to in the following order: (xy, xz, yz)
    
    x, y, z         -- lists or arrays of length N
    color           -- list or array of length N, or color name string for constant color
    orientation     -- list of length N, or None. If None, movie uses circles, else uses oriented wedges from fly_plot_lib.get_wedges...
    artists         -- list of length 3, corresponding to x/y/z axes, each component should be a list of matplotlib artists
    lims            -- list of limits for the three axes (xy, xz, yz), if None, automatically generate limits based on trajectory
    ghost_tail      -- number of frames of 'tail' to show
    
    TODO: test and animate images, add image function instead of requiring premade list
    '''
    
    # prep plot
    fig = plt.figure()
    ax_xy = fig.add_subplot(221)
    ax_xz = fig.add_subplot(223)
    ax_yz = fig.add_subplot(224)
    axes = [ax_xy, ax_xz, ax_yz]
    anim_params = {'frame': -1*(1+nskip)}

    if colornorm is None:
        colornorm = [np.min(color), np.max(color)]    
    norm = matplotlib.colors.Normalize(colornorm[0], colornorm[1])
    color_mappable = matplotlib.cm.ScalarMappable(norm, plt.get_cmap('jet'))
    
    if orientation is None:
        radius = 0.01
        alpha = 1
        radiusnorm = None
        maxradius = 1
        minradius = 0
        flies_xy = fpl.get_circles_for_scatter(x, y, color=color, colormap=colormap, radius=radius, colornorm=colornorm, alpha=alpha, radiusnorm=radiusnorm, maxradius=maxradius, minradius=minradius)
        flies_xz = fpl.get_circles_for_scatter(x, z, color=color, colormap=colormap, radius=radius, colornorm=colornorm, alpha=alpha, radiusnorm=radiusnorm, maxradius=maxradius, minradius=minradius)
        flies_yz = fpl.get_circles_for_scatter(y, z, color=color, colormap=colormap, radius=radius, colornorm=colornorm, alpha=alpha, radiusnorm=radiusnorm, maxradius=maxradius, minradius=minradius)
        
    def get_images(f):
        ims = [None for i in range(3)]
        origin = 'lower'
        zorder_image = 0
        alpha_image = 0.5
        for i, image in enumerate(images):
            if image is not None:
                if image_extents[i] is None:
                    print 'Must include extent for image'
                    raise(ValueError)
                    
                if type(image) is list:
                    im = image[f]
                else:
                    im = image
                ims[i] = im
        return ims
        
    # initialize images:
    ims = get_images(0)
    ax_ims = [None for i in range(3)]
    for i, ax in enumerate(axes):
        if ims[i] is not None:
            im = ax.imshow(ims[i], extent=image_extens[i], origin='lower', cmap=plt.get_cmap(colormap), norm=norm, alpha=0.5, zorder=0)
            ax_ims[i] = im
    
    # add artists
    for i, ax in enumerate(axes):
        for artist in artists[i]:
            ax.add_artist(artist)
    
    ax_xy.add_collection(flies_xy)
    ax_xz.add_collection(flies_xz)
    ax_yz.add_collection(flies_yz)
    
    def init_plot(): 
        flies_xy.set_color('none')
        flies_xz.set_color('none')
        flies_yz.set_color('none')
        flies_xy.set_edgecolors('none')
        flies_xz.set_edgecolors('none')
        flies_yz.set_edgecolors('none')
        return flies_xy, flies_xz, flies_yz
    
    def updatefig(*args):
        #x = anim_params['x']
        #y = anim_params['y']
        #z = anim_params['z']
        #color = anim_params['color']
        print anim_params['frame']
        
        anim_params['frame'] += 1 + nskip
        frame_end = anim_params['frame'] + ghost_tail
        if frame_end > len(x):
            frame_end = len(x)
        if anim_params['frame'] >= len(x)-1:
            anim_params['frame'] = 0
            frame_end =  anim_params['frame'] + ghost_tail
        frames = np.arange(anim_params['frame'], frame_end, 1).tolist()
                                
        colors = ['none' for i in range(len(x))]
        for f in frames:
            colors[f] = color_mappable.to_rgba(color[f])
        
        flies_xy.set_facecolors(colors)
        flies_xz.set_facecolors(colors)
        flies_yz.set_facecolors(colors)
        
        if save:
            print anim_params['frame']
            frame_prefix = '_tmp'
            frame_prefix = os.path.join(save_movie_path, frame_prefix)
            strnum = str(anim_params['frame'])
            num_frame_digits = np.ceil(len(x) / 10.) + 1
            while len(strnum) < num_frame_digits:
                strnum = '0' + strnum
            frame_name = frame_prefix + '_' + strnum + '_' + '.png'
            fig.savefig(frame_name, format='png')
        
        return flies_xy, flies_xz, flies_yz

    # generate and set limits:
    maxs = [np.max(x), np.max(y), np.max(z)] 
    mins = [np.min(x), np.min(y), np.min(z)] 
    for i, lim in enumerate(lims):
        if lim is None:
            lims[i] = [mins[i], maxs[i]]
            
    ax_xy.set_xlim(lims[0][0], lims[0][1])
    ax_xy.set_ylim(lims[1][0], lims[1][1])
    
    ax_xz.set_xlim(lims[0][0], lims[0][1])
    ax_xz.set_ylim(lims[2][0], lims[2][1])
    
    ax_yz.set_xlim(lims[1][0], lims[1][1])
    ax_yz.set_ylim(lims[2][0], lims[2][1])
    
    for ax in axes:
        ax.set_aspect('equal')
    #
    
    fpl.adjust_spines(ax_xy, ['left'], yticks=[-.15, 0, .15])
    fpl.adjust_spines(ax_xz, ['left', 'bottom'], xticks=[-.2, 0, 1], yticks=[-.15, 0, .15])
    fpl.adjust_spines(ax_yz, ['right', 'bottom'], xticks=[-.15, 0, .15], yticks=[-.15, 0, .15])
    
    # top left plot
    ax_xy.set_ylabel('y position, m')
    
    # bottom left plot
    ax_xz.set_xlabel('x position, m')
    ax_xz.set_ylabel('z position, m')
    
    # bottom right plot
    ax_yz.set_xlabel('y position, m')
    #ax_yz.set_ylabel('z position, m') # I can't figure out how to make this appear on the right side...
    
    ani = animation.FuncAnimation(fig, updatefig, init_func=init_plot, fargs=anim_params, interval=50, blit=True)
    
    plt.show()
