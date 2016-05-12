import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import time

NAN = np.nan

def get_indices(x, y, xmesh, ymesh, radius=1, colors=None):
    # pull out non NAN numbers only
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    ix = [np.argmin( np.abs( xmesh-xval ) ) for xval in x]
    iy = [np.argmin( np.abs( ymesh-yval ) ) for yval in y]
    
    '''
    ix_enlarged = []
    iy_enlarged = []
    if colors is not None:
        colors_enlarged = []
    
    for n, i in enumerate(ix):
        min_i = np.max([0, i-radius])
        max_i = np.min([len(xmesh), i+radius])
        a = np.arange(min_i, max_i)
        ix_enlarged.extend(a)
        if colors is not None:
            colors_enlarged.extend([colors[n]]*len(a))
        
    for i in iy:
        min_i = np.max([0, i-radius])
        max_i = np.min([len(ymesh), i+radius])
        a = np.arange(min_i, max_i)
        iy_enlarged.extend(a)
        
    #if len(ix) == 1:
    #    return ix[0], iy[0]
    #else:
    
    if colors is None:
        return ix_enlarged, iy_enlarged
    else:
        return ix_enlarged, iy_enlarged, colors_enlarged
    '''
    
    return ix, iy
        
def synchronize_frames(x, y, sync_frames, padval=NAN, colors=None, n_frames_before_sync_to_show='all'):
    xsync = []
    ysync = []
    if colors is not None:
        colors_sync = []
    largest_sync_frame = np.max(sync_frames)
    for i, xi in enumerate(x):
        padding = [padval]*(largest_sync_frame - sync_frames[i])
        xsync.append( np.hstack((padding, x[i])) )
        ysync.append( np.hstack((padding, y[i])) )
        if colors is not None:
            colors_sync.append( np.hstack((padding, colors[i])) )
        
    # pad back
    lengths = [len(x) for x in xsync]
    length_of_longest_sequence = np.max(lengths)
    for i, xi in enumerate(xsync):
        padding = [padval]*(length_of_longest_sequence - len(xi))
        xsync[i] = np.hstack((xsync[i], padding))
        ysync[i] = np.hstack((ysync[i], padding))
        if colors is not None:
            colors_sync[i] = np.hstack((colors_sync[i], padding))
            
    if n_frames_before_sync_to_show != 'all':
        first_frame = largest_sync_frame - n_frames_before_sync_to_show
        for i, xi in enumerate(xsync):
            xsync[i] = xsync[i][first_frame:]
            ysync[i] = ysync[i][first_frame:]
            if colors is not None:
                colors_sync[i] = colors_sync[i][first_frame:]
    
    if colors is None:
        return xsync, ysync
    else:
        return xsync, ysync, colors_sync
    
def animate_matrix(x, y, colors=None, xlim=[0,1], ylim=[0,1], resolution=0.005, filename='', sync_frames=[], framerate=100, ghost_tail=20, radius=2, static_indices=[], static_color=[0,0,255], colormap='hot', colornorm=[0,1], n_frames_before_sync_to_show='all'):
    xmesh = np.arange(xlim[0], xlim[1], resolution)
    ymesh = np.arange(ylim[0], ylim[1], resolution)
    mat = np.ones([len(ymesh), len(xmesh), 3], dtype=np.uint8)
    mat *= 255

    kernel = np.ones((5,5),np.uint8)
    norm = matplotlib.colors.Normalize(colornorm[0], colornorm[1])
    color_mappable = matplotlib.cm.ScalarMappable(norm, plt.get_cmap(colormap))
    
    print 'synchronizing trajectories'
    if colors is None:
        xsync, ysync = synchronize_frames(x, y, sync_frames, n_frames_before_sync_to_show=n_frames_before_sync_to_show)
        xsync = np.array(xsync)
        ysync = np.array(ysync)
    else:
        xsync, ysync, colors_sync = synchronize_frames(x, y, sync_frames, colors=colors, n_frames_before_sync_to_show=n_frames_before_sync_to_show)
        xsync = np.array(xsync)
        ysync = np.array(ysync)
        colors_sync = np.array(colors_sync)
            
    
    #this works:
    #writer = cv2.VideoWriter(filename,cv.CV_FOURCC('P','I','M','1'),sampleRate,(panelsFrames.shape[1],panelsFrames.shape[0]),True) # works for Linux
    # but this works better:
    print 'initializing writer'
    writer = cv2.VideoWriter(filename,cv.CV_FOURCC('m','p','4','v'),framerate,(mat.shape[1], mat.shape[0]),True) # works on Linux and Windows
    
    print filename
    
    nframes = len(xsync[0])
    for frame in range(nframes):
        s = str(frame) + ' of ' + str(nframes)
        print s
        mat[:,:,:] = 255
        if len(static_indices) > 0:
            c = [static_color for i in range(len(static_indices[0]))]
            mat[static_indices[1], static_indices[0], :] = np.array(c)
        
        
        first_frame = np.max([0, frame-ghost_tail])
        last_frame = frame
        
        if 1:
            
            x = xsync[:, first_frame:last_frame]
            y = ysync[:, first_frame:last_frame]
            x = np.reshape(x, x.shape[0]*x.shape[1])
            y = np.reshape(y, y.shape[0]*y.shape[1])
            if colors is not None:
                c = colors_sync[:, first_frame:last_frame]
                c = np.reshape(c, c.shape[0]*c.shape[1])
                rgba = color_mappable.to_rgba(c,bytes=True) 
                
        if colors is None:
            indicesx, indicesy = get_indices(np.array(x), np.array(y), xmesh, ymesh, radius)
        else:
            indicesx, indicesy = get_indices(np.array(x), np.array(y), xmesh, ymesh, radius)
            


        if len(indicesx) < 1:
            continue        
        if colors is None:
            mat[indicesy, indicesx, :] = 0
        else:
            rgba = rgba[np.isfinite(x)][:,[2,1,0]]
            mat[indicesy, indicesx, :] = rgba
        
        mat = cv2.erode(mat, kernel, radius)
        
        
        # using uint8 for the values in the frame seems to work best. Also, I think rgb should be ordered bgr....
        matflipped = np.array(np.flipud(mat))
        writer.write(matflipped)
        
        del(x)
        del(y)
        
        
        
    writer.release()
    
def animate_matrix_2views(x, y, z, 
                            colors=None, 
                            xlim=[0,1], 
                            ylim=[0,1], 
                            zlim=[0,1], 
                            resolution=0.005, 
                            filename='', 
                            sync_frames=[], 
                            framerate=100, 
                            ghost_tail=20, 
                            radius=2, 
                            artist_function_xy=None, 
                            artist_function_xz=None, 
                            colormap='hot', 
                            colornorm=[0,1], 
                            n_frames_before_sync_to_show='all'):
                            
    def stack_mats(mat_xy, mat_xz):
        # add border to mats
        mat_xy[:,0,:] = 0
        mat_xy[:,-1,:] = 0
        mat_xy[0,:,:] = 0
        mat_xy[-1,:,:] = 0
        
        mat_xz[:,0,:] = 0
        mat_xz[:,-1,:] = 0
        mat_xz[0,:,:] = 0
        mat_xz[-1,:,:] = 0
        
        mat = np.vstack((mat_xy, mat_xz))
        return mat
                            
    xmesh = np.arange(xlim[0], xlim[1], resolution)
    ymesh = np.arange(ylim[0], ylim[1], resolution)
    zmesh = np.arange(zlim[0], zlim[1], resolution)
    
    mat_xy = np.ones([len(ymesh), len(xmesh), 3], dtype=np.uint8)
    mat_xy *= 255
    
    mat_xz = np.ones([len(zmesh), len(xmesh), 3], dtype=np.uint8)
    mat_xz *= 255

    kernel = np.ones((5,5),np.uint8)
    norm = matplotlib.colors.Normalize(colornorm[0], colornorm[1])
    color_mappable = matplotlib.cm.ScalarMappable(norm, plt.get_cmap(colormap))
    
    print 'synchronizing trajectories'
    if colors is None:
        xsync, ysync = synchronize_frames(x, y, sync_frames, n_frames_before_sync_to_show=n_frames_before_sync_to_show)
        xsync, zsync = synchronize_frames(x, z, sync_frames, n_frames_before_sync_to_show=n_frames_before_sync_to_show)
        xsync = np.array(xsync)
        ysync = np.array(ysync)
        zsync = np.array(zsync)
    else:
        xsync, ysync, colors_sync = synchronize_frames(x, y, sync_frames, colors=colors, n_frames_before_sync_to_show=n_frames_before_sync_to_show)
        xsync, zsync, colors_sync = synchronize_frames(x, z, sync_frames, colors=colors, n_frames_before_sync_to_show=n_frames_before_sync_to_show)
        xsync = np.array(xsync)
        ysync = np.array(ysync)
        zsync = np.array(zsync)
        colors_sync = np.array(colors_sync)
            
    
    #this works:
    #writer = cv2.VideoWriter(filename,cv.CV_FOURCC('P','I','M','1'),sampleRate,(panelsFrames.shape[1],panelsFrames.shape[0]),True) # works for Linux
    # but this works better:
    print 'initializing writer'
    mat = stack_mats(mat_xy, mat_xz)
    writer = cv2.VideoWriter(filename,cv.CV_FOURCC('m','p','4','v'),framerate,(mat.shape[1], mat.shape[0]),True) # works on Linux and Windows
    
    print filename
    
    nframes = len(xsync[0])
    for frame in range(nframes):
        s = str(frame) + ' of ' + str(nframes)
        print s
        mat_xy[:,:,:] = 255
        mat_xz[:,:,:] = 255
        
        first_frame = np.max([0, frame-ghost_tail])
        last_frame = frame
        
        if 1:
            
            x = xsync[:, first_frame:last_frame]
            y = ysync[:, first_frame:last_frame]
            z = zsync[:, first_frame:last_frame]
            x = np.reshape(x, x.shape[0]*x.shape[1])
            y = np.reshape(y, y.shape[0]*y.shape[1])
            z = np.reshape(z, z.shape[0]*z.shape[1])
            if colors is not None:
                c = colors_sync[:, first_frame:last_frame]
                c = np.reshape(c, c.shape[0]*c.shape[1])
                rgba = color_mappable.to_rgba(c,bytes=True) 
                
        if colors is None:
            indicesx, indicesy = get_indices(np.array(x), np.array(y), xmesh, ymesh, radius)
            indicesx, indicesz = get_indices(np.array(x), np.array(z), xmesh, zmesh, radius)
        else:
            indicesx, indicesy = get_indices(np.array(x), np.array(y), xmesh, ymesh, radius)
            indicesx, indicesz = get_indices(np.array(x), np.array(z), xmesh, zmesh, radius)


        if len(indicesx) < 1:
            continue        
        if colors is None:
            mat_xy[indicesy, indicesx, :] = 0
            mat_xz[indicesz, indicesx, :] = 0
        else:
            rgba = rgba[np.isfinite(x)][:,[2,1,0]]
            mat_xy[indicesy, indicesx, :] = rgba
            mat_xz[indicesz, indicesx, :] = rgba
        
        mat_xy = cv2.erode(mat_xy, kernel, radius)
        mat_xz = cv2.erode(mat_xz, kernel, radius)
        
        if artist_function_xy is not None:
            mat_xy = artist_function_xy(mat_xy)
        if artist_function_xz is not None:
            mat_xz = artist_function_xz(mat_xz)
        
        mat = stack_mats(mat_xy, mat_xz)
        
        matflipped = np.array(np.flipud(mat))
        writer.write(matflipped)
        
        del(x)
        del(y)
        del(z)
        
        
    writer.release()
