import numpy as np
import cv2
import cv2.cv as cv
import matplotlib
import matplotlib.pyplot as plt
import time

NAN = np.nan

def draw_cv_trajectory(img, x, y, color, thickness):
    if 0:
        for i in range(len(x)-3):
            try:
                cv2.line(img, (int(x[i]), int(y[i])), (int(x[i+1]), int(y[i+1])), color[i].tolist(), thickness)
            except:
                pass
                print 'could not draw trajectory line, length pts: ', len(x), 'i: ', i

    for i in range(len(x)):
        cv2.circle(img, (x[i],y[i]), 1, color=color[i].tolist(), thickness=-1)


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
    
    
    for frame in range(2,nframes):
        s = str(frame) + ' of ' + str(nframes)
        print s
        mat_xy[:,:,:] = 255
        mat_xz[:,:,:] = 255
        
        if artist_function_xy is not None:
            mat_xy = artist_function_xy(mat_xy)
        if artist_function_xz is not None:
            mat_xz = artist_function_xz(mat_xz)
            
        first_frame = np.max([0, frame-ghost_tail])
        last_frame = frame
        
        x = xsync[:, first_frame:last_frame]
        y = ysync[:, first_frame:last_frame]
        z = zsync[:, first_frame:last_frame]
        #alpha = np.arange(first_frame, last_frame).reshape(1,last_frame-first_frame).astype(np.float32)
        #alpha /= float(last_frame)
        #alpha *= 255
        #alpha = alpha.astype(np.uint8)
        #alpha = np.repeat(alpha, len(x), axis=0)
        x = np.reshape(x, x.shape[0]*x.shape[1])
        y = np.reshape(y, y.shape[0]*y.shape[1])
        z = np.reshape(z, z.shape[0]*z.shape[1])
        #alpha = np.reshape(alpha, alpha.shape[0]*alpha.shape[1])
        if colors is not None:
            c = colors_sync[:, first_frame:last_frame]
            c = np.reshape(c, c.shape[0]*c.shape[1])
            rgba = color_mappable.to_rgba(c,bytes=True)
            rgba[:,[0, 2]] = rgba[:,[2, 0]] # convert from RGB to BGR
            #rgba[:,3] = alpha
            #print rgba
                        
        if len(x) > 1:
            if colors is None:
                indicesx, indicesy = get_indices(np.array(x), np.array(y), xmesh, ymesh, radius)
                indicesx, indicesz = get_indices(np.array(x), np.array(z), xmesh, zmesh, radius)
            else:
                indicesx, indicesy = get_indices(np.array(x), np.array(y), xmesh, ymesh, radius)
                indicesx, indicesz = get_indices(np.array(x), np.array(z), xmesh, zmesh, radius)
            
            # draw the ghost tails
            draw_cv_trajectory(mat_xy, indicesx, indicesy, rgba, 1)
            draw_cv_trajectory(mat_xz, indicesx, indicesz, rgba, 1)
            
            # draw the points as circles
            if 1:
                x = xsync[:, last_frame]
                y = ysync[:, last_frame]
                z = zsync[:, last_frame]
                c = colors_sync[:, last_frame]
                rgba = color_mappable.to_rgba(c,bytes=True) 
                rgba[:,[0, 2]] = rgba[:,[2, 0]] # convert from RGB to BGR
                
                indicesx, indicesy = get_indices(np.array(x), np.array(y), xmesh, ymesh, radius)
                indicesx, indicesz = get_indices(np.array(x), np.array(z), xmesh, zmesh, radius)
                
                for i in range(len(x)):
                    try:
                        cv2.circle(mat_xy, (indicesx[i],indicesy[i]), 5, color=rgba[i].tolist(), thickness=-1)
                        cv2.circle(mat_xz, (indicesx[i],indicesz[i]), 5, color=rgba[i].tolist(), thickness=-1)
                    except:
                        pass
                        
        mat = stack_mats(mat_xy, mat_xz)
        
        matflipped = np.array(np.flipud(mat))
        writer.write(matflipped)
        
        del(x)
        del(y)
        del(z)
        
        
    writer.release()
