import os
import numpy as np
import matplotlib.pyplot as plt
import fly_plot_lib.animate as flyanim
import pickle

'''
Note: to save movies, set these flags:

save = True
save_movie_path = 'path to where you want tmp files saved'

Then the animation will write png files of each frame. You can turn that into a movie by running something like this, from inside the directory with the pngs:
mencoder 'mf://*.png' -mf type=png:fps=30 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o animation.avi


'''


def example_movie(save_movie=False):

    # note: this example is done in pixel units.
    image_directory = 'images'
    
    # unpack data from the pickled file - this will vary depending on your data, of course
    datafile = open('data.pickle', 'r')
    data = pickle.load(datafile)
    timestamps = []
    x = []
    y = []
    orientation = []
    for frame in data.keys():
        framedata = data[frame]
        if framedata is not None:
            timestamps.append(framedata['timestamp'])
            x.append(framedata['position'][0])
            y.append(framedata['position'][1])
            orientation.append(framedata['orientation'])
        else:
            timestamps.append(0)
            x.append(0)
            y.append(0)
            orientation.append(0)
    
    # check to make sure we have same number of images as data points
    images = flyanim.get_image_file_list(image_directory)
    print 'n images: ', len(images)
    print 'n data pts: ', len(x)
            
    # optional parameters
    color = 'none'
    edgecolor = 'red'
    ghost_tail = 20
    nskip = 0
    wedge_radius = 25
    imagecolormap = 'gray'
    
    # get x/y limits (from image)
    img = flyanim.get_nth_image_from_directory(0, image_directory)
    xlim = [0, img.shape[0]]
    ylim = [0, img.shape[1]]
    
    if save_movie is False:
        save = False
        save_movie_path = ''
    else:
        save = True
        save_movie_path = 'tmp/'
        
        if not os.path.isdir(save_movie_path):
            os.mkdir(save_movie_path)
            
    # useful parameters for aligning image and data:
    # extent, origin, flipimgx
            
    # play the movie!
    flyanim.play_movie(x, y, color=color, images=image_directory, orientation=orientation, save=save, save_movie_path=save_movie_path, nskip=nskip, ghost_tail=ghost_tail, wedge_radius=wedge_radius, xlim=xlim, ylim=ylim, imagecolormap=imagecolormap, edgecolor=edgecolor)
    
    
if __name__ == '__main__':
    example_movie()
