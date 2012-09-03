import numpy as np

import fly_plot_lib.animate1d as flyanim

def example_movie():
    t = np.linspace(0, 2*np.pi, 200)
    
    # fly 1
    x1 = 0.5*np.cos(t) + 0.5
    y1 = 0.1*np.sin(t)
    orientation1 = t
    color1 = t
    
    # fly 2
    x2 = 0.3*np.sin(t) + 0.5
    y2 = 0.3*np.cos(t)
    orientation2 = None
    color2 = "black"
    
    # flies
    x = [x1, x2]
    y = [y1, y2]
    orientation = [orientation1, orientation2]
    color = [color1, color2]
    
    # optional parameters
    colormap = 'jet'
    colornorm = [0, 2*np.pi]
    ghost_tail = 20
    nskip = 1
    xlim = [0, 1]
    ylim = [-.5, .5]
    wedge_radius = 0.05
    circle_radius = 0.01
    edgecolor = 'black'
    
    save = False
    save_movie_path = ''
    
    print len(color[1])
    
    flyanim.play_movie(x, y, color, orientation=orientation, save=save, save_movie_path=save_movie_path, nskip=nskip, artists=[[], [], []], colornorm=colornorm, colormap=colormap, ghost_tail=ghost_tail, xlim=xlim, ylim=ylim, wedge_radius=wedge_radius, circle_radius=circle_radius, edgecolor=edgecolor)
    
    
if __name__ == '__main__':
    example_movie()
