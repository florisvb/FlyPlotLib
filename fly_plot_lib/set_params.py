from matplotlib import rcParams

# NOTE: run this function prior to import matplotlib.pyplot!!!

def pdf(params={}, presentation='powerpoint'):

    if presentation == 'powerpoint':
        fontsize = 14
        figsize = (10,7.5)
        subplot_left = 0.15
        subplot_right = 0.85
        subplot_top = 0.8
        subplot_bottom = 0.15
        
    if presentation == 'paper':
        fontsize = 8
        figsize = (8,8)
        subplot_left = 0.2
        subplot_right = 0.8
        subplot_top = 0.8
        subplot_bottom = 0.2

    print('Loading rcparams for saving to PDF')
    print('NOTE: ipython plotting may not work as expected with these parameters loaded!')
    default_params = {'backend': 'Agg',
                      'ps.usedistiller': 'xpdf',
                      'ps.fonttype' : 3,
                      'pdf.fonttype' : 3,
                      'font.family' : 'sans-serif',
                      'font.serif' : 'Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman',
                      'font.sans-serif' : 'Helvetica, Avant Garde, Computer Modern Sans serif',
                      'font.cursive' : 'Zapf Chancery',
                      'font.monospace' : 'Courier, Computer Modern Typewriter',
                      'font.size' : fontsize,
                      'text.fontsize': fontsize,
                      'axes.labelsize': fontsize,
                      'axes.linewidth': 1.0,
                      'xtick.major.linewidth': 1,
                      'xtick.minor.linewidth': 1,
                      #'xtick.major.size': 6,
                      #'xtick.minor.size' : 3,
                      'xtick.labelsize': fontsize,
                      #'ytick.major.size': 6,
                      #'ytick.minor.size' : 3,
                      'ytick.labelsize': fontsize,
                      'figure.figsize': figsize,
                      'figure.dpi' : 72,
                      'figure.facecolor' : 'white',
                      'figure.edgecolor' : 'white',
                      'savefig.dpi' : 300,
                      'savefig.facecolor' : 'white',
                      'savefig.edgecolor' : 'white',
                      'figure.subplot.left': subplot_left,
                      'figure.subplot.right': subplot_right,
                      'figure.subplot.bottom': subplot_bottom,
                      'figure.subplot.top': subplot_top,
                      'figure.subplot.wspace': 0.2,
                      'figure.subplot.hspace': 0.2,
                      'lines.linewidth': 1.0,
                      'text.usetex': True, 
                      }
    for key, val in params.items():
        default_params[key] = val
    rcParams.update(default_params) 
