from matplotlib import rcParams

# NOTE: run this function prior to import matplotlib.pyplot!!!

def set_params(default='pdf', params={}):
    if default == 'pdf':
        default_params = {'backend': 'Agg',
                      'ps.usedistiller': 'xpdf',
                      'ps.fonttype' : 3,
                      'pdf.fonttype' : 3,
                      'font.family' : 'sans-serif',
                      'font.serif' : 'Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman',
                      'font.sans-serif' : 'Helvetica, Avant Garde, Computer Modern Sans serif',
                      'font.cursive' : 'Zapf Chancery',
                      'font.monospace' : 'Courier, Computer Modern Typewriter',
                      'font.size' : 8,
                      'text.fontsize': 8,
                      'axes.labelsize': 8,
                      'axes.linewidth': 1.0,
                      'xtick.major.linewidth': 1,
                      'xtick.minor.linewidth': 1,
                      #'xtick.major.size': 6,
                      #'xtick.minor.size' : 3,
                      'xtick.labelsize': 8,
                      #'ytick.major.size': 6,
                      #'ytick.minor.size' : 3,
                      'ytick.labelsize': 8,
                      'figure.figsize': (8,8),
                      'figure.dpi' : 72,
                      'figure.facecolor' : 'white',
                      'figure.edgecolor' : 'white',
                      'savefig.dpi' : 300,
                      'savefig.facecolor' : 'white',
                      'savefig.edgecolor' : 'white',
                      'figure.subplot.left': 0.2,
                      'figure.subplot.right': 0.8,
                      'figure.subplot.bottom': 0.25,
                      'figure.subplot.top': 0.9,
                      'figure.subplot.wspace': 0.0,
                      'figure.subplot.hspace': 0.0,
                      'lines.linewidth': 1.0,
                      'text.usetex': True, 
                      }
    for key, val in params.items():
        default_params[key] = val
    rcParams.update(default_params) 
