This is a library with various plotting functions. 

INSTALLING:

from inside the main FlyPlotLib directory run:

>> python setup.py install

Note: you may wish to do this inside a virtual environment

########################################################################################################

EXAMPLES: 

>> cd example_plots
>> python ./make_example_plots.py

#########################################################################################################

NOTE on SAVING to PDF:

Install the following: texlive, ghostscript, dvipng, texlive-latex-extra

Then, do the following BEFORE importing matplotlib.pyplot (and thus fly_plot_lib.plot)

>> import fly_plot_lib
>> fly_plot_lib.set_params.pdf()

You can specify some parameters you wish to use like so:

>> params = {'figure.figsize': (8,8)}
>> fly_plot_lib.set_params.pdf(params)







