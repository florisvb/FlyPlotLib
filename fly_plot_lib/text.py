# This file includes functions for fancy text handling, such as automatic text wrapping.
# Original text-wrapping code written by Joe Kington, via StackOverflow

import matplotlib.pyplot as plt
    
    
def text_box_example():
    text = 'hi am a super long string... alskdjaklsjdhflkajhdsflkjas aksjdfasdl alskdfj asldkfasd lasldkfjasd oaeasdf adglkajdfga adlfj asdf asldf ajsdflksdj asdf asldkja ldsf a asdkfj alskdjf a a alskdfj alkdj awldsf asldf laej fldsf alskdfj alksdf al as as lkdj ladsaslaskdjfalskdjf  alsdk fjlaskdjf laskdj'
    
    fig = plt.figure()
    text_box(fig, 0.1, 0.9, 0.8, 0.8, text, family='serif', fontsize=12, horizontalalignment='left')
    plt.show()
    
def text_box(fig, left, top, width, height, text, family=None, fontsize=None, horizontalalignment='left', verticalalignment='top', rotation=0, style=None, **kwargs):
    """
    Function for creating a text box with automatic text wrapping. Thanks to Joe Kington via StackOverflow.
    The function creates a new set of axes without frames or ticks and wraps the text to fit inside the box.
    Note: if the text does not fit in the text box, the auto-wrapping will not happen.
    
    fig                         -- the figure instance on which you would like to make a text box
    left, top, width, height    -- for the boundaries of the text box, in fractions of the fig (from 0 to 1). Can be overlaid on 
                                   top of existing axes
    text                        -- the text
    
    Optional arguments - see fig.text or ax.text:
    family, fontsize, horizontalalignment, verticalalignment, rotation, style
    
    Note: style does not appear to work robustly
    
    
    """

    text_box = fig.add_axes([left, top-height, width, height])
    text_box.set_xlim(0,1)
    text_box.set_ylim(0,1)
    text_box.text(  0,1, text,
                    verticalalignment=verticalalignment, 
                    horizontalalignment=horizontalalignment,
                    family=family,
                    fontsize=fontsize,
                    style=style,
                    **kwargs)
    
    text_box.set_frame_on(False)
    text_box.set_xticks([])
    text_box.set_yticks([])
    
    fig.canvas.mpl_connect('draw_event', __on_draw__)
    
def __on_draw__(event):
    """Auto-wraps all text objects in a figure at draw-time"""
    import matplotlib as mpl
    fig = event.canvas.figure

    # Cycle through all artists in all the axes in the figure
    for ax in fig.axes:
        for artist in ax.get_children():
            # If it's a text artist, wrap it...
            if isinstance(artist, mpl.text.Text):
                __autowrap_text__(artist, event.renderer)
                
    # Temporarily disconnect any callbacks to the draw event...
    # (To avoid recursion)
    func_handles = fig.canvas.callbacks.callbacks[event.name]
    fig.canvas.callbacks.callbacks[event.name] = {}
    # Re-draw the figure..
    fig.canvas.draw()
    # Reset the draw event callbacks
    fig.canvas.callbacks.callbacks[event.name] = func_handles

def __autowrap_text__(textobj, renderer):
    """Wraps the given matplotlib text object so that it exceed the boundaries
    of the axis it is plotted in."""
    import textwrap
    # Get the starting position of the text in pixels...
    x0, y0 = textobj.get_transform().transform(textobj.get_position())
    # Get the extents of the current axis in pixels...
    clip = textobj.get_axes().get_window_extent()
    # Set the text to rotate about the left edge (doesn't make sense otherwise)
    textobj.set_rotation_mode('anchor')

    # Get the amount of space in the direction of rotation to the left and 
    # right of x0, y0 (left and right are relative to the rotation, as well)
    rotation = textobj.get_rotation()
    right_space = __min_dist_inside__((x0, y0), rotation, clip)
    left_space = __min_dist_inside__((x0, y0), rotation - 180, clip)

    # Use either the left or right distance depending on the horiz alignment.
    alignment = textobj.get_horizontalalignment()
    if alignment is 'left':
        new_width = right_space 
    elif alignment is 'right':
        new_width = left_space
    else:
        new_width = 2 * min(left_space, right_space)

    # Estimate the width of the new size in characters...
    aspect_ratio = 0.5 # This varies with the font!! 
    fontsize = textobj.get_size()
    pixels_per_char = aspect_ratio * renderer.points_to_pixels(fontsize)

    # If wrap_width is < 1, just make it 1 character
    wrap_width = max(1, new_width // pixels_per_char)
    try:
        wrapped_text = textwrap.fill(textobj.get_text(), wrap_width)
    except TypeError:
        # This appears to be a single word
        wrapped_text = textobj.get_text()
    textobj.set_text(wrapped_text)

def __min_dist_inside__(point, rotation, box):
    """Gets the space in a given direction from "point" to the boundaries of
    "box" (where box is an object with x0, y0, x1, & y1 attributes, point is a
    tuple of x,y, and rotation is the angle in degrees)"""
    from math import sin, cos, radians
    x0, y0 = point
    rotation = radians(rotation)
    distances = []
    threshold = 0.0001 
    if cos(rotation) > threshold: 
        # Intersects the right axis
        distances.append((box.x1 - x0) / cos(rotation))
    if cos(rotation) < -threshold: 
        # Intersects the left axis
        distances.append((box.x0 - x0) / cos(rotation))
    if sin(rotation) > threshold: 
        # Intersects the top axis
        distances.append((box.y1 - y0) / sin(rotation))
    if sin(rotation) < -threshold: 
        # Intersects the bottom axis
        distances.append((box.y0 - y0) / sin(rotation))
    return min(distances)


