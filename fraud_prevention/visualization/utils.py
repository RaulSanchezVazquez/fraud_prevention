#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def add_bar_values(ax, fontsize=10, kind=None, rotation=None, color='black'):
    """Add numeric values on top of the bars.

    Parameters
    -----------
    ax: matplotlib.axes
        Plot containing the plot-bars.
    fontsize: int
        Font size.
    kind: str
        Either 'bar' or 'barh'
    rotation: float
        Rotation angle of the inclination of the text on to of bars in degrees.
    color: str
        Color of the font.
    """

    if kind is None:
        kind = 'bar'

    if (rotation is None) and (kind == 'bar'):
        rotation = 45
    elif (rotation is None) and (kind == 'barh'):
        rotation = 0

    for bar in ax.patches:
        flag_prevent_plot = False
        height = bar.get_height()
        width = bar.get_width()

        if kind == 'bar':
            x = bar.get_x() + (width / 2)
            y = height
            text = str(height)
            if y == 0:
                flag_prevent_plot = True
        else:
            x = width
            y = bar.get_y() + (bar.get_height() / 2)

            text = str(width)
            if x == 0:
                flag_prevent_plot = True

        if not flag_prevent_plot:
            ax.text(
                x=x, y=y,
                s=text,
                fontsize=fontsize,
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
                color=color)

    return ax
