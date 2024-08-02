import numpy as np
from scipy.optimize import brentq
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)

def intersect_area(ra, rb, d):
    """Calculate the area of the intersection of two circles.

    Parameters
    ----------
    ra : `float`
        The radius of the first circle
    rb : `float`
        The radius of the second circle
    d : `float`
        The separation between the two circles

    Returns
    -------
    A : `float`
        The area of the intersection of the two circles
    """
    # find the distance between the centre of circle A and the middle of the intersection
    x = (ra**2 - rb**2 + d**2) / (2 * d)

    # find the distance between the two intersection points
    y = (ra**2 - x**2)**(0.5)

    # get the angles for the triangle between the two intersection points and the centre of the circles
    theta_a = 2 * math.atan2(y, x)
    theta_b = 2 * math.atan2(y, (d - x))

    # calculate the area of the circle segments
    segment_a = 0.5 * theta_a * ra**2
    segment_b = 0.5 * theta_b * rb**2

    # calculate the area of the triangles
    triangle_a = x * y
    triangle_b = (d - x) * y

    # return the total area
    return segment_a + segment_b - triangle_a - triangle_b


def find_intersect(ra, rb, A):
    """Find the separation between two circles that gives a certain intersection area.

    Parameters
    ----------

    ra : `float`
        The radius of the first circle
    rb : `float`
        The radius of the second circle
    A : `float`
        The desired intersection area
    """
    # flip radii so ra is always the larger circle
    if rb > ra:
        ra, rb = rb, ra

    # SPECIAL CASE: intersection area is more than the area of the smaller circle (subset)
    if A >= np.pi * rb**2:
        return ra - rb
    
    # SPECIAL CASE: intersection area is zero (no overlap)
    if A == 0:
        return ra + rb

    return brentq(lambda x: intersect_area(ra, rb, x) - A, ra - rb + 1e-10, ra + rb - 1e-10)


def plot_venn(n_A, n_B, n_AB,
              A_kwargs={},
              B_kwargs={},
              show_labels=True,
              label_kwargs={},
              fig=None, ax=None):
    """Plot a Venn diagram of two circles automatically.

    Parameters
    ----------
    n_A : `int`
        The number of elements in set A
    n_B : `int`
        The number of elements in set B
    n_AB : `int`
        The number of elements in the intersection of sets A and B
    A_kwargs : `dict`, optional
        Style arguments for A circle
    B_kwargs : `dict`, optional
        Style arguments for B circle
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    default_A_kwargs = {
        "facecolor": (.1, 1, .5, 0.2),
        "edgecolor": "tab:green",
        "linewidth": 1,
    }
    default_A_kwargs.update(A_kwargs)

    default_B_kwargs = {
        "facecolor": (.1, .5, 1, 0.2),
        "edgecolor": "tab:blue",
        "linewidth": 1,     
    }
    default_B_kwargs.update(B_kwargs)
              

    ra = np.sqrt(n_A / np.pi)
    rb = np.sqrt(n_B / np.pi)
    d = find_intersect(ra, rb, n_AB)

    circles = [
        mpl.patches.Circle((0, 0), ra, **default_A_kwargs),
        mpl.patches.Circle((d, 0), rb, **default_B_kwargs)
    ]

    if show_labels:
        dlk = {
            "fontsize": 20,
        }
        dlk.update(label_kwargs)
        ax.annotate(f"{n_A - n_AB}", xy=(np.mean([-ra, d - rb]), 0), ha="center", va="center", **dlk)
        ax.annotate(f"{n_AB}", xy=(np.mean([ra, d - rb]), 0), ha="center", va="center", **dlk)
        ax.annotate(f"{n_B - n_AB}", xy=(np.mean([ra, d + rb]), 0), ha="center", va="center", **dlk)

    for c in circles:
        ax.add_artist(c)

    r, R = min(ra, rb), max(ra, rb)
    lower_lim = -R - 1
    upper_lim = d + r + 1

    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(-R - 1, R + 1)

    ax.axis("off")
    ax.set_aspect("equal")

    return fig, ax
