"""Shared plotting style for the thesis figures.

Print-only: these are PDFs dropped into a LaTeX document, so there is no dark
mode and no hover layer.  What the style does enforce is the part that was
getting broken by hand -- categorical hues assigned in a fixed validated order,
a single-hue ramp for magnitude, recessive solid chrome, and text that never
wears a series colour.

The four categorical slots below were validated all-pairs on the light surface
(worst CVD dE 9.2, worst normal-vision dE 16.3), which is the pairlist that
applies to scatter plots.  Do not add a fifth without re-validating.
"""

# Categorical slots, fixed order.  Never cycle past the list.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

# Single-hue sequential ramp (blue, light -> dark) for magnitude.
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
       "#0d366b"]

# Ink, never a series colour.
INK        = "#0b0b0b"
INK_SOFT   = "#52514e"
INK_MUTED  = "#8a8984"
GRID       = "#e3e2de"
SURFACE    = "#ffffff"


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("thesis_blue", SEQ)


def on_color(hexstr) -> str:
    """Ink for text drawn on top of a filled cell: pick by relative luminance so
    a label is never white-on-yellow."""
    r, g, b = (int(hexstr[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
           for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return INK if lum > 0.42 else "#ffffff"


def apply(plt) -> None:
    """Recessive chrome: hairline solid grid and spines, ink-coloured text."""
    plt.rcParams.update({
        "figure.facecolor":  SURFACE,
        "axes.facecolor":    SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   INK,
        "axes.titlesize":    9.5,
        "axes.labelsize":    9,
        "axes.linewidth":    0.6,
        "axes.grid":         True,
        "grid.color":        GRID,
        "grid.linewidth":    0.6,
        "grid.linestyle":    "-",      # solid: never dash the grid
        "grid.alpha":        1.0,
        "xtick.color":       INK_SOFT,
        "ytick.color":       INK_SOFT,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize":   8,
        "legend.frameon":    True,
        "legend.framealpha": 0.92,
        "legend.edgecolor":  GRID,
        "legend.facecolor":  SURFACE,
        "font.size":         9,
        "lines.linewidth":   1.6,
    })


def strip_spines(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def sequential_cmap_visible():
    """Sequential ramp for marks drawn on the white surface: starts at step 250
    so the lightest mark still clears 2:1 contrast instead of disappearing."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("thesis_blue_vis", SEQ[3:])
