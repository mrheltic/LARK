"""
ui.theme
========
Nord-inspired dark colour palette shared across all KrakenSDR matplotlib and
tkinter UIs.

Usage
-----
Individual colours::

    from ui.theme import BG, BLUE, AMBER

Apply to matplotlib (call once at startup)::

    from ui.theme import apply_mpl_style
    apply_mpl_style()

Apply to a ttk.Style instance::

    from ui.theme import apply_tk_theme
    TK_BG, TK_BG2, TK_BG3, TK_FG, TK_ACC = apply_tk_theme(style)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG     = "#1a1d27"   # window / figure background
BG2    = "#21253a"   # axes / widget background
BG3    = "#2a2f47"   # raised elements (buttons, panels)
BORDER = "#3b4263"   # grid lines, separators
DIM    = "#4e5680"   # dimmed / inactive elements

BLUE   = "#5ea4e0"   # primary accent
TEAL   = "#4ecdc4"   # secondary accent
AMBER  = "#f4a431"   # warnings / highlight
VIOLET = "#a78bfa"   # tertiary accent
ROSE   = "#f16b6f"   # alert / negative
LIME   = "#6dd97d"   # positive / green

TEXT   = "#d8dae8"   # primary text
MUTED  = "#8891b0"   # secondary / axis labels

# ---------------------------------------------------------------------------
# Matplotlib rcParams
# ---------------------------------------------------------------------------
MPL_STYLE: dict = {
    "figure.facecolor": BG,
    "axes.facecolor":   BG2,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  MUTED,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "grid.color":       BORDER,
    "grid.alpha":       0.4,
}


def apply_mpl_style() -> None:
    """Apply the dark Nord theme to :data:`matplotlib.pyplot.rcParams`."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(MPL_STYLE)


# ---------------------------------------------------------------------------
# tkinter / ttk theme
# ---------------------------------------------------------------------------
def apply_tk_theme(style, acc: str = BLUE) -> tuple[str, str, str, str, str]:
    """
    Apply the dark Nord theme to a :class:`tkinter.ttk.Style` instance.

    Parameters
    ----------
    style : ttk.Style
    acc   : accent colour (default :data:`BLUE`)

    Returns
    -------
    tuple
        ``(TK_BG, TK_BG2, TK_BG3, TK_FG, TK_ACC)`` — convenience tuple so
        callers can immediately unpack the relevant colours for further widget
        customisation.
    """
    FONT = ("Segoe UI", 9)
    style.theme_use("clam")
    style.configure(
        ".",
        background=BG, foreground=TEXT, font=FONT,
        fieldbackground=BG2,
        selectbackground=acc, selectforeground=BG,
        troughcolor=BG3, bordercolor=BG3,
        darkcolor=BG2, lightcolor=BG2,
    )
    for w in ("TLabel", "TFrame", "TLabelframe", "TLabelframe.Label"):
        style.configure(w, background=BG, foreground=TEXT)

    style.configure("TEntry",    fieldbackground=BG2, foreground=TEXT, insertcolor=TEXT)
    style.configure("TCombobox", fieldbackground=BG2, foreground=TEXT)
    style.map("TCombobox",       fieldbackground=[("readonly", BG2)])

    style.configure(
        "TButton",
        background=BG3, foreground=TEXT,
        relief="flat", font=("Segoe UI", 10, "bold"), padding="8 4",
    )
    style.map("TButton", background=[("active", "#3d4675")])

    return BG, BG2, BG3, TEXT, acc
