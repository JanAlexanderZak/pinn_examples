import os
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

# Style constants
CMAP_SEQUENTIAL = "viridis"
CMAP_DIVERGING = "RdBu_r"

FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 11
FONT_FAMILY = "serif"

DPI_SAVE = 300

FIGSIZE_HEATMAP_COMPARISON = (8, 10)
FIGSIZE_SINGLE_HEATMAP = (9, 5)
FIGSIZE_LINE_PLOT = (7, 4)


@contextmanager
def pinn_style():
    """Context manager that temporarily applies PINN publication style."""
    prev = mpl.rcParams.copy()
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "figure.dpi": 100,
        "savefig.dpi": DPI_SAVE,
        "savefig.bbox": "tight",
        "axes.grid": False,
    })
    try:
        yield
    finally:
        mpl.rcParams.update(prev)


def save_figure(fig, path, dpi=DPI_SAVE):
    """Save figure, creating parent directories if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_comparison(
    exact,
    pred,
    x_coords,
    y_coords,
    x_label,
    y_label,
    field_label,
    save_path,
    title=None,
):
    """Plot 3-row heatmap comparison: Exact, PINN, Difference.

    Args:
        exact: 2D array (ny, nx) of exact/analytical solution.
        pred: 2D array (ny, nx) of PINN prediction.
        x_coords: 1D array of x-axis coordinates.
        y_coords: 1D array of y-axis coordinates.
        x_label: Label for x-axis.
        y_label: Label for y-axis.
        field_label: Colorbar label (e.g. "T(x,t)", "u(x,t)").
        save_path: File path to save the figure.
        title: Optional figure title.
    """
    diff = exact - pred
    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

    vmin = min(exact.min(), pred.min())
    vmax = max(exact.max(), pred.max())

    diff_abs_max = max(abs(diff.min()), abs(diff.max()))

    with pinn_style():
        fig, axes = plt.subplots(3, 1, figsize=FIGSIZE_HEATMAP_COMPARISON)

        subtitles = ["Exact", "PINN", "Difference"]
        datasets = [exact, pred, diff]
        cmaps = [CMAP_SEQUENTIAL, CMAP_SEQUENTIAL, CMAP_DIVERGING]
        vmins = [vmin, vmin, -diff_abs_max]
        vmaxs = [vmax, vmax, diff_abs_max]

        for ax, data, subtitle, cmap, vn, vx in zip(
            axes, datasets, subtitles, cmaps, vmins, vmaxs
        ):
            im = ax.imshow(
                data,
                extent=extent,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=vn,
                vmax=vx,
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.10)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(field_label)
            ax.set_ylabel(f"{y_label} ({subtitle})")

        # Only show x-axis label/ticks on bottom subplot
        for ax in axes[:-1]:
            ax.set_xticklabels([])
        axes[-1].set_xlabel(x_label)

        if title:
            fig.suptitle(title, fontsize=FONT_SIZE_TITLE + 2, y=1.01)

        save_figure(fig, save_path)


def plot_single_heatmap(
    data,
    extent,
    title,
    x_label,
    y_label,
    cbar_label,
    save_path,
    scatter_data=None,
    scatter_kwargs=None,
    vlines=None,
):
    """Plot a single heatmap with optional scatter overlay and vertical lines.

    Args:
        data: 2D array to plot.
        extent: [x_min, x_max, y_min, y_max].
        title: Plot title.
        x_label: Label for x-axis.
        y_label: Label for y-axis.
        cbar_label: Colorbar label.
        save_path: File path to save the figure.
        scatter_data: Optional (x, y) tuple of arrays for scatter overlay.
        scatter_kwargs: Optional dict of kwargs for scatter plot (label, marker, etc.).
        vlines: Optional list of x-positions for vertical lines.
    """
    with pinn_style():
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_HEATMAP)

        im = ax.imshow(
            data,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap=CMAP_SEQUENTIAL,
            interpolation="nearest",
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        fig.colorbar(im, cax=cax)

        if scatter_data is not None:
            kwargs = scatter_kwargs or {}
            ax.plot(scatter_data[0], scatter_data[1], **kwargs)

        if vlines is not None:
            y_line = np.linspace(extent[2], extent[3], 2).reshape(-1, 1)
            for vx in vlines:
                ax.plot(vx * np.ones((2, 1)), y_line, "w-", linewidth=1)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        if scatter_data is not None and scatter_kwargs and "label" in scatter_kwargs:
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.9, -0.05),
                ncol=5,
                frameon=False,
            )

        save_figure(fig, save_path)


def plot_line(
    x,
    ys,
    labels,
    title,
    xlabel,
    ylabel,
    save_path,
    colors=None,
    hlines=None,
):
    """Plot one or more line series with optional horizontal reference lines.

    Args:
        x: 1D array of x values.
        ys: List of 1D arrays of y values.
        labels: List of labels for each series.
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        save_path: File path to save the figure.
        colors: Optional list of colors for each series.
        hlines: Optional list of (y_value, label, color) tuples for reference lines.
    """
    with pinn_style():
        fig, ax = plt.subplots(figsize=FIGSIZE_LINE_PLOT)

        for i, (y, label) in enumerate(zip(ys, labels)):
            color = colors[i] if colors else None
            ax.plot(x, y, label=label, color=color)

        if hlines:
            for y_val, label, color in hlines:
                ax.axhline(
                    y=y_val,
                    linestyle="--",
                    color=color,
                    label=label,
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

        save_figure(fig, save_path)


def create_gif(frame_folder, output_path, duration=400):
    """Create a GIF from PNG frames in a folder.

    Args:
        frame_folder: Path to directory containing PNG frames.
        output_path: Path to save the output GIF.
        duration: Duration of each frame in milliseconds.
    """
    frame_paths = sorted(
        Path(frame_folder).glob("*.png"),
        key=lambda p: int(p.stem),
    )
    frames = [Image.open(p) for p in frame_paths]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
