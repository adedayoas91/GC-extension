"""Visualisation utilities for Granger causality results.

Includes:
* :class:`TopographyVisualizer` – connectivity matrix on a 3-D brain map.
* Hemisphere / motoneuron graph helpers (ported from ``others_funcs``).
* Hindbrain brain-image overlays.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..constants import (
    RGB_LEFT_HEMISPHERE,
    RGB_RIGHT_HEMISPHERE,
    SIGNIFICANCE_NS,
    SIGNIFICANCE_THRESHOLDS,
)
from ..core.granger_causality import GrangerCausality


# ---------------------------------------------------------------------------
# Topography visualiser
# ---------------------------------------------------------------------------


class TopographyVisualizer(GrangerCausality):
    """Visualise connectivity on a 3-D brain topography.

    Inherits from :class:`~gc_extension.core.GrangerCausality` to reuse
    the fitted correlation matrices and connectivity.

    Parameters
    ----------
    n_perm, n_pasts, n_lags : int
        Forwarded to :class:`GrangerCausality`.
    """

    def __init__(self, n_perm: int, n_pasts: int, n_lags: int) -> None:
        super().__init__(
            n_perm=n_perm,
            n_pasts=n_pasts,
            n_lags=n_lags,
            temporal=True,
            method="cgc",
        )
        self.topography: Optional[np.ndarray] = None
        self.inf: Optional[np.ndarray] = None
        self.roi_count: Optional[int] = None

    def repopulate(self, roi_idx: np.ndarray) -> np.ndarray:
        """Embed sub-population connectivity into the full population matrix.

        Parameters
        ----------
        roi_idx : np.ndarray
            Integer indices of the analysed ROIs in the full population.

        Returns
        -------
        np.ndarray
            Full-population binary connectivity matrix.
        """
        inferred_full = np.zeros((self.roi_count, self.roi_count))
        positions = np.transpose(np.where(self.inf != 0))
        for i in range(positions.shape[0]):
            inferred_full[roi_idx[positions[i, 0]], roi_idx[positions[i, 1]]] = 1
        return inferred_full

    def get_coordinates(
        self, inferred: np.ndarray
    ) -> Tuple[list, list, list, np.ndarray]:
        """Extract 3-D endpoint coordinates for each active connection.

        Parameters
        ----------
        inferred : np.ndarray
            Full-population connectivity matrix.

        Returns
        -------
        x_val, y_val, z_val : list
            Start/end coordinate pairs for each edge.
        t : np.ndarray
            Array of ``[from, to]`` index pairs.
        """
        t = np.transpose(np.where(inferred > 0))
        x_val, y_val, z_val = [], [], []
        for i in range(len(t)):
            p1 = self.topography[t[i, 0], :2]
            p2 = self.topography[t[i, 1], :2]
            x_val.append([p1[0], p2[0]])
            y_val.append([p1[1], p2[1]])
            if self.topography.shape[1] == 3:
                z_val.append(
                    [self.topography[t[i, 0], 2], self.topography[t[i, 1], 2]]
                )
        return x_val, y_val, z_val, t

    def plot_conn_mat_on_topography(
        self,
        topography: np.ndarray,
        inferred: np.ndarray,
        roi_idx: np.ndarray,
    ):
        """3-D visualisation of the connectivity matrix on the brain surface.

        Parameters
        ----------
        topography : np.ndarray
            3-D pixel coordinates of all ROIs, shape ``[n_rois, 3]``.
        inferred : np.ndarray
            Inferred connectivity matrix for the sub-population.
        roi_idx : np.ndarray
            Integer indices mapping sub-population rows to *topography*.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self.topography = topography
        self.inf = inferred
        self.roi_count = topography.shape[0]

        inferred_rep = self.repopulate(roi_idx=roi_idx)
        x_val, y_val, z_val, _ = self.get_coordinates(inferred_rep)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            topography[:, 0], topography[:, 1], topography[:, 2],
            color="red", s=10, marker=".",
        )
        ax.scatter(
            topography[roi_idx, 0], topography[roi_idx, 1],
            topography[roi_idx, 2],
            color="green", s=15, marker="*",
        )
        for a in range(len(x_val)):
            ax.plot(x_val[a], y_val[a], z_val[a], lw=0.7, alpha=0.7)
        ax.grid(False)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        return fig

    def plot_correlation_with_behavior(
        self, corr_with_beh: np.ndarray
    ) -> plt.Figure:
        """Scatter plot of per-ROI behavioral correlation on the topography.

        Parameters
        ----------
        corr_with_beh : np.ndarray
            Correlation coefficient for each ROI.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(10, 6))
        if self.topography.shape[1] == 2:
            plt.scatter(
                self.topography[:, 0], self.topography[:, 1],
                marker="o", s=10, c=corr_with_beh, cmap="seismic",
            )
            plt.colorbar()
        else:
            ax = fig.add_subplot(111, projection="3d")
            p = ax.scatter(
                self.topography[:, 0], self.topography[:, 1],
                self.topography[:, 2],
                marker="o", s=10, c=corr_with_beh, cmap="seismic",
            )
            fig.colorbar(p)
        return fig


# ---------------------------------------------------------------------------
# Statistical annotation helper
# ---------------------------------------------------------------------------


def pval_to_star(pvalue: float) -> str:
    """Convert a p-value to an asterisk-based significance marker.

    Parameters
    ----------
    pvalue : float

    Returns
    -------
    str
        Significance marker (``"****"``, ``"***"``, ``"**"``, ``"*"``,
        or ``"ns"``).
    """
    for threshold, marker in SIGNIFICANCE_THRESHOLDS:
        if pvalue <= threshold:
            return marker
    return SIGNIFICANCE_NS


# ---------------------------------------------------------------------------
# Hemisphere layout helpers (motoneuron circuits)
# ---------------------------------------------------------------------------


def get_label_and_color_lists(
    mid: int, tot: int
) -> Tuple[list, list]:
    """Build label and colour lists for a left/right hemisphere layout.

    Parameters
    ----------
    mid : int
        Number of left-hemisphere nodes.
    tot : int
        Total number of nodes.

    Returns
    -------
    Tuple[list, list]
        ``(label_list, color_list)``
    """
    label_list = [2 * i + 1 for i in range(mid)]
    label_list.extend([2 * i + 2 for i in range(tot - mid)])
    color_list = [RGB_LEFT_HEMISPHERE] * mid
    color_list.extend([RGB_RIGHT_HEMISPHERE] * (tot - mid))
    return label_list, color_list


def get_coords_left(n_points: int) -> np.ndarray:
    """Return x-y coordinates on the left semicircle.

    Parameters
    ----------
    n_points : int
        Number of nodes on the left side.

    Returns
    -------
    np.ndarray
        Shape ``[2, n_points]``.
    """
    half = int(n_points / 2)
    x = np.linspace(-1 / half, -1, half)
    if n_points % 2 == 1:
        x = np.concatenate((x, [-1.1]))
    x = np.concatenate((x, np.linspace(-1, -1 / half, half)))
    y = np.linspace(1, -1, n_points)
    return np.array([x, y])


def get_coords_right(n_points: int) -> np.ndarray:
    """Return x-y coordinates on the right semicircle.

    Parameters
    ----------
    n_points : int
        Number of nodes on the right side.

    Returns
    -------
    np.ndarray
        Shape ``[2, n_points]``.
    """
    half = int(n_points / 2)
    x = np.linspace(1 / half, 1, half)
    if n_points % 2 == 1:
        x = np.concatenate((x, [1.1]))
    x = np.concatenate((x, np.linspace(1, 1 / half, half)))
    y = np.linspace(1, -1, n_points)
    return np.array([x, y])


def get_coords(mid: int, tot: int) -> np.ndarray:
    """Calculate node coordinates on a bipartite semicircle layout.

    Parameters
    ----------
    mid : int
        Number of left-side nodes.
    tot : int
        Total number of nodes.

    Returns
    -------
    np.ndarray
        Shape ``[tot, 2]``.
    """
    left = get_coords_left(mid)
    right = get_coords_right(tot - mid)
    return np.concatenate([left.T, right.T])


def plot_gc_matrix(gc: np.ndarray, mid: int) -> None:
    """Visualise a GC matrix with left/right hemisphere colour coding.

    Parameters
    ----------
    gc : np.ndarray
        GC connectivity matrix.
    mid : int
        Number of left-hemisphere nodes.
    """
    n_cells = len(gc)
    gc[gc == 0] = np.nan
    plt.imshow(gc, cmap="YlOrRd")
    ax = plt.gca()
    ax.set_xticks(np.arange(n_cells))
    ax.set_yticks(np.arange(n_cells))
    label_list, color_list = get_label_and_color_lists(mid, n_cells)
    ax.set_xticklabels(label_list)
    ax.set_yticklabels(label_list)
    for color, tick in zip(color_list, ax.xaxis.get_major_ticks()):
        tick.label1.set_color(color)
    for color, tick in zip(color_list, ax.yaxis.get_major_ticks()):
        tick.label1.set_color(color)
    plt.xlabel("to neuron", size=20)
    plt.ylabel("from neuron", size=20)
    plt.tight_layout()


def plot_directed_graph(gc: np.ndarray, mid: int, hide_digits: bool = False) -> None:
    """Draw the directed motoneuron network on a circle layout.

    Node size is proportional to the ipsilateral delta (GC_out - GC_in).

    Parameters
    ----------
    gc : np.ndarray
        GC connectivity matrix.
    mid : int
        Number of left-hemisphere nodes.
    hide_digits : bool
        If ``True``, node labels are omitted.
    """
    textsize = 35
    cmap = plt.cm.Greys
    n_cells = len(gc)
    gc_ipsi_left = gc[:mid, :mid]
    gc_ipsi_right = gc[mid:, mid:]

    d_in_ipsi_left = np.nansum(gc_ipsi_left, axis=0)
    d_out_ipsi_left = np.nansum(gc_ipsi_left, axis=1)
    d_in_ipsi_right = np.nansum(gc_ipsi_right, axis=0)
    d_out_ipsi_right = np.nansum(gc_ipsi_right, axis=1)

    delta_ipsi_left = d_out_ipsi_left - d_in_ipsi_left
    delta_ipsi_right = d_out_ipsi_right - d_in_ipsi_right

    circle_centers = get_coords(mid, n_cells)
    circle_size = np.concatenate([delta_ipsi_left, delta_ipsi_right])
    max_size = np.max(np.abs(circle_size))
    circle_size = circle_size / max_size if max_size > 0 else circle_size

    for i, center in enumerate(circle_centers):
        color = "firebrick" if circle_size[i] > 0 else ("navy" if circle_size[i] < 0 else "purple")
        plt.scatter(
            center[0], center[1],
            s=100 + np.abs(circle_size[i]) * 400,
            c=color,
        )

    plt.axis("equal")
    gc_flat = gc.flatten()
    gc_flat[np.isnan(gc_flat)] = 0
    cells_sorted = np.argsort(gc_flat)
    gc_max = gc_flat[cells_sorted[-1]]
    gc_min = gc_flat[cells_sorted[0]]

    for cell in cells_sorted:
        gc_cell = gc_flat[cell]
        if gc_cell > 0:
            prop = (gc_cell - gc_min) / (gc_max - gc_min) if gc_max != gc_min else 0.0
            color = cmap(prop)
            width = prop * 0.05
            cell_from = int(cell / len(circle_centers))
            cell_to = cell % len(circle_centers)
            plt.arrow(
                circle_centers[cell_from, 0], circle_centers[cell_from, 1],
                circle_centers[cell_to, 0] - circle_centers[cell_from, 0],
                circle_centers[cell_to, 1] - circle_centers[cell_from, 1],
                color=color, width=width, length_includes_head=True,
            )

    label_list, color_list = get_label_and_color_lists(mid, n_cells)
    if not hide_digits:
        for label, color, pos in zip(label_list, color_list, circle_centers):
            plt.text(
                pos[0] + np.sign(pos[0]) * 0.4, pos[1], label,
                va="center", ha="center", color=color,
                size=textsize, fontweight="bold",
            )
    plt.axis("off")
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Hindbrain brain-image overlays
# ---------------------------------------------------------------------------


def plot_background_and_cells(
    cell_centers: np.ndarray,
    subset_neurons: np.ndarray,
    background: np.ndarray,
    subplots: bool = False,
    invert: bool = False,
):
    """Overlay cell positions on a brain-image background.

    Parameters
    ----------
    cell_centers : np.ndarray
        ``[n_cells, 2]`` pixel coordinates.
    subset_neurons : np.ndarray
        Indices of the highlighted subset.
    background : np.ndarray
        2-D image array.
    subplots : bool
        If ``True``, create two panels (rostral / caudal).
    invert : bool
        Mirror x coordinates (for fish 2, 5, 6, 8).

    Returns
    -------
    matplotlib.figure.Figure or Tuple
    """
    if subplots:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
        for ax in (ax1, ax2):
            ax.imshow(background, cmap=plt.cm.gist_yarg)
        ax1.set_title("Rostral to caudal", fontsize=25)
        ax2.set_title("Caudal to rostral", fontsize=25)
        _scatter_cells_on_axes(ax1, ax2, cell_centers, subset_neurons, invert)
        res = fig, (ax1, ax2)
    else:
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(
            background, cmap=plt.cm.gist_yarg,
            aspect="equal", vmax=np.max(background) / 2,
        )
        _scatter_cells(cell_centers, subset_neurons, invert)
        res = fig
    plt.axis("off")
    plt.tight_layout()
    return res


def _scatter_cells_on_axes(ax1, ax2, cell_centers, subset_neurons, invert):
    """Helper: scatter cells on two axes."""
    x = 512 - cell_centers[:, 0] if invert else cell_centers[:, 0]
    for ax in (ax1, ax2):
        ax.scatter(x, cell_centers[:, 1], color="silver", edgecolor="black", s=100)
        ax.scatter(
            x[subset_neurons], cell_centers[subset_neurons, 1],
            color="crimson", edgecolor="black", s=100,
        )


def _scatter_cells(cell_centers, subset_neurons, invert):
    """Helper: scatter cells on the current pyplot figure."""
    x = 512 - cell_centers[:, 0] if invert else cell_centers[:, 0]
    plt.scatter(x, cell_centers[:, 1], color="silver", edgecolor="black", s=100)
    plt.scatter(
        x[subset_neurons], cell_centers[subset_neurons, 1],
        color="crimson", edgecolor="black", s=100,
    )


def plot_drive(
    cell_centers: np.ndarray,
    subset_neurons: np.ndarray,
    drive: np.ndarray,
    background: np.ndarray,
    default_color: str = "silver",
    cst: float = 1,
    invert: bool = False,
) -> plt.Figure:
    """Plot cell positions scaled by their drive value.

    Parameters
    ----------
    cell_centers, subset_neurons, drive, background, default_color, cst, invert :
        See :func:`plot_background_and_cells`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(
        background, cmap=plt.cm.gist_yarg,
        aspect="equal", vmax=np.max(background) / 2,
    )
    next_sub = 0
    for j in range(len(cell_centers)):
        if next_sub < len(subset_neurons) and j == subset_neurons[next_sub]:
            color = "crimson"
            next_sub += 1
        else:
            color = default_color
        x = 512 - cell_centers[j, 0] if invert else cell_centers[j, 0]
        plt.scatter(
            x, cell_centers[j, 1], color=color, edgecolor="black",
            s=100 * (int(drive[j] * cst) + 1), alpha=0.8,
        )
    plt.axis("off")
    return fig
