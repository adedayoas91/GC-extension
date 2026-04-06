"""Evaluation metrics for Granger causality inference."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.granger_causality import causalisedGrangerCausality  # noqa: F401


class ComputeMetrics:
    """Compute performance metrics for an inferred connectivity matrix.

    Parameters
    ----------
    conn_mat : np.ndarray
        Inferred connectivity matrix.
    ground_truth : np.ndarray
        Ground truth adjacency matrix.
    n_pasts : int
        Number of past states used during inference.
    """

    def __init__(
        self,
        conn_mat: np.ndarray,
        ground_truth: np.ndarray,
        n_pasts: int,
    ) -> None:
        self.conn_mat = conn_mat
        self.gt = ground_truth
        self.n_pasts = n_pasts
        self.confusion_matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute the confusion matrix comparing *conn_mat* to *gt*.

        Returns
        -------
        np.ndarray
            ``[[TP, FN], [FP, TN]]``.
        """
        a = self.gt
        inferred = self.conn_mat
        tp = int(np.sum(np.logical_and(a != 0, inferred != 0)))
        fn = int(np.sum(np.logical_and(a != 0, inferred == 0)))
        fp = int(np.sum(np.logical_and(a == 0, inferred != 0)))
        tn = int(np.sum(np.logical_and(a == 0, inferred == 0)))
        self.confusion_matrix = np.array([[tp, fn], [fp, tn]])
        return self.confusion_matrix

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def compute_metrics_from_confusion_matrix(self) -> np.ndarray:
        """Compute accuracy, precision, recall and FPR.

        Returns
        -------
        np.ndarray
            ``[accuracy, precision, recall, fpr]``.
        """
        cm = self.confusion_matrix.flatten()
        tp, fn, fp, tn = int(cm[0]), int(cm[1]), int(cm[2]), int(cm[3])
        total = tp + fn + fp + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return np.array([accuracy, precision, recall, fpr])

    # ------------------------------------------------------------------
    # Graph / topology utilities
    # ------------------------------------------------------------------

    @staticmethod
    def repopulate(
        inf: np.ndarray, traces: np.ndarray, idx: np.ndarray
    ) -> np.ndarray:
        """Embed a sub-population connectivity matrix into the full population.

        Parameters
        ----------
        inf : np.ndarray
            Inferred connectivity matrix for the sub-population.
        traces : np.ndarray
            Full trace array (used for shape only).
        idx : np.ndarray
            Indices of the sub-population within the full population.

        Returns
        -------
        np.ndarray
            Full-population connectivity matrix (binary).
        """
        inferred_full = np.zeros((traces.shape[0], traces.shape[0]))
        positions = np.transpose(np.where(inf != 0))
        for i in range(len(positions)):
            inferred_full[idx[positions[i, 0]], idx[positions[i, 1]]] = 1
        return inferred_full

    def compute_projection_distances(
        self, centers: np.ndarray
    ) -> np.ndarray:
        """Compute Euclidean distances for each inferred projection.

        Parameters
        ----------
        centers : np.ndarray
            3-D coordinates for each ROI, shape ``[n_rois, 3]``.

        Returns
        -------
        np.ndarray
            Distance for each non-zero entry in ``conn_mat``.
        """
        loc = np.transpose(np.where(self.conn_mat > 0))
        dist = np.zeros(len(loc))
        for i in range(len(loc)):
            p1, p2 = centers[loc[i, 0]], centers[loc[i, 1]]
            dist[i] = np.sqrt(
                (p2[0] - p1[0]) ** 2
                + (p2[1] - p1[1]) ** 2
                + (p2[2] - p1[2]) ** 2
            )
        return dist

    @staticmethod
    def count_in_out_edges(
        out_nodes: list,
        in_nodes: list,
        emitters: list,
        receivers: list,
    ) -> Tuple[int, int]:
        """Count unique out- and in-edges after removing known emitters/receivers.

        Parameters
        ----------
        out_nodes : list
            Candidate emitter node IDs.
        in_nodes : list
            Candidate receiver node IDs.
        emitters : list
            Known emitter IDs to exclude.
        receivers : list
            Known receiver IDs to exclude.

        Returns
        -------
        Tuple[int, int]
            ``(n_out, n_in)`` after exclusion.
        """
        n_out = len(out_nodes) - sum(1 for el in out_nodes if el in emitters)
        n_in = len(in_nodes) - sum(1 for el in in_nodes if el in receivers)
        return n_out, n_in

    def roi_neighbors(self) -> Tuple[dict, dict]:
        """Return outgoing and incoming neighbour maps for each node.

        Returns
        -------
        Tuple[dict, dict]
            ``(nodes_out, nodes_in)`` where each value is a NumPy array
            of neighbour indices.
        """
        inferred = self.conn_mat
        nodes_out = {
            i: np.where(inferred[i] != 0)[0]
            for i in range(inferred.shape[0])
        }
        nodes_in = {
            i: np.where(inferred.T[i] != 0)[0]
            for i in range(inferred.shape[0])
        }
        return nodes_out, nodes_in
