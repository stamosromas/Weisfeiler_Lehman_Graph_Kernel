"""
k-means where cluster assignment and center updates use the Wasserstein
(earth-mover) distance/barycenter over two independent histograms per
vector: a "row" half (indices [0:n_r]) and a "column" half ([n_r:]).
"""

import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import ot

MAX_BARYCENTER_RETRIES = 8


class ModifiedWassersteinKMeans:
    def __init__(
        self,
        vecs: List[np.ndarray],
        costMat_r: np.ndarray,
        costMat_c: np.ndarray,
        clusts: int,
        numOfIters: int,
        n_r: int,
        n_c: int,
        num_threads: int = 16,
        seed: int = 42,
    ):
        if len(vecs) < clusts:
            raise ValueError("Number of clusters (k) cannot exceed the number of data points.")

        self.X = vecs
        self.M_r = costMat_r
        self.M_c = costMat_c
        self.k = clusts
        self.numOfIters = numOfIters
        self.n_r = n_r
        self.n_c = n_c
        self.num_threads = num_threads
        self.rng = random.Random(seed)
        self.res: Dict[int, List[np.ndarray]] = {i: [] for i in range(clusts)}

        self.assignmentList = self.main()

    @staticmethod
    def normalize_histogram(vec: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        total = np.sum(vec)
        norm_vec = vec / total if total > 0 else np.ones_like(vec, dtype=float) / len(vec)
        return np.clip(norm_vec, epsilon, None)

    def wasserstein_distance(self, vec1: np.ndarray, vec2: np.ndarray, M: np.ndarray) -> float:
        a = self.normalize_histogram(vec1)
        b = self.normalize_histogram(vec2)
        if not np.allclose(np.sum(a), np.sum(b), atol=1e-8):
            raise ValueError("Histogram vectors must have the same (approx) sum.")
        return ot.emd2(a, b, M)

    def wasserstein_barycenter(
        self,
        vecs: List[np.ndarray],
        M: np.ndarray,
        reg: float = 1e-1,
        numItermax: int = 20000,
        stopThr: float = 1e-3,
    ) -> np.ndarray:
        if len(vecs) == 0:
            return np.ones(M.shape[0]) / M.shape[0]

        histograms = np.stack([self.normalize_histogram(v) for v in vecs], axis=1)
        sums = np.sum(histograms, axis=0)
        if not np.allclose(sums, sums[0], atol=1e-8):
            raise ValueError("All input histograms must have the same total for barycenter calculation.")

        current_reg = reg
        for attempt in range(MAX_BARYCENTER_RETRIES):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", category=RuntimeWarning)
                    return ot.bregman.barycenter(
                        histograms, M, current_reg, numItermax=numItermax, stopThr=stopThr
                    )
            except RuntimeWarning as exc:
                current_reg *= 2
                warnings.warn(
                    f"Barycenter computation unstable (attempt {attempt + 1}); "
                    f"increasing reg to {current_reg} and retrying ({exc}).",
                    stacklevel=2,
                )

        raise RuntimeError(
            f"Wasserstein barycenter failed to converge after {MAX_BARYCENTER_RETRIES} "
            f"reg-doubling retries."
        )

    def find_nearest_center(self, vec: np.ndarray, centers_r: List[np.ndarray], centers_c: List[np.ndarray]) -> int:
        vec_r, vec_c = vec[: self.n_r], vec[self.n_r :]
        if not centers_r or not centers_c:
            raise ValueError("Centers cannot be empty when assigning clusters.")

        distances = [
            self.wasserstein_distance(vec_r, centers_r[i], self.M_r)
            + self.wasserstein_distance(vec_c, centers_c[i], self.M_c)
            for i in range(self.k)
        ]
        return int(np.argmin(distances))

    def main(self) -> List[int]:
        initial_centers = self.rng.sample(list(self.X), self.k)
        centers_r = [self.normalize_histogram(vec[: self.n_r]) for vec in initial_centers]
        centers_c = [self.normalize_histogram(vec[self.n_r :]) for vec in initial_centers]

        assignment_list = [0] * len(self.X)

        for _ in range(self.numOfIters):
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                assignment_list = list(
                    executor.map(
                        lambda i: self.find_nearest_center(self.X[i], centers_r, centers_c),
                        range(len(self.X)),
                    )
                )

            clusters_r: Dict[int, List[np.ndarray]] = {i: [] for i in range(self.k)}
            clusters_c: Dict[int, List[np.ndarray]] = {i: [] for i in range(self.k)}
            self.res = {i: [] for i in range(self.k)}

            for i, data_point in enumerate(self.X):
                cluster = assignment_list[i]
                self.res[cluster].append(data_point)
                clusters_r[cluster].append(data_point[: self.n_r])
                clusters_c[cluster].append(data_point[self.n_r :])

            for i in range(self.k):
                if clusters_r[i]:
                    centers_r[i] = self.wasserstein_barycenter(clusters_r[i], self.M_r)
                if clusters_c[i]:
                    centers_c[i] = self.wasserstein_barycenter(clusters_c[i], self.M_c)

        return assignment_list


    findNearestCenter = find_nearest_center
