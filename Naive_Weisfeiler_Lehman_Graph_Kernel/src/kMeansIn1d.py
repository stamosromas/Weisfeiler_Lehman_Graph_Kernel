"""
Simple, deterministic (seeded) 1-D k-means, used to bucket scalar node
statistics (out-degree, descendant count, ...) into k clusters for labeling.
"""

import random


class kMeans1d:
    def __init__(self, dataSet, k, numOfIters, seed=42):
        self.dataset = dataSet
        self.k = k
        self.numOfIters = numOfIters
        self.assignments = []
        self.centers = []
        self.rng = random.Random(seed)
        self.main()

    def select_initial_centers(self):
        distinct_vals = list(set(self.dataset))
        if len(distinct_vals) < self.k:
            raise ValueError(
                "Not enough distinct values (%d) to form %d clusters."
                % (len(distinct_vals), self.k)
            )
        return sorted(self.rng.sample(distinct_vals, self.k))

    @staticmethod
    def compute_assignment(val, centers):
        return min(range(len(centers)), key=lambda i: abs(val - centers[i]))

    def update_centers(self, assignments):
        cluster_dict = self.get_cluster_dict(assignments)
        new_centers = []
        for c in range(self.k):
            points = cluster_dict.get(c)
            if not points:
                new_centers.append(self.rng.choice(self.dataset))
            else:
                new_centers.append(float(sum(points)) / len(points))
        return new_centers

    @staticmethod
    def check_for_convergence(old_centers, new_centers, tol=1e-9):
        return all(abs(oc - nc) <= tol for oc, nc in zip(old_centers, new_centers))

    def get_cluster_dict(self, assignments):
        cluster_to_vals = {}
        for i, cluster_idx in enumerate(assignments):
            cluster_to_vals.setdefault(cluster_idx, []).append(self.dataset[i])
        return cluster_to_vals

    def main(self):
        current_centers = self.select_initial_centers()

        for _ in range(self.numOfIters):
            assignments = [self.compute_assignment(v, current_centers) for v in self.dataset]
            new_centers = self.update_centers(assignments)
            if self.check_for_convergence(current_centers, new_centers):
                current_centers = new_centers
                break
            current_centers = new_centers

        self.centers = current_centers
        self.assignments = [self.compute_assignment(v, current_centers) for v in self.dataset]

    selectInitialCenters = select_initial_centers
    computeAssignment = staticmethod(compute_assignment)
    updateCenters = update_centers
    checkForConvergence = staticmethod(check_for_convergence)
    getClusterDict = get_cluster_dict