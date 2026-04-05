import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union, Any, Tuple, Dict
import math

__all__ = ["DBSCAN", "HDBSCAN", "OPTICS"]

from joblib import Parallel, delayed

try:
    from torch.func import vmap
except ImportError:
    from torch import vmap
from .....models.utils import MLCluster
from ...regression.knn.knn import BallTree, KDTree
import joblib


class DBSCAN(MLCluster):
    def __init__(self,
                 eps: float = 0.5,
                 min_samples: int = 5,
                 metric: Union[str, Callable, nn.Module] = "euclidean",
                 metric_params: dict = None,
                 algorithm: Union[str, Callable, nn.Module] = "auto",
                 leaf_size: int = 30,
                 p: float = None,
                 n_jobs: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        if metric_params is None:
            metric_params = {}
        metric_params["p"] = p
        self._create_metric(metric, metric_params)
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype
        self.algorithm = algorithm
        self.p = p
        self.core_sample_indices_ = None
        self.components_ = None
        self.labels_ = None
        self.n_features_in_ = None
        self.multiheaded = kwargs.get("multiheaded", False)
        if isinstance(algorithm, nn.Module):
            self.module_kwargs = kwargs.get("module_kwargs", None)
        else:
            self.module_kwargs = None

    def _init_module(self, X: torch.Tensor):
        self.n_features_in_ = X.size(-1)
        if isinstance(self.algorithm, str):
            if self.algorithm == "auto":
                # Heuristic: KDTree for small dimensions, BallTree for higher
                if self.n_features_in_ <= 30:
                    algo_cls = KDTree
                else:
                    algo_cls = BallTree

                self.algorithm = algo_cls(X,
                                          leaf_size=self.leaf_size,
                                          metric=self.metric,
                                          device=self.device,
                                          dtype=self.dtype,
                                          n_jobs=self.n_jobs)
            elif self.algorithm == "ball_tree":
                self.algorithm = BallTree(X,
                                          leaf_size=self.leaf_size,
                                          metric=self.metric,
                                          device=self.device,
                                          dtype=self.dtype,
                                          n_jobs=self.n_jobs)
            elif self.algorithm == "kd_tree":
                self.algorithm = KDTree(X,
                                        leaf_size=self.leaf_size,
                                        metric=self.metric,
                                        device=self.device,
                                        dtype=self.dtype,
                                        n_jobs=self.n_jobs)
            elif self.algorithm == "brute":
                self.algorithm = "brute"  # Handled in fit/query
        elif isinstance(self.algorithm, nn.Module):
            self.algorithm = self.algorithm(**self.module_kwargs)
        return self

    def fit(self, data_or_X, y=None, sample_weight=None, **kwargs):
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if self._metric_name != "precomputed":
            n_samples = X.size(-2)
            algo_str = isinstance(self.algorithm, str) and self.algorithm in ("auto", "ball_tree", "kd_tree")
            use_brute = self.algorithm == "brute" or (algo_str and n_samples <= 50_000)
            if not use_brute:
                self._init_module(X)
        else:
            n_samples = X.size(0)
            use_brute = False
            if X.size(0) != X.size(1):
                raise ValueError("Precomputed distance matrix must be square.")

        self.labels_ = torch.full((n_samples,), -1, device=self.device, dtype=torch.long)

        if self._metric_name == "precomputed":
            neighbors_mask = X <= self.eps
        elif use_brute:
            dist_matrix = self.metric(X, X)
            neighbors_mask = dist_matrix <= self.eps
        elif isinstance(self.algorithm, (BallTree, KDTree)):
            neighbors = self.algorithm.query_radius(X, r=self.eps, return_distance=False)
            counts = torch.tensor([n.numel() if hasattr(n, 'numel') else len(n) for n in neighbors], device=self.device)
            row_indices = torch.repeat_interleave(torch.arange(n_samples, device=self.device), counts)
            col_indices = torch.cat([n.to(self.device).long() if isinstance(n, torch.Tensor) else torch.as_tensor(n, device=self.device, dtype=torch.long) for n in neighbors])
            neighbors_mask = torch.zeros((n_samples, n_samples), dtype=torch.bool, device=self.device)
            neighbors_mask[row_indices, col_indices] = True
        else:
            dist_matrix = self.metric(X, X)
            neighbors_mask = dist_matrix <= self.eps

        neighborhood_counts = neighbors_mask.sum(dim=1).to(self.dtype)
        core_samples_mask = neighborhood_counts >= self.min_samples
        self.core_sample_indices_ = torch.where(core_samples_mask)[0]

        if self._metric_name != "precomputed":
            self.components_ = X[self.core_sample_indices_]
        else:
            self.components_ = None

            # BFS/Loop-based expansion replaced with vectorized label propagation
        # 1. Connected components of foundational points
        # Only foundational points can propagate labels to each other_decomposition
        core_adj = neighbors_mask[core_samples_mask][:, core_samples_mask]
        n_core = self.core_sample_indices_.size(0)

        if n_core > 0:
            # Initialize foundational labels with their own indices (Long)
            core_labels = torch.arange(n_core, device=self.device, dtype=torch.long)

            # Iterative Label Propagation (Parallel Connected Components)
            # This is a vectorized implementation of the Shiloach-Vishkin inspired propagation
            for _ in range(n_core):
                prev_labels = core_labels.clone()
                # Each node takes the minimum label of its neighbors (including itself)
                # Use a large constant (n_core + 1) for non-neighbors to ignore them in min
                mask_adj = torch.where(core_adj, core_labels.unsqueeze(0), torch.tensor(n_core + 1, device=self.device))
                core_labels = torch.min(mask_adj, dim=1).values
                if torch.equal(prev_labels, core_labels):
                    break

            # Finalize foundational labels (contiguous mapping)
            unique_ids, inv_indices = torch.unique(core_labels, return_inverse=True)
            mapped_core_labels = inv_indices

            self.labels_[self.core_sample_indices_] = mapped_core_labels

            # 2. Assign non-foundational points (border points) to the cluster of their foundational neighbor
            non_core_mask = ~core_samples_mask & (neighborhood_counts > 0)
            if non_core_mask.any():
                border_core_adj = neighbors_mask[non_core_mask][:, core_samples_mask]
                has_core_neighbor = border_core_adj.any(dim=1)
                if has_core_neighbor.any():
                    # Find first foundational neighbor (vectorized argmax)
                    first_core_neighbor_idx = torch.argmax(border_core_adj.to(torch.uint8), dim=1)
                    assigned_labels = mapped_core_labels[first_core_neighbor_idx]

                    targeted_non_core_indices = torch.where(non_core_mask)[0][has_core_neighbor]
                    self.labels_[targeted_non_core_indices] = assigned_labels[has_core_neighbor]

        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        n_samples = X.size(0)
        labels = torch.full((n_samples,), -1, device=self.device, dtype=torch.long)
        dists = self.metric(X, self.components_)
        min_dist, min_idx = torch.min(dists, dim=1)
        in_range_mask = min_dist <= self.eps
        core_labels = self.labels_[self.core_sample_indices_]
        labels[in_range_mask] = core_labels[min_idx[in_range_mask]]
        return labels

    def fit_predict(self, X: torch.Tensor, **kwargs):
        self.fit(X, **kwargs)
        return self.labels_

    def transform(self, X: torch.Tensor, **kwargs):
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        unique_labels = torch.unique(self.labels_)
        unique_labels = unique_labels[unique_labels >= 0]
        n_clusters = unique_labels.size(0)
        n_samples = X.size(0)

        if n_clusters == 0:
            return torch.zeros((n_samples, 0), device=self.device, dtype=self.dtype)

        # 1. Compute distances to ALL foundational points
        all_dists = self.metric(X, self.components_)  # (n_samples, n_core)

        # 2. Get cluster labels of those foundational points
        core_labels = self.labels_[self.core_sample_indices_]

        # 3. Vectorized minimization across cluster groups
        # Map labels to [0, n_clusters-1]
        unique_labels, cluster_indices = torch.unique(core_labels, return_inverse=True)

        distance_matrix = torch.full((n_samples, n_clusters), float('inf'),
                                     device=self.device, dtype=self.dtype)

        # Expand cluster_indices to match dists shape (batch_size, n_core)
        indices_expanded = cluster_indices.unsqueeze(0).expand(n_samples, -1)

        # Vectorized scatter_reduce for parallel min finding across labels
        distance_matrix.scatter_reduce_(1, indices_expanded, all_dists, reduce='amin', include_self=False)
        return distance_matrix

    def fit_transform(self, X: torch.Tensor, **kwargs):
        self.fit(X, **kwargs)
        return self.transform(X, **kwargs)


class HDBSCAN(DBSCAN):
    def __init__(self,
                 min_cluster_size: int = 5,
                 min_samples: int = None,
                 eps: float = 0.0,  # Not used in HDBSCAN but kept for compatibility
                 metric: Union[str, Callable, nn.Module] = "euclidean",
                 metric_params: dict = None,
                 cluster_selection_epsilon: float = 0.0,
                 max_cluster_size: int = None,
                 alpha: float = 0.0,
                 cluster_selection_method: Union[str, Callable, nn.Module] = "eom",
                 allow_single_cluster: bool = False,
                 store_centers: str = None,
                 copy: bool = True,
                 algorithm: Union[str, Callable, nn.Module] = "auto",
                 leaf_size: int = 30,
                 p: float = None,
                 n_jobs: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,

                 *args, **kwargs):
        if min_samples is None:
            min_samples = min_cluster_size
        super().__init__(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.min_cluster_size = min_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.max_cluster_size = max_cluster_size
        self.alpha = alpha
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.store_centers = store_centers
        self.copy = copy
        self.probabilities_ = None
        self.centroids_ = None
        self.medoids_ = None
        self.kwargs = kwargs

    def _init_module(self, X: torch.Tensor):
        super()._init_module(X)
        selection_map = {
            "eom": self._cluster_selection_eom,
            "leaf": self._cluster_selection_leaf
        }
        if isinstance(self.cluster_selection_method, str):
            method_name = self.cluster_selection_method.lower()
            if method_name in selection_map:
                self.cluster_selection_method = selection_map[method_name]
            else:
                self.cluster_selection_method = selection_map["eom"]
        elif isinstance(self.cluster_selection_method, nn.Module):
            kwargs = self.kwargs.get("cluster_selection_kwargs", {})
            self.cluster_selection_method = self.cluster_selection_method(**kwargs)

    def _cluster_selection_eom(self, condensed_tree: Dict[str, Any]) -> torch.Tensor:
        """
        Excess of Mass (EOM) cluster selection.
        """
        stabilities = condensed_tree['stabilities']
        node_list = sorted(condensed_tree['nodes'], reverse=True)
        max_cluster_size = self.max_cluster_size

        selected_clusters = set()
        cluster_stabilities = {node: stabilities[node] for node in node_list}

        # Track node sizes for max_cluster_size constraint
        node_sizes = condensed_tree['node_sizes']

        for node in node_list:
            child_nodes = condensed_tree['children'].get(node, [])
            if child_nodes:
                child_stability = sum(cluster_stabilities.get(child, 0.0) for child in child_nodes)

                if stabilities[node] < child_stability:
                    cluster_stabilities[node] = child_stability
                else:
                    # Parent is more stable; check max_cluster_size
                    if max_cluster_size is None or node_sizes[node] <= max_cluster_size:
                        selected_clusters.add(node)
                        for child in child_nodes:
                            if child in selected_clusters:
                                selected_clusters.remove(child)
                    else:
                        # Parent too big, stick with children
                        cluster_stabilities[node] = child_stability

        return torch.tensor(list(selected_clusters), device=self.device)

    def _cluster_selection_leaf(self, condensed_tree: Dict[str, Any]) -> torch.Tensor:
        """
        Leaf cluster selection.

        Simply selects all nodes in the condensed tree that do not have
        any further splits (children).
        """
        all_nodes = set(condensed_tree['nodes'])
        parents = set(condensed_tree['children'].keys())

        # Leaves are nodes that are not parents
        leaves = all_nodes - parents
        return torch.tensor(list(leaves), device=self.device)

    def fit(self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs):
        """
        Fit the HDBSCAN model from features.
        """
        # Ensure data is on the correct device and initialize algorithm/selection methods
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if self.copy:
            X = X.clone()

        # Performance Optimization: For small datasets, CUDA overhead (launch/sync)
        # is much larger than the computation. CPU is significantly faster for N < 100.
        original_device = self.device
        if X.size(0) < 100 and self.device == "cuda":
            X = X.to("cpu")
            self.device = "cpu"

        self._init_module(X)  # Initializes self.algorithm and self.cluster_selection_method
        n_samples = X.size(0)

        # 1. Compute Distances and Core Distances
        # core_dist(p) is the distance to the k-th nearest neighbor (k = min_samples)
        if isinstance(self.algorithm, (BallTree, KDTree)):
            # Use tree-based query from knn.py
            dists, _ = self.algorithm.query(X, k=self.min_samples, return_distance=True)
            core_distances = dists[:, -1]
            # Mutual reachability needs dist(a,b), so we still need the full matrix or a sparse one
            # For simplicity and to match original behavior (which computed it later), we compute it here if not available
            dist_matrix = self.metric(X, X)
        else:
            # Re-use dist_matrix for both core distances and mutual reachability
            if self.device == "cpu" and n_samples > 500 and self.n_jobs is not None and self.n_jobs != 1:
                # Use Parallel only for larger datasets to avoid overhead
                # CPU Parallelization with joblib for larger datasets
                def compute_row(i):
                    return self.metric(X[i:i + 1], X)

                dist_matrix = torch.cat(Parallel(n_jobs=self.n_jobs)(delayed(compute_row)(i) for i in range(n_samples)))
            elif self.device != "cpu" and n_samples > 100:
                # GPU/Vmap acceleration (only for larger samples to avoid overhead)
                try:
                    def dist_row(xi):
                        return self.metric(xi.unsqueeze(0), X)

                    dist_matrix = vmap(dist_row)(X).squeeze(1)
                except Exception:
                    dist_matrix = self.metric(X, X)
            else:
                dist_matrix = self.metric(X, X)

            core_distances, _ = torch.topk(dist_matrix, k=self.min_samples, largest=False, dim=-1)
            core_distances = core_distances[:, -1]

        # Apply alpha scaling if provided
        actual_dist = dist_matrix if self.alpha <= 0 else dist_matrix / self.alpha

        mr_matrix = torch.max(
            core_distances.view(-1, 1),
            torch.max(core_distances.view(1, -1), actual_dist)
        )

        # 3. Minimum Spanning Tree (MST)
        mst_edges = self._compute_mst(mr_matrix)

        # 4. Construct Single Linkage Tree
        self.single_linkage_tree_ = self._compute_linkage(mst_edges, n_samples)

        # 5. Condense the Tree
        condensed_tree = self._condense_tree(self.single_linkage_tree_, n_samples)

        # 6. Cluster Selection
        selected_clusters = self.cluster_selection_method(condensed_tree)

        # 7. Extract Labels and Probabilities
        self.labels_, self.probabilities_ = self._extract_labels(
            condensed_tree, selected_clusters, n_samples
        )

        # Set foundational points (all non-noise points for HDBSCAN)
        mask = self.labels_ >= 0
        self.core_sample_indices_ = torch.where(mask)[0]
        self.components_ = X[mask]

        # 8. Compute Centers if requested
        if self.store_centers in ["centroid", "medoid", "both"]:
            self._compute_centers(X)

        if original_device == "cuda":
            self.labels_ = self.labels_.to(original_device)
            self.probabilities_ = self.probabilities_.to(original_device)
            self.device = original_device

        return self

    def _compute_mst(self, mr_matrix: torch.Tensor) -> torch.Tensor:
        """Computes MST using Kruskal's algorithm (vectorized edge extraction and sort)."""
        n = mr_matrix.size(0)
        # Vectorized edge extraction: upper triangle only
        i, j = torch.triu_indices(n, n, 1, device=self.device)
        weights = mr_matrix[i, j]
        edges = torch.stack([i.float(), j.float(), weights], dim=1)
        # Sort by weight (fully vectorized)
        sort_idx = torch.argsort(edges[:, 2])
        edges = edges[sort_idx]
        # Union-Find to select MST edges (sequential by nature, minimal Python overhead)
        parent = torch.arange(n, device=self.device, dtype=torch.long)

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            # Path compression
            while parent[x] != root:
                next_p = parent[x]
                parent[x] = root
                x = next_p
            return root

        mst_edges = []
        for k in range(edges.size(0)):
            u, v = int(edges[k, 0].item()), int(edges[k, 1].item())
            w = edges[k, 2].item()
            ru, rv = find(u), find(v)
            if ru != rv:
                mst_edges.append([u, v, w])
                parent[ru] = rv
            if len(mst_edges) >= n - 1:
                break
        return torch.tensor(mst_edges, device=self.device)

    def _compute_linkage(self, mst: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Converts MST edges into a single linkage tree (dendrogram)."""
        # Sort edges by weight
        sorted_indices = torch.argsort(mst[:, 2])
        mst = mst[sorted_indices]

        # Union-Find parent array (using list for faster scalar recursion/iteration)
        parent = list(range(2 * n_samples))
        size = torch.cat([torch.ones(n_samples), torch.zeros(n_samples)]).tolist()

        def find(i):
            root = i
            while parent[root] != root:
                root = parent[root]
            # Path compression
            while parent[i] != root:
                next_p = parent[i]
                parent[i] = root
                i = next_p
            return root

        linkage = []
        next_cluster = n_samples
        for i in range(mst.size(0)):
            u, v, weight = int(mst[i, 0]), int(mst[i, 1]), mst[i, 2].item()
            root_u, root_v = find(u), find(v)

            if root_u != root_v:
                # Merge clusters into a new node
                new_size = size[root_u] + size[root_v]
                linkage.append([float(root_u), float(root_v), weight, float(new_size)])

                parent[root_u] = next_cluster
                parent[root_v] = next_cluster
                size[next_cluster] = new_size
                next_cluster += 1

        return torch.tensor(linkage, device=self.device, dtype=torch.float32)

    def _condense_tree(self, linkage: torch.Tensor, n_samples: int) -> Dict[str, Any]:
        """
        Condenses the single linkage tree based on min_cluster_size and cluster_selection_epsilon.
        Optimized to avoid O(N^2) leaf discovery.
        """
        self._tree_linkage = linkage
        min_cluster_size = self.min_cluster_size
        cluster_selection_epsilon = self.cluster_selection_epsilon

        n_links = linkage.size(0)
        # Tracking lambda values for each point
        lambdas = torch.zeros(n_samples, device=self.device)

        # node_remap[orig_id] -> current_condensed_id
        node_remap = torch.arange(n_samples + n_links, device=self.device)
        # node_size for the hierarchy
        node_size = torch.cat([torch.ones(n_samples, device=self.device), linkage[:, 3]])

        condensed_edges = {}
        node_sizes = {}
        birth_lambdas = {}

        # Bottom-up traversal to shuck points
        for i in range(n_links):
            parent = n_samples + i
            left, right, dist, total_size = linkage[i]
            lambd = 1.0 / dist if dist > 0 else 1e10

            if dist < cluster_selection_epsilon:
                node_remap[parent] = -1
                continue

            size_l = node_size[int(left)]
            size_r = node_size[int(right)]

            if size_l >= min_cluster_size and size_r >= min_cluster_size:
                # True Split
                child_l = int(node_remap[int(left)])
                child_r = int(node_remap[int(right)])

                if child_l != -1 and child_r != -1:
                    condensed_edges[parent] = [child_l, child_r]
                    node_sizes[parent] = total_size.item()
                    birth_lambdas[child_l] = lambd
                    birth_lambdas[child_r] = lambd
                    node_remap[parent] = parent
                else:
                    node_remap[parent] = -1
            elif size_l < min_cluster_size and size_r < min_cluster_size:
                # Both shucked
                leaves = self._get_leaves_under_node(parent, n_samples)
                lambdas[leaves] = lambd
                node_remap[parent] = -1
            elif size_l < min_cluster_size:
                # Left shucked
                leaves = self._get_leaves_under_node(int(left), n_samples)
                lambdas[leaves] = lambd
                node_remap[parent] = node_remap[int(right)]
            else:
                # Right shucked
                leaves = self._get_leaves_under_node(int(right), n_samples)
                lambdas[leaves] = lambd
                node_remap[parent] = node_remap[int(left)]

        # Collect all nodes that are valid condensed clusters
        # A cluster is valid if it's a target of node_remap and >= n_samples
        all_condensed_nodes = set()
        for i in range(n_samples, n_samples + n_links):
            remapped = int(node_remap[i])
            if remapped >= n_samples:
                all_condensed_nodes.add(remapped)

        stabilities = {node: 0.0 for node in all_condensed_nodes}

        # Pre-accumulate lambda sums and point counts
        # leaf_sum[node] = sum of lambdas of all points under this node
        # leaf_count[node] = count of points under this node
        leaf_sums = torch.zeros(n_samples + n_links, device=self.device)
        leaf_counts = torch.zeros(n_samples + n_links, device=self.device)

        # Initial values for leaves
        leaf_sums[:n_samples] = lambdas
        leaf_counts[:n_samples] = 1.0

        # Bottom-up pass for all original nodes (MST linkage)
        for i in range(n_links):
            parent = n_samples + i
            left, right = int(linkage[i, 0]), int(linkage[i, 1])
            leaf_sums[parent] = leaf_sums[left] + leaf_sums[right]
            leaf_counts[parent] = leaf_counts[left] + leaf_counts[right]

        # Calculate stability using pre-accumulated values
        for node in condensed_edges.keys():
            b_lambda = birth_lambdas.get(node, 0.0)
            # Stability = sum(lambda_p) - count(p) * lambda_birth
            stabilities[node] = (leaf_sums[node] - leaf_counts[node] * b_lambda).item()

        return {
            'nodes': list(all_condensed_nodes),
            'children': condensed_edges,
            'stabilities': stabilities,
            'lambdas': lambdas,
            'node_sizes': node_sizes,
            'linkage': linkage
        }

    def _get_leaves_under_node(self, node_idx: int, n_samples: int) -> torch.Tensor:
        """
        Iteratively finds all leaf indices (0 to n_samples-1) under a given tree node.
        """
        leaves = []
        stack = [node_idx]
        while stack:
            curr = stack.pop()
            if curr >= n_samples:
                row = self._tree_linkage[curr - n_samples]
                stack.append(int(row[1]))
                stack.append(int(row[0]))
            else:
                leaves.append(curr)
        return torch.tensor(leaves, device=self.device, dtype=torch.long)

    def _extract_labels(self, condensed_tree: Dict[str, Any],
                        selected_clusters: torch.Tensor,
                        n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts final labels and persistence-based probabilities.
        Fully vectorized where possible.
        """
        labels = torch.full((n_samples,), -1, device=self.device, dtype=torch.long)
        probs = torch.zeros(n_samples, device=self.device)

        if selected_clusters.size(0) == 0:
            return labels, probs

        # 1. Map each point to its selected cluster
        # Pre-calculate all memberships in a single pass using the original linkage
        # memberships[node] = selected_cluster_id or -1
        linkage = condensed_tree['linkage']
        selected_map = {node.item(): i for i, node in enumerate(selected_clusters)}

        node_memberships = torch.full((n_samples + linkage.size(0),), -1,
                                      dtype=torch.long, device=self.device)

        # Mark selected nodes
        for node, i in selected_map.items():
            node_memberships[node] = i

        # Top-down pass to propagate memberships to descendants
        # But linkage is bottom-up. Standard HDBSCAN uses top-down from roots.
        # Alternatively, we can use the condensed_tree structure which is already filtered.

        linkage_cpu = linkage.cpu()  # Small metadata is faster to iterate if needed, but we use torch
        n_nodes = n_samples + linkage.size(0)

        # Traverse top-down (reverse linkage)
        for i in range(linkage.size(0) - 1, -1, -1):
            parent = n_samples + i
            m = node_memberships[parent]
            if m != -1:
                left, right = int(linkage[i, 0]), int(linkage[i, 1])
                # Only overwrite if children aren't their own selected clusters
                if node_memberships[left] == -1: node_memberships[left] = m
                if node_memberships[right] == -1: node_memberships[right] = m

        labels = node_memberships[:n_samples]

        # 2. Persistence-based probabilities (vectorized via scatter_reduce)
        # Normalized lambda: (lambda_p - lambda_birth) / (lambda_max - lambda_birth)
        lambdas = condensed_tree['lambdas']
        n_clusters = selected_clusters.size(0)
        valid_mask = labels >= 0
        probs = torch.zeros(n_samples, device=self.device)

        if valid_mask.any():
            valid_labels_idx = labels[valid_mask].long()
            valid_lambdas = lambdas[valid_mask]
            l_min = torch.full((n_clusters,), float('inf'), device=self.device, dtype=lambdas.dtype)
            l_max = torch.full((n_clusters,), float('-inf'), device=self.device, dtype=lambdas.dtype)
            l_min.scatter_reduce_(0, valid_labels_idx, valid_lambdas, reduce='amin', include_self=False)
            l_max.scatter_reduce_(0, valid_labels_idx, valid_lambdas, reduce='amax', include_self=False)
            l_min_per_point = l_min[labels[valid_mask]]
            l_max_per_point = l_max[labels[valid_mask]]
            span = l_max_per_point - l_min_per_point
            probs[valid_mask] = torch.where(
                span > 1e-10,
                (valid_lambdas - l_min_per_point) / span,
                torch.ones_like(valid_lambdas)
            )

        return labels, probs

    def _handle_invalid_data(self, X: torch.Tensor):
        """
        Identifies infinite or NaN values to assign labels -2 and -3.
        """
        nan_mask = torch.isnan(X).any(dim=1)
        inf_mask = torch.isinf(X).any(dim=1) & ~nan_mask

        return nan_mask, inf_mask

    def _apply_outlier_labels(self, nan_mask, inf_mask):
        """
        Applies specific outlier codes: -2 for Infinite, -3 for Missing.
        """
        if nan_mask.any():
            self.labels_[nan_mask] = -3
            self.probabilities_[nan_mask] = float('nan')

        if inf_mask.any():
            self.labels_[inf_mask] = -2
            self.probabilities_[inf_mask] = 0.0

    def _compute_centers(self, X: torch.Tensor):
        """
        Populates self.centroids_ and self.medoids_ for valid clusters.
        """
        unique_labels = torch.unique(self.labels_)
        valid_labels = unique_labels[unique_labels >= 0]
        n_clusters = len(valid_labels)

        if n_clusters == 0:
            return

        if self.store_centers in ["centroid", "both"]:
            # Vectorized centroid calculation (O(N) instead of O(K*N))
            valid_mask = self.labels_ >= 0
            if valid_mask.any():
                valid_labels_raw = self.labels_[valid_mask]
                unique_labels, mapped_labels = torch.unique(valid_labels_raw, return_inverse=True)
                n_valid = unique_labels.size(0)

                sums = torch.zeros((n_valid, X.size(-1)), device=self.device, dtype=self.dtype)
                counts = torch.zeros(n_valid, device=self.device, dtype=self.dtype)

                sums.index_add_(0, mapped_labels, X[valid_mask])
                counts.index_add_(0, mapped_labels, torch.ones_like(valid_labels_raw, dtype=self.dtype))
                self.centroids_ = sums / counts.unsqueeze(-1)
            else:
                self.centroids_ = None

        if self.store_centers in ["medoid", "both"]:
            # Vectorized: single metric call over all valid points, then index per cluster
            valid_mask = self.labels_ >= 0
            if valid_mask.any():
                all_valid = X[valid_mask]
                labels_valid = self.labels_[valid_mask]
                d_full = self.metric(all_valid, all_valid)
                medoids = []
                for label in valid_labels:
                    idx = (labels_valid == label).nonzero(as_tuple=True)[0]
                    d_block = d_full[idx][:, idx]
                    medoid_local = torch.argmin(d_block.sum(dim=1))
                    medoids.append(all_valid[idx[medoid_local]])
                if medoids:
                    self.medoids_ = torch.stack(medoids)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict the cluster labels for new data.
        Assigns the label of the closest foundational point in the persistent clusters.
        """
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)

        # We use the medoids or centroids as representatives if they exist
        if self.medoids_ is not None:
            representatives = self.medoids_
        elif self.centroids_ is not None:
            representatives = self.centroids_
        else:
            # Fallback to all non-noise points
            mask = self.labels_ >= 0
            representatives = self.components_  # Inherited from DBSCAN foundational points

        dists = self.metric(X, representatives)
        min_dist, min_idx = torch.min(dists, dim=1)

        # For prediction, we use the cluster_selection_epsilon as a threshold
        labels = torch.full((X.size(0),), -1, device=self.device, dtype=torch.long)

        # Map back to original cluster IDs
        unique_labels = torch.unique(self.labels_)
        valid_labels = unique_labels[unique_labels >= 0]

        labels = valid_labels[min_idx]
        return labels

    def fit_predict(self, X: torch.Tensor, **kwargs):
        """Fit the model and return the labels."""
        self.fit(X, **kwargs)
        return self.labels_

    def transform(self, X: torch.Tensor, **kwargs):
        """
        Transform X to a cluster-distance space.

        In this space, the value of each feature is the distance to the cluster
        representative (centroid or medoid). If both are stored, medoids are
        prioritized as they are guaranteed to be observed data points.

        Parameters
        ----------
        X : torch.Tensor
            New data to transform.

        Returns
        -------
        X_new : torch.Tensor of shape (n_samples, n_clusters)
            Distances to each cluster representative.
        """
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)

        # Determine which cluster centers to use for distance calculation
        if self.medoids_ is not None:
            centers = self.medoids_
        elif self.centroids_ is not None:
            centers = self.centroids_
        else:
            raise RuntimeError(
                "No cluster centers stored. Set 'store_centers' to 'centroid', "
                "'medoid', or 'both' during initialization to use transform."
            )

        # Calculate distances from each point in X to each cluster center
        # Using the metric defined during initialization
        return self.metric(X, centers)

    def fit_transform(self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Fit the model to X and return the cluster-distance space.

        Parameters
        ----------
        X : torch.Tensor
            Training data.
        y : Ignored
            Not used, present here for API consistency.

        Returns
        -------
        X_new : torch.Tensor of shape (n_samples, n_clusters)
            Distances to each cluster representative.
        """
        return self.fit(X, **kwargs).transform(X)


class OPTICS(DBSCAN):
    def __init__(self,
                 min_samples: int = None,
                 max_eps: float = float('inf'),
                 eps: float = 0.0,
                 metric: Union[str, Callable, nn.Module] = "euclidean",
                 metric_params: dict = None,
                 cluster_method: Union[str, Callable, nn.Module] = "xi",
                 xi: float = 0.05,
                 predecessor_correction: bool = True,
                 min_cluster_size: Union[int, float] = None,
                 algorithm: Union[str, Callable, nn.Module] = "auto",
                 leaf_size: int = 30,
                 memory: Union[str] = None,
                 p: float = None,
                 n_jobs: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,

                 *args, **kwargs):
        if min_samples is None:
            min_samples = 5
        super().__init__(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.max_eps = max_eps
        self.cluster_method = cluster_method.lower()
        self.xi = max(0.0, min(1.0, abs(xi)))
        self.predecessor_correlation = predecessor_correction
        if min_cluster_size is None:
            self.min_cluster_size = xi
        if isinstance(min_cluster_size, float):
            self.min_cluster_size = max(0.0, min(1.0, abs(min_cluster_size)))
        elif isinstance(min_cluster_size, int):
            if abs(min_cluster_size) < 1:
                self.min_cluster_size = xi
        else:
            self.min_cluster_size = xi
        self.memory = memory
        self.reachability_ = None
        self.ordering_ = None
        self.core_distances_ = None
        self.predecessor_ = None
        self.cluster_hierarchy_ = None
        self.kwargs = kwargs

    def _init_module(self, X: torch.Tensor):
        super()._init_module(X)
        if isinstance(self.cluster_method, str):
            if self.cluster_method == "xi":
                self.cluster_method = self._xi_method
            elif self.cluster_method == "dbscan":
                self.cluster_method = self._dbscan_method
            else:
                self.cluster_method = self._xi_method
        elif isinstance(self.cluster_method, nn.Module):
            self.cluster_method = self.cluster_method(**self.kwargs.get("cluster_method_kwargs", {}))
        return self

    def _dbscan_method(self) -> torch.Tensor:
        """
        Extract clusters based on a fixed reachability threshold (eps).
        Equivalent to a DBSCAN 'cut' of the OPTICS reachability plot.
        Vectorized approach using parallel label propagation concepts.
        """
        n_samples = self.reachability_.size(0)
        labels = torch.full((n_samples,), -1, device=self.device, dtype=torch.long)
        eps = self.eps if self.eps > 0 else self.max_eps

        if eps == float('inf'):
            return labels

        # In cluster order, identify where reachability > eps (new cluster potential)
        reach_ordered = self.reachability_[self.ordering_]
        core_ordered = self.core_distances_[self.ordering_]

        # A point starts a new cluster if its reachability > eps AND it is a foundational point
        # Subsequent points stay in the cluster if their reachability <= eps
        is_split = (reach_ordered > eps)
        is_core = (core_ordered <= eps)

        # New cluster starts where it's a split point but also a foundational point
        new_cluster_starts = is_split & is_core
        cluster_ids = torch.cumsum(new_cluster_starts.long(), dim=0) - 1

        # Points with reachability > eps and NOT foundational are noise
        noise_mask = is_split & (~is_core)

        # Temporary labels in cluster order
        ordered_labels = cluster_ids.clone()

        # Propagation: If a point is not a split, it inherits the previous cluster ID
        # (This is already handled by cumsum for non-split points)

        # Apply noise mask
        ordered_labels[noise_mask] = -1

        # Handle the very first point if it's not a foundational point (it's always noise or assigned above)
        # If no foundational points are found yet, labels will be -1

        # Map back to original indices
        labels[self.ordering_] = ordered_labels
        return labels

    def _xi_method(self, ) -> torch.Tensor:
        """
        Automatically extract clusters according to the Xi-steep method.
        Optimized steepness detection and vectorized labeling.
        """
        reach_ordered = self.reachability_[self.ordering_]
        n_samples = reach_ordered.size(0)

        # Vectorized steepness detection
        # reach[i] <= reach[i+1] * (1-xi) -> steep up
        # reach[i] * (1-xi) >= reach[i+1] -> steep down
        steep_up = torch.zeros(n_samples, dtype=torch.bool, device=self.device)
        steep_down = torch.zeros(n_samples, dtype=torch.bool, device=self.device)

        steep_up[:-1] = (reach_ordered[:-1] <= reach_ordered[1:] * (1 - self.xi))
        steep_down[1:] = (reach_ordered[:-1] * (1 - self.xi) >= reach_ordered[1:])

        # Hierarchical cluster finding (Valleys)
        # This part is naturally stack-based/sequential but we minimize the work
        clusters = []
        down_stack = []

        # We process the ordering to find [down, up] pairs
        for i in range(n_samples):
            if steep_down[i]:
                down_stack.append(i)
            elif steep_up[i]:
                while down_stack:
                    start = down_stack.pop()
                    # Check min_cluster_size and xi-specific constraints (delta reachability)
                    if (i - start + 1) >= self.min_cluster_size:
                        # Find the peak between start and i
                        peak = reach_ordered[start:i + 1].max()
                        if reach_ordered[start] >= peak * (1 - self.xi) or reach_ordered[i] >= peak * (1 - self.xi):
                            clusters.append([start, i])
                            # In xi-method, once an up-steep matches a down-stack,
                            # we can potentially break or continue based on hierarchy.
                            # Standard OPTICS allows overlapping/nested clusters.
                # Reset stack for the next potential valley unless we find another down

        if not clusters:
            return torch.full((n_samples,), -1, device=self.device, dtype=torch.long)

        # Sort clusters by end index (ascending) and negative start (descending)
        clusters.sort(key=lambda x: (x[1], -x[0]))
        self.cluster_hierarchy_ = torch.tensor(clusters, device=self.device)

        # Batch labeling: collect flat_indices and flat_labels, then single index_put
        self.labels_ = torch.full((n_samples,), -1, device=self.device, dtype=torch.long)
        flat_indices_list = []
        flat_labels_list = []

        for cluster_id, (start, end) in enumerate(clusters):
            indices = self.ordering_[start:end + 1]
            if self.predecessor_correlation:
                pred_ids = self.predecessor_[indices]
                is_member = torch.isin(pred_ids, indices)
                is_member[0] = True
                valid_mask = torch.cumprod(is_member.long(), dim=0).bool()
                final_indices = indices[valid_mask]
            else:
                final_indices = indices
            flat_indices_list.append(final_indices)
            flat_labels_list.append(torch.full((final_indices.numel(),), cluster_id, device=self.device, dtype=torch.long))

        if flat_indices_list:
            flat_indices = torch.cat(flat_indices_list)
            flat_labels = torch.cat(flat_labels_list)
            self.labels_.index_put_((flat_indices,), flat_labels)

        return self.labels_

    def fit(self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs):
        """
        Fit the OPTICS model from features.
        """
        # Ensure data is on the correct device and initialize algorithm
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        n_samples = int(X.size(0))

        # Performance Optimization: For small datasets, CUDA overhead (launch/sync)
        # is much larger than the computation. CPU is significantly faster for N < 100.
        original_device = self.device
        if X.size(0) < 100 and self.device == "cuda":
            X = X.to("cpu")
            self.device = "cpu"

        # Initialize trees/algorithm and extraction methods
        self._init_module(X)

        # 1. Pre-calculate Core Distances and precompute full distance matrix
        # Distance to the k-th nearest neighbor (k = min_samples)
        if isinstance(self.algorithm, (BallTree, KDTree)):
            dists, _ = self.algorithm.query(X, k=self.min_samples, return_distance=True)
            self.core_distances_ = torch.as_tensor(dists[:, -1], device=self.device, dtype=self.dtype)
            # Compute full dist matrix once for _expand_cluster_order
            self._dist_matrix_ = self.metric(X, X)
        else:
            # Brute force: single vectorized dist_matrix for both core_distances and expansion
            dist_matrix = self.metric(X, X)
            core_dists, _ = torch.topk(dist_matrix, k=self.min_samples, largest=False, dim=-1)
            self.core_distances_ = core_dists[:, -1]
            self._dist_matrix_ = dist_matrix

        # Points with foundational distance > max_eps are not considered foundational points
        self.core_distances_[self.core_distances_ > self.max_eps] = float('inf')

        # 3. Initialize OPTICS metadata
        self.reachability_ = torch.full((n_samples,), float('inf'), device=self.device)
        self.predecessor_ = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)
        self.ordering_ = []
        processed = torch.zeros(n_samples, dtype=torch.bool, device=self.device)

        # 4. Main Loop: Generate Cluster Ordering
        for i in range(n_samples):
            if not processed[i]:
                # Expand from a new unvisited point
                self._expand_cluster_order(i, X, processed)

        self.ordering_ = torch.tensor(self.ordering_, device=self.device, dtype=torch.long)

        # 4. Extract Labels using the method set in __init__ (xi or dbscan)
        # self.cluster_method is now a bound method (_xi_method or _dbscan_method)
        self.labels_ = self.cluster_method()

        if original_device == "cuda":
            self.labels_ = self.labels_.to(original_device)
            self.device = original_device

        return self

    def _expand_cluster_order(self, root_idx: int, X: torch.Tensor, processed: torch.Tensor):
        """
        Expands the cluster order starting from root_idx.
        Optimized to use global reachability updates.
        """
        import heapq
        # Use a priority queue (min-heap) to manage seeds: (reachability, point_index)
        seeds = [(self.reachability_[root_idx].item(), root_idx)]

        while seeds:
            # Pick the point with the lowest reachability in the seed set
            reach_u, u = heapq.heappop(seeds)

            if processed[u]:
                continue

            processed[u] = True
            self.ordering_.append(u)

            # If u is a foundational point, update reachabilities of its neighbors
            core_dist_u = self.core_distances_[u]
            if not torch.isinf(core_dist_u):
                unprocessed_mask = ~processed
                unprocessed_indices = torch.where(unprocessed_mask)[0]

                if unprocessed_indices.numel() > 0:
                    # Use precomputed distance matrix (vectorized)
                    dist_to_neighbors = self._dist_matrix_[u, unprocessed_indices].view(-1)

                    # Mutual Reachability: max(core_dist(u), dist(u, neighbors))
                    new_reach = torch.clamp(dist_to_neighbors, min=core_dist_u)

                    # Update if within max_eps and shorter than current reachability
                    valid_mask = (dist_to_neighbors <= self.max_eps) & (
                                new_reach < self.reachability_[unprocessed_indices])

                    if valid_mask.any():
                        target_indices = unprocessed_indices[valid_mask]
                        new_reach_vals = new_reach[valid_mask]
                        
                        self.reachability_[target_indices] = new_reach_vals
                        self.predecessor_[target_indices] = u

                        # Push updated points into the priority queue
                        for idx_val, r_val in zip(target_indices.tolist(), new_reach_vals.tolist()):
                            heapq.heappush(seeds, (r_val, idx_val))

    def _update_reachability(self, core_idx, X, processed):
        """
        Calculates reachability: max(core_dist(core_idx), dist(core_idx, neighbors))
        Only updates for unprocessed points within max_eps.
        """
        unprocessed_mask = ~processed
        unprocessed_indices = torch.where(unprocessed_mask)[0]

        if unprocessed_indices.numel() == 0:
            return

        # Compute distances from the current foundational point to all unprocessed points
        dist_to_neighbors = self.metric(X[core_idx:core_idx + 1], X[unprocessed_mask]).squeeze(0)

        # Filter by max_eps
        valid_mask = dist_to_neighbors <= self.max_eps
        if not valid_mask.any():
            return

        target_indices = unprocessed_indices[valid_mask]
        dist_to_neighbors = dist_to_neighbors[valid_mask]

        # Reachability(p, o) = max(core_dist(o), dist(o, p))
        new_reach = torch.max(self.core_distances_[core_idx], dist_to_neighbors)

        # Update if the new reachability is smaller than the current one
        current_reach = self.reachability_[target_indices]
        update_mask = new_reach < current_reach

        if update_mask.any():
            final_targets = target_indices[update_mask]
            self.reachability_[final_targets] = new_reach[update_mask]
            self.predecessor_[final_targets] = core_idx

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict the cluster labels for new data.
        Vectorized nearest-neighbor matching.
        """
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if self.labels_ is None or not (self.labels_ >= 0).any():
            return torch.full((X.size(0),), -1, device=self.device, dtype=torch.long)

        # OPTICS representatives: points that belong to any detected cluster
        # In OPTICS, we often check against all fit points, but here we prioritize cluster members.
        fit_X = self.components_ if self.components_ is not None else X
        fit_labels = self.labels_[self.labels_ >= 0] if self.components_ is not None else self.labels_

        # Batch Distance Calculation
        dists = self.metric(X, fit_X)
        min_dist, min_idx = torch.min(dists, dim=1)

        labels = torch.full((X.size(0),), -1, device=self.device, dtype=torch.long)

        # Max Eps constraint for assignment
        valid_mask = (min_dist <= self.max_eps)
        labels[valid_mask] = fit_labels[min_idx[valid_mask]]

        return labels

    def transform(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Transform X to a cluster-distance space.
        Vectorized using scatter_reduce for minimum distance per cluster.
        """
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        unique_labels = torch.unique(self.labels_)
        unique_labels = unique_labels[unique_labels >= 0]
        n_clusters = unique_labels.size(0)
        n_samples = X.size(0)

        if n_clusters == 0:
            return torch.zeros((n_samples, 0), device=self.device, dtype=self.dtype)

        # Compute distances to all points that are in a cluster
        # Using the same logic as DBSCAN transform for efficiency
        cluster_mask = self.labels_ >= 0
        if not cluster_mask.any():
            return torch.zeros((n_samples, 0), device=self.device, dtype=self.dtype)

        clustered_X = X[cluster_mask]  # Use X since fit stores no components_ by default unless handled
        labels_of_clustered = self.labels_[cluster_mask]

        # Batch distance matrix: (n_samples, n_clustered_points)
        all_dists = self.metric(X, clustered_X)

        # Vectorized label mapping: unique_labels is sorted, use searchsorted
        mapped_labels = torch.searchsorted(unique_labels, labels_of_clustered)

        # Use scatter_reduce to find min distance per cluster for each sample
        # We need to broadcast mapped_labels to match all_dists shape
        # mapped_labels: (n_clustered_points,) -> expanded to (n_samples, n_clustered_points)
        target_indices = mapped_labels.unsqueeze(0).expand(n_samples, -1)

        res = torch.full((n_samples, n_clusters), float('inf'), device=self.device, dtype=self.dtype)
        res = torch.scatter_reduce(res, 1, target_indices, all_dists, reduce='amin', include_self=False)

        return res

    def fit_predict(self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Fit the model and return the cluster labels found during ordering.
        """
        self.fit(X, y, **kwargs)
        return self.labels_

    def fit_transform(self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Fit the model and transform the data into cluster-distance space.
        """
        return self.fit(X, y, **kwargs).transform(X)
