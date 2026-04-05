import torch
import math
from typing import Union, Any, Tuple, List
from .....models.utils import MLRegressor
import heapq
from torch.func import vmap
import joblib

__all__ = ["DecisionTreeRegressor", "ExtraTreeRegressor"]


class DecisionTreeRegressor(MLRegressor):
    def __init__(self,
                 criterion: str = "squared_error",
                 splitter: str = "best",
                 max_depth: int = None,
                 min_samples_split: Union[int, float] = 2,
                 min_samples_leaf: Union[int, float] = 1,
                 min_weight_fraction_leaf: float = 0.0,
                 max_features: Union[int, float, str] = None,
                 random_state: int = None,
                 max_leaf_nodes: int = None,
                 min_impurity_decrease: float = 0.0,
                 ccp_alpha: float = 0.0,
                 monotonic_cst: Union[List[int], Tuple[int], torch.Tensor] = None,
                 interaction_cst: Union[str, List[Any], Tuple[Any]] = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.interaction_cst = interaction_cst
        self.device = device
        self.dtype = dtype
        self.max_features = max_features
        self.is_fit = False
        self.in_features = None
        self.out_features = None
        self._feature_importances = None
        self._features_out = None
        self.tree_structure = None
        self.estimator_type = 'regressor'

    def state_dict(self, *args, **kwargs):
        """
        Include tree structure and fit metadata in the state dict.
        """
        sd = super().state_dict(*args, **kwargs)
        # Add non-tensor metadata
        sd['_tree_metadata'] = {
            'is_fit': self.is_fit,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'max_features_val': getattr(self, 'max_features_val', None),
            'min_samples_split_abs': getattr(self, 'min_samples_split_abs', None),
            'min_samples_leaf_abs': getattr(self, 'min_samples_leaf_abs', None),
            'min_weight_leaf_abs': getattr(self, 'min_weight_leaf_abs', None),
            'interaction_cst': self.interaction_cst,
            'tree_structure': self.tree_structure
        }
        return sd

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        """
        Restore tree structure and fit metadata from the state dict.
        """
        metadata = state_dict.pop('_tree_metadata', None)
        if metadata:
            self.is_fit = metadata.get('is_fit', False)
            self.in_features = metadata.get('in_features')
            self.out_features = metadata.get('out_features')
            self.max_features_val = metadata.get('max_features_val')
            self.min_samples_split_abs = metadata.get('min_samples_split_abs')
            self.min_samples_leaf_abs = metadata.get('min_samples_leaf_abs')
            self.min_weight_leaf_abs = metadata.get('min_weight_leaf_abs')
            self.interaction_cst = metadata.get('interaction_cst')
            self.tree_structure = metadata.get('tree_structure')

        # Pre-register _fit_* buffers from state_dict so load_state_dict can load into them
        # (a fresh model may not have these buffers yet; they are created by _register_fitted_state)
        for key in list(state_dict.keys()):
            if key.startswith('_fit_') and key not in self._buffers:
                t = state_dict[key]
                if isinstance(t, torch.Tensor):
                    self.register_buffer(key, torch.empty_like(t))
                    if key not in getattr(self, '_dynamic_buffers', set()):
                        try:
                            object.__getattribute__(self, '_dynamic_buffers').add(key)
                        except AttributeError:
                            pass

        # Lazy registration of feature importances if loading into an unfitted model
        if '_feature_importances' in state_dict:
            importances = state_dict['_feature_importances']
            if not hasattr(self, '_feature_importances') or '_feature_importances' not in self._buffers:
                if hasattr(self, '_feature_importances'):
                    del self._feature_importances
                self.register_buffer('_feature_importances', torch.zeros_like(importances))

        return super().load_state_dict(state_dict, strict=strict, *args, **kwargs)

    def _init_module(self, X, y):
        in_features = X.size(-1)
        out_features = y.size(-1) if y.ndim > 1 else 1
        
        if self.max_features is None:
            self.max_features_val = in_features
        elif isinstance(self.max_features, float):
            self.max_features_val = max(1, int(self.max_features * in_features))
        elif isinstance(self.max_features, int):
            self.max_features_val = self.max_features
        elif isinstance(self.max_features, str):
            if self.max_features.lower() == "sqrt":
                self.max_features_val = int(math.sqrt(in_features))
            elif self.max_features.lower() == "log2":
                self.max_features_val = int(math.log2(in_features))
            else:
                self.max_features_val = in_features

        self.in_features = in_features
        self.out_features = out_features
        # Use register_buffer so it's handled by state_dict automatically
        if hasattr(self, '_feature_importances'):
            # If it already exists (e.g. from a previous fit or load), 
            # we just ensure it's a buffer or reset it.
            if '_feature_importances' not in self._buffers:
                if hasattr(self, '_feature_importances'):
                    del self._feature_importances
                self.register_buffer('_feature_importances', torch.zeros(in_features, device=self.device, dtype=self.dtype))
            else:
                self._feature_importances = torch.zeros(in_features, device=self.device, dtype=self.dtype)
        else:
            self.register_buffer('_feature_importances', torch.zeros(in_features, device=self.device, dtype=self.dtype))
        return self

    @property
    def feature_importances_(self):
        if self._feature_importances is None:
            return None
        return self._feature_importances / (self._feature_importances.sum() + 1e-9)

    @property
    def max_features_(self):
        return self.max_features_val

    @property
    def n_features_in_(self):
        return self.in_features

    @property
    def n_outputs_(self):
        return self.out_features

    @property
    def tree_(self):
        return self.tree_structure

    def _parse_interaction_cst(self, n_features):
        if self.interaction_cst is None:
            return None
        if self.interaction_cst == "pairwise":
            from itertools import combinations
            return [set(c) for c in combinations(range(n_features), 2)]
        if self.interaction_cst == "no_interactions":
            return [{i} for i in range(n_features)]
        
        # Convert to list of sets
        cst = []
        provided_features = set()
        for s in self.interaction_cst:
            new_set = set(s)
            cst.append(new_set)
            provided_features.update(new_set)
        
        # Add remaining features as individual sets
        for i in range(n_features):
            if i not in provided_features:
                cst.append({i})
        return cst

    def fit(self, data_or_X, y=None, sample_weight=None, **kwargs):
        """
        Build a decision tree regressor from the training set (X, y).
        """
        if y is None:
            # Handle the case where data_or_X is a DataLoader or similar
            X_list, y_list, sw_list = [], [], []
            for batch in data_or_X:
                if isinstance(batch, (list, tuple)):
                    X_list.append(batch[0])
                    y_list.append(batch[1])
                    if len(batch) > 2:
                        sw_list.append(batch[2])
                else:
                    X_list.append(batch)
            X = torch.cat(X_list, dim=0)
            y = torch.cat(y_list, dim=0)
            if sw_list:
                sample_weight = torch.cat(sw_list, dim=0)
        else:
            X, y = data_or_X, y

        X = X.to(self.device).to(self.dtype)
        y = y.to(self.device).to(self.dtype)
        
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        if sample_weight is None:
            sample_weight = torch.ones(X.shape[0], device=self.device, dtype=self.dtype)
        else:
            sample_weight = sample_weight.to(self.device).to(self.dtype)

        if not self.is_fit:
            self._init_module(X, y)

        n_samples, n_features = X.shape
        
        # Convert relative min_samples to absolute
        if isinstance(self.min_samples_split, float):
            self.min_samples_split_abs = max(2, int(self.min_samples_split * n_samples))
        else:
            self.min_samples_split_abs = self.min_samples_split

        if isinstance(self.min_samples_leaf, float):
            self.min_samples_leaf_abs = max(1, int(self.min_samples_leaf * n_samples))
        else:
            self.min_samples_leaf_abs = self.min_samples_leaf

        total_weight = sample_weight.sum()
        self.min_weight_leaf_abs = self.min_weight_fraction_leaf * total_weight

        # Parse interaction constraints once at fit time
        parsed_cst = self._parse_interaction_cst(n_features)

        if self.max_leaf_nodes is not None:
            self.tree_structure = self._build_tree_best_first(X, y, sample_weight, parsed_cst)
        else:
            self.tree_structure = self._build_tree(X, y, sample_weight, depth=0, allowed_sets=parsed_cst)
        
        if self.ccp_alpha > 0:
            self.tree_structure = self._prune_tree(self.tree_structure)

        self.is_fit = True
        return self

    def _build_tree(self, X, y, sample_weight, depth, allowed_sets=None):
        n_samples = X.shape[0]
        node_impurity = self._calculate_impurity(y, sample_weight)
        weighted_n_samples = sample_weight.sum().item()
        
        # Leaf condition
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split_abs or \
           torch.all(y == y[0]):
            return {
                'leaf': True, 
                'value': self._calculate_leaf_value(y, sample_weight), 
                'n_samples': n_samples,
                'impurity': node_impurity,
                'weighted_n_samples': weighted_n_samples
            }

        # Find best split
        best_split = self._find_best_split(X, y, sample_weight, allowed_sets=allowed_sets)
        
        if best_split is None:
            return {
                'leaf': True, 
                'value': self._calculate_leaf_value(y, sample_weight), 
                'n_samples': n_samples,
                'impurity': node_impurity,
                'weighted_n_samples': weighted_n_samples
            }

        # Recursive split
        left_mask = X[:, best_split['feature']] <= best_split['threshold']
        right_mask = ~left_mask
        
        # Check if split is valid
        if torch.sum(left_mask) < self.min_samples_leaf_abs or \
           torch.sum(right_mask) < self.min_samples_leaf_abs or \
           sample_weight[left_mask].sum() < self.min_weight_leaf_abs or \
           sample_weight[right_mask].sum() < self.min_weight_leaf_abs:
            return {
                'leaf': True, 
                'value': self._calculate_leaf_value(y, sample_weight), 
                'n_samples': n_samples,
                'impurity': node_impurity,
                'weighted_n_samples': weighted_n_samples
            }

        # Update feature importance
        self._feature_importances[best_split['feature']] += best_split['impurity_reduction'] * weighted_n_samples

        # Narrow down allowed sets for children
        child_allowed_sets = None
        if allowed_sets is not None:
            child_allowed_sets = [s for s in allowed_sets if best_split['feature'] in s]

        left_child = self._build_tree(X[left_mask], y[left_mask], sample_weight[left_mask], depth + 1, allowed_sets=child_allowed_sets)
        right_child = self._build_tree(X[right_mask], y[right_mask], sample_weight[right_mask], depth + 1, allowed_sets=child_allowed_sets)
        
        return {
            'leaf': False,
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'impurity_reduction': best_split['impurity_reduction'],
            'left': left_child,
            'right': right_child,
            'n_samples': n_samples,
            'impurity': node_impurity,
            'weighted_n_samples': weighted_n_samples
        }

    def _build_tree_best_first(self, X, y, sample_weight, initial_allowed_sets=None):
        n_samples = X.shape[0]
        node_impurity = self._calculate_impurity(y, sample_weight)
        weighted_n_samples = sample_weight.sum().item()
        
        root_split = self._find_best_split(X, y, sample_weight, allowed_sets=initial_allowed_sets)
        
        if root_split is None:
            return {
                'leaf': True, 
                'value': self._calculate_leaf_value(y, sample_weight), 
                'n_samples': n_samples,
                'impurity': node_impurity,
                'weighted_n_samples': weighted_n_samples
            }
            
        root_node = {
            'leaf': False,
            'feature': root_split['feature'],
            'threshold': root_split['threshold'],
            'impurity_reduction': root_split['impurity_reduction'],
            'X': X, 'y': y, 'sw': sample_weight,
            'depth': 0,
            'allowed_sets': initial_allowed_sets,
            'n_samples': n_samples,
            'impurity': node_impurity,
            'weighted_n_samples': weighted_n_samples
        }
        
        leaves = []
        # Priority queue stores (-gain, node_count, node)
        # Using node_count as tie-breaker
        heapq.heappush(leaves, (-root_split['impurity_reduction'] * weighted_n_samples, 0, root_node))
        
        active_nodes = 1
        
        while leaves and active_nodes < self.max_leaf_nodes:
            weighted_gain, _, node = heapq.heappop(leaves)
            
            X_node, y_node, sw_node = node['X'], node['y'], node['sw']
            left_mask = X_node[:, node['feature']] <= node['threshold']
            right_mask = ~left_mask
            
            # Update feature importance for this split
            self._feature_importances[node['feature']] += node['impurity_reduction'] * node['weighted_n_samples']

            # Create children
            for mask in [left_mask, right_mask]:
                X_child, y_child, sw_child = X_node[mask], y_node[mask], sw_node[mask]
                child_samples = X_child.shape[0]
                child_impurity = self._calculate_impurity(y_child, sw_child)
                child_weighted_n = sw_child.sum().item()
                
                # Narrow down allowed sets for children
                child_allowed_sets = None
                if node['allowed_sets'] is not None:
                    child_allowed_sets = [s for s in node['allowed_sets'] if node['feature'] in s]

                child_split = self._find_best_split(X_child, y_child, sw_child, allowed_sets=child_allowed_sets)
                
                if child_split is None or (self.max_depth is not None and node['depth'] + 1 >= self.max_depth):
                    child_node = {
                        'leaf': True, 
                        'value': self._calculate_leaf_value(y_child, sw_child), 
                        'n_samples': child_samples,
                        'impurity': child_impurity,
                        'weighted_n_samples': child_weighted_n
                    }
                else:
                    child_node = {
                        'leaf': False,
                        'feature': child_split['feature'],
                        'threshold': child_split['threshold'],
                        'impurity_reduction': child_split['impurity_reduction'],
                        'X': X_child, 'y': y_child, 'sw': sw_child,
                        'depth': node['depth'] + 1,
                        'allowed_sets': child_allowed_sets,
                        'n_samples': child_samples,
                        'impurity': child_impurity,
                        'weighted_n_samples': child_weighted_n
                    }
                    heapq.heappush(leaves, (-child_split['impurity_reduction'] * child_weighted_n, id(child_node), child_node))
                
                if mask is left_mask: node['left'] = child_node
                else: node['right'] = child_node
            
            # Remove data from internal node to save memory
            if 'X' in node: node.pop('X'); node.pop('y'); node.pop('sw')
            active_nodes += 1

        # Clean up remaining leaves in queue
        while leaves:
            _, _, node = heapq.heappop(leaves)
            if 'leaf' in node and not node['leaf']:
                node['leaf'] = True
                node['value'] = self._calculate_leaf_value(node['y'], node['sw'])
                node.pop('X'); node.pop('y'); node.pop('sw')
                
        return root_node

    def _calculate_leaf_value(self, y, sw):
        if self.criterion == "poisson":
            return torch.exp(torch.sum(sw * torch.log(y + 1e-9)) / sw.sum())
        return torch.sum(y * sw.unsqueeze(-1), dim=0) / sw.sum()

    def _find_best_split(self, X, y, sw, allowed_sets=None):
        n_samples, n_features = X.shape
        if n_samples < 2: return None
        
        best_gain = -1.0
        best_split = None
        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            
        feature_indices = torch.randperm(n_features, device=self.device)[:self.max_features_val]
        
        if allowed_sets is not None:
            # Filter feature_indices by allowed_features (union of allowed_sets)
            allowed_features = set().union(*allowed_sets)
            feature_indices = [f for f in feature_indices if f.item() in allowed_features]
            if not feature_indices:
                return None
            feature_indices = torch.tensor(feature_indices, device=self.device, dtype=torch.long)

        current_impurity = self._calculate_impurity(y, sw)
        
        for feat_idx in feature_indices:
            feat_values = X[:, feat_idx]
            if self.splitter == "random":
                thresholds = torch.tensor([feat_values[torch.randint(0, n_samples, (1,), device=self.device)]], device=self.device, dtype=self.dtype)
            else:
                thresholds = torch.unique(feat_values)
            
            if len(thresholds) <= 1: continue
            
            # Potential split points
            if self.splitter == "best":
                sorted_vals = torch.sort(thresholds).values
                split_points = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0
            else:
                split_points = thresholds

            for threshold in split_points:
                left_mask = feat_values <= threshold
                right_mask = ~left_mask
                
                n_left = left_mask.sum()
                n_right = n_samples - n_left
                sw_left = sw[left_mask].sum()
                sw_right = sw[right_mask].sum()
                
                if n_left < self.min_samples_leaf_abs or n_right < self.min_samples_leaf_abs or \
                   sw_left < self.min_weight_leaf_abs or sw_right < self.min_weight_leaf_abs:
                    continue
                
                # Monotonicity check
                if self.monotonic_cst is not None:
                    m_val = self.monotonic_cst[feat_idx]
                    if m_val != 0:
                        y_left = self._calculate_leaf_value(y[left_mask], sw[left_mask])
                        y_right = self._calculate_leaf_value(y[right_mask], sw[right_mask])
                        if (m_val == 1 and y_left > y_right) or (m_val == -1 and y_left < y_right):
                            continue

                impurity_left = self._calculate_impurity(y[left_mask], sw[left_mask])
                impurity_right = self._calculate_impurity(y[right_mask], sw[right_mask])
                
                weighted_impurity = (sw_left / sw.sum()) * impurity_left + (sw_right / sw.sum()) * impurity_right
                gain = current_impurity - weighted_impurity
                
                if self.criterion == "friedman_mse":
                    # Friedman's improved score for gain
                    diff = y[left_mask].mean() - y[right_mask].mean()
                    gain = (sw_left * sw_right / (sw_left + sw_right)) * (diff ** 2)

                if gain > best_gain and gain >= self.min_impurity_decrease:
                    best_gain = gain
                    best_split = {
                        'feature': feat_idx.item(),
                        'threshold': threshold.item(),
                        'impurity_reduction': gain.item()
                    }
        return best_split

    def _calculate_impurity(self, y, sw):
        if len(y) <= 1: return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        y_mean = torch.sum(y * sw.unsqueeze(-1), dim=0) / sw.sum()
        
        if self.criterion in ["squared_error", "mse", "friedman_mse"]:
            return torch.sum(sw.unsqueeze(-1) * (y - y_mean)**2) / sw.sum()
        elif self.criterion in ["absolute_error", "mae"]:
            y_median = torch.median(y, dim=0).values
            return torch.sum(sw.unsqueeze(-1) * torch.abs(y - y_median)) / sw.sum()
        elif self.criterion == "poisson":
            y_pred = torch.clamp(y_mean, min=1e-9)
            y_safe = torch.clamp(y, min=1e-9)
            return 2 * torch.sum(sw.unsqueeze(-1) * (y * torch.log(y_safe / y_pred) - (y - y_pred))) / sw.sum()
        
        return torch.sum(sw.unsqueeze(-1) * (y - y_mean)**2) / sw.sum()

    def _prune_tree(self, node):
        if node['leaf']:
            return node
            
        node['left'] = self._prune_tree(node['left'])
        node['right'] = self._prune_tree(node['right'])
        
        if node['left']['leaf'] and node['right']['leaf']:
            # Total error of children
            error_children = (node['left']['impurity'] * node['left']['weighted_n_samples'] + 
                             node['right']['impurity'] * node['right']['weighted_n_samples'])
            # Error if this node was a leaf
            error_node = node['impurity'] * node['weighted_n_samples']
            
            # alpha = (error_node - error_children) / (n_leaves - 1)
            # Simplified pruning for ccp_alpha
            if (error_node - error_children) / node['weighted_n_samples'] <= self.ccp_alpha:
                # Calculate value for leaf
                # Note: We don't have X, y here, so we'd need to store 'value' in internal nodes too
                # or pass it up. Let's assume we store it.
                # Since I didn't store value in internal nodes yet, I'll add a quick fix.
                # Actually, in trees, value of internal node is mean of children's values weighted by samples.
                val_left = node['left']['value']
                val_right = node['right']['value']
                w_left = node['left']['weighted_n_samples']
                w_right = node['right']['weighted_n_samples']
                leaf_value = (val_left * w_left + val_right * w_right) / (w_left + w_right)
                
                return {
                    'leaf': True, 
                    'value': leaf_value, 
                    'n_samples': node['n_samples'],
                    'impurity': node['impurity'],
                    'weighted_n_samples': node['weighted_n_samples']
                }
        return node

    def predict(self, X):
        if not self.is_fit:
            # MLModule.forward should handle this, but if we are here we might need a default
            # In trees, we can't really do anything without fit
            return torch.zeros((X.shape[0], self.out_features or 1), device=self.device, dtype=self.dtype)
            
        X = X.to(self.device).to(self.dtype)
        if X.shape[1] == 0:
            return torch.zeros((X.shape[0], self.out_features or 1), device=self.device, dtype=self.dtype)
            
        predictions = torch.stack([self._predict_single(x, self.tree_structure) for x in X])
        return predictions

    def _predict_single(self, x, node):
        if node['leaf']:
            return node['value']
        
        feat_val = x[node['feature']]
        if torch.isnan(feat_val):
            # Stochastic NaN handling during prediction
            if torch.rand(1, device=self.device, dtype=self.dtype) > 0.5:
                return self._predict_single(x, node['left'])
            else:
                return self._predict_single(x, node['right'])

        if feat_val <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])


class ExtraTreeRegressor(DecisionTreeRegressor):
    def __init__(self,
                 criterion: str = "squared_error",
                 splitter: str = "random",
                 max_depth: int = None,
                 min_samples_split: Union[int, float] = 2,
                 min_samples_leaf: Union[int, float] = 1,
                 min_weight_fraction_leaf: float = 0.0,
                 max_features: Union[int, float, str] = 1.0,
                 random_state: int = None,
                 max_leaf_nodes: int = None,
                 min_impurity_decrease: float = 0.0,
                 ccp_alpha: float = 0.0,
                 monotonic_cst: Union[List[int], Tuple[int], torch.Tensor] = None,
                 interaction_cst: Union[str, List[Any], Tuple[Any]] = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst,
            interaction_cst=interaction_cst,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )

    def _find_best_split(self, X, y, sw, allowed_sets=None):
        n_samples, n_features = X.shape
        if n_samples < 2: return None
        
        best_gain = -1.0
        best_split = None
        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            
        feature_indices = torch.randperm(n_features, device=self.device)[:self.max_features_val]
        
        if allowed_sets is not None:
            # Filter feature_indices by allowed_features (union of allowed_sets)
            allowed_features = set().union(*allowed_sets)
            feature_indices = [f for f in feature_indices if f.item() in allowed_features]
            if not feature_indices:
                return None
            feature_indices = torch.tensor(feature_indices, device=self.device, dtype=torch.long)

        current_impurity = self._calculate_impurity(y, sw)
        
        for feat_idx in feature_indices:
            feat_values = X[:, feat_idx]
            
            # Mask for non-missing values
            valid_mask = ~torch.isnan(feat_values)
            if not torch.any(valid_mask):
                continue
            
            v_min, v_max = feat_values[valid_mask].min(), feat_values[valid_mask].max()
            if v_min == v_max:
                continue
                
            # Extremely Randomized Trees: draw one random threshold uniformly per feature
            # between its observed min and max in this node
            threshold = torch.rand(1, device=self.device, dtype=self.dtype) * (v_max - v_min) + v_min
            threshold = threshold.item()

            nan_mask = torch.isnan(feat_values)
            # Standard comparison for non-NaNs
            left_mask = feat_values <= threshold
            # Randomly assign NaNs to left (50/50 chance)
            if torch.any(nan_mask):
                random_nan_direction = torch.rand(nan_mask.sum().item(), device=self.device, dtype=self.dtype) > 0.5
                left_mask[nan_mask] = random_nan_direction
            
            right_mask = ~left_mask
            
            n_left = left_mask.sum()
            n_right = n_samples - n_left
            sw_left = sw[left_mask].sum()
            sw_right = sw[right_mask].sum()
            
            if n_left < self.min_samples_leaf_abs or n_right < self.min_samples_leaf_abs or \
               sw_left < self.min_weight_leaf_abs or sw_right < self.min_weight_leaf_abs:
                continue

            # Calculate gains based on this random split
            impurity_left = self._calculate_impurity(y[left_mask], sw[left_mask])
            impurity_right = self._calculate_impurity(y[right_mask], sw[right_mask])
            
            weighted_impurity = (sw_left / sw.sum()) * impurity_left + (sw_right / sw.sum()) * impurity_right
            gain = current_impurity - weighted_impurity
            
            if gain > best_gain and gain >= self.min_impurity_decrease:
                best_gain = gain
                best_split = {
                    'feature': feat_idx.item(),
                    'threshold': threshold,
                    'impurity_reduction': gain.item()
                }
        return best_split

