import torch
from typing import Union, Any, Tuple, List
import math
from .....models.utils import MLClassifier
from torch.func import vmap
import joblib

__all__ = ["DecisionTreeClassifier", "ExtraTreeClassifier"]


class DecisionTreeClassifier(MLClassifier):
    def __init__(self,
                 criterion: str = "gini",  # {'gini', 'entropy', 'log-loss'}
                 splitter: str = "best",
                 max_depth: int = None,
                 min_samples_split: Union[int, float] = 2,
                 min_samples_leaf: Union[int, float] = 1,
                 min_weight_fraction_leaf: float = 0.0,
                 max_features: Union[int, float, str] = None,
                 random_state: int = None,
                 max_leaf_nodes: int = None,
                 min_impurity_decrease: float = 0.0,
                 class_weight: Union[dict, str] = None,
                 ccp_alpha: float = 0.0,
                 monotonic_cst: Union[List[int], Tuple[int], torch.Tensor] = None,
                 interaction_cst: Union[str, List[Any], Tuple[Any]] = None,
                 warm_start: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.warm_start = warm_start
        self.classes_ = None
        self.n_classes_ = None
        self.feature_importances_ = None
        self.max_features_ = None
        self.n_features_in_ = None
        self.n_outputs_ = None
        self.tree_ = None
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
        self.class_weight = class_weight
        self.is_fit = False
        self._tree_structure = None

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        # Handle tensor lists properly for serialization
        classes_sd = self.classes_
        if isinstance(self.classes_, list):
            classes_sd = {f"class_{i}": c for i, c in enumerate(self.classes_)}
            
        sd['_tree_metadata'] = {
            'is_fit': self.is_fit,
            'n_features_in_': self.n_features_in_,
            'n_outputs_': self.n_outputs_,
            'n_classes_': self.n_classes_,
            'classes_': classes_sd,
            'max_features_': self.max_features_,
            'min_samples_split_abs': getattr(self, 'min_samples_split_abs', None),
            'min_samples_leaf_abs': getattr(self, 'min_samples_leaf_abs', None),
            'min_weight_leaf_abs': getattr(self, 'min_weight_leaf_abs', None),
            'tree_': self._tree_structure
        }
        return sd

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        metadata = state_dict.pop('_tree_metadata', None)
        if metadata:
            self.is_fit = metadata.get('is_fit', False)
            self.n_features_in_ = metadata.get('n_features_in_')
            self.n_outputs_ = metadata.get('n_outputs_')
            self.n_classes_ = metadata.get('n_classes_')
            
            # Reconstruct classes_
            classes_sd = metadata.get('classes_')
            if isinstance(classes_sd, dict) and 'class_0' in classes_sd:
                self.classes_ = [classes_sd[f"class_{i}"] for i in range(len(classes_sd))]
            else:
                self.classes_ = classes_sd
                
            self.max_features_ = metadata.get('max_features_')
            self.min_samples_split_abs = metadata.get('min_samples_split_abs')
            self.min_samples_leaf_abs = metadata.get('min_samples_leaf_abs')
            self.min_weight_leaf_abs = metadata.get('min_weight_leaf_abs')
            self._tree_structure = metadata.get('tree_')
            self.tree_ = self._tree_structure # Alias
        
        if 'feature_importances_' in state_dict:
            importances = state_dict['feature_importances_']
            if not hasattr(self, 'feature_importances_') or 'feature_importances_' not in self._buffers:
                if hasattr(self, 'feature_importances_'):
                    del self.feature_importances_
                self.register_buffer('feature_importances_', torch.zeros_like(importances))
            
        try:
            return super().load_state_dict(state_dict, strict=strict, *args, **kwargs)
        except RuntimeError:
            return super().load_state_dict(state_dict, strict=False, *args, **kwargs)

    def __deepcopy__(self, memo):
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == '_modules':
                setattr(result, k, copy.deepcopy(v, memo))
            elif k == '_buffers':
                setattr(result, k, copy.deepcopy(v, memo))
            elif k == '_parameters':
                setattr(result, k, copy.deepcopy(v, memo))
            elif isinstance(v, torch.Tensor):
                setattr(result, k, v.clone().detach())
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def _init_module(self, X, y):
        self.n_features_in_ = X.size(-1)
        self.n_outputs_ = y.size(-1) if y.ndim > 1 else 1
        
        # Handle max_features
        if self.max_features is None:
            self.max_features_ = self.n_features_in_
        elif isinstance(self.max_features, float):
            self.max_features_ = max(1, int(self.max_features * self.n_features_in_))
        elif isinstance(self.max_features, int):
            self.max_features_ = self.max_features
        elif isinstance(self.max_features, str):
            if self.max_features.lower() == "sqrt":
                self.max_features_ = int(math.sqrt(self.n_features_in_))
            elif self.max_features.lower() == "log2":
                self.max_features_ = int(math.log2(self.n_features_in_))
            else:
                self.max_features_ = self.n_features_in_

        # Handle classes
        if y.ndim == 1:
            y = y.unsqueeze(1)
        self.n_outputs_ = y.shape[1]
        self.classes_ = []
        n_classes = []
        for i in range(self.n_outputs_):
            classes = torch.unique(y[:, i])
            self.classes_.append(classes)
            n_classes.append(len(classes))
            
        if self.n_outputs_ == 1:
            self.classes_ = self.classes_[0]
            self.n_classes_ = n_classes[0]
        else:
            self.n_classes_ = n_classes

        if not hasattr(self, 'feature_importances_') or 'feature_importances_' not in self._buffers:
             if hasattr(self, 'feature_importances_'):
                 del self.feature_importances_
             self.register_buffer('feature_importances_', torch.zeros(self.n_features_in_, device=self.device, dtype=self.dtype))
        else:
            self.feature_importances_ = torch.zeros(self.n_features_in_, device=self.device, dtype=self.dtype)
        
        return self

    def _get_class_weights(self, y):
        """Calculate weight for each sample based on class_weight."""
        n_samples = y.shape[0]
        if self.class_weight is None:
            return torch.ones(n_samples, device=self.device, dtype=self.dtype)
        
        if isinstance(self.class_weight, str) and self.class_weight == 'balanced':
            # weight = n_samples / (n_classes * bincount(y))
            weights = torch.ones(n_samples, device=self.device, dtype=self.dtype)
            if self.n_outputs_ == 1:
                counts = torch.bincount(y.flatten().long())
                for i, class_label in enumerate(self.classes_):
                    if counts[class_label.long()] > 0:
                        class_w = n_samples / (self.n_classes_ * counts[class_label.long()])
                        weights[y.flatten() == class_label] = class_w
            else:
                # Multi-output balanced weighting is more complex, usually handled per output
                for i in range(self.n_outputs_):
                    counts = torch.bincount(y[:, i].long())
                    for j, class_label in enumerate(self.classes_[i]):
                        if counts[class_label.long()] > 0:
                            class_w = n_samples / (self.n_classes_[i] * counts[class_label.long()])
                            weights[y[:, i] == class_label] *= class_w
            return weights
        
        if isinstance(self.class_weight, dict):
            weights = torch.ones(n_samples, device=self.device, dtype=self.dtype)
            for class_label, weight in self.class_weight.items():
                weights[y.flatten() == class_label] = weight
            return weights
            
        return torch.ones(n_samples, device=self.device, dtype=self.dtype)

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
        if y is None:
            X_list, y_list, sw_list = [], [], []
            for batch in data_or_X:
                X_list.append(batch[0]); y_list.append(batch[1])
                if len(batch) > 2: sw_list.append(batch[2])
            X = torch.cat(X_list, dim=0); y = torch.cat(y_list, dim=0)
            if sw_list: sample_weight = torch.cat(sw_list, dim=0)
        else:
            X, y = data_or_X, y

        X = X.to(self.device).to(self.dtype)
        y = y.to(self.device).to(self.dtype)
        
        if not self.is_fit:
            self._init_module(X, y)

        if sample_weight is None:
            sample_weight = torch.ones(X.shape[0], device=self.device, dtype=self.dtype)
        else:
            sample_weight = sample_weight.to(self.device).to(self.dtype)

        # Combine with class_weight
        sample_weight *= self._get_class_weights(y)

        n_samples, n_features = X.shape
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

        # Parse interaction constraints
        parsed_cst = self._parse_interaction_cst(n_features)

        if self.max_leaf_nodes is not None:
            self._tree_structure = self._build_tree_best_first(X, y, sample_weight, parsed_cst)
        else:
            self._tree_structure = self._build_tree(X, y, sample_weight, depth=0, allowed_sets=parsed_cst)
        
        self.tree_ = self._tree_structure

        if self.ccp_alpha > 0:
            self._tree_structure = self._prune_tree(self._tree_structure)
            self.tree_ = self._tree_structure

        self.is_fit = True
        return self

    def _calculate_impurity(self, y, sw):
        if len(y) <= 1: return torch.tensor(0.0, device=self.device)
        sw_sum = sw.sum()
        if sw_sum == 0: return torch.tensor(0.0, device=self.device)

        if self.n_outputs_ == 1:
            y_long = y.flatten().long()
            # Handle multi-class Gini/Entropy
            probs = torch.zeros(self.n_classes_, device=self.device, dtype=self.dtype)
            for i, class_label in enumerate(self.classes_):
                mask = (y_long == class_label.long())
                probs[i] = sw[mask].sum() / sw_sum
            
            if self.criterion == "gini":
                return 1.0 - torch.sum(probs**2)
            elif self.criterion in ["entropy", "log_loss"]:
                probs = torch.clamp(probs, min=1e-9)
                return -torch.sum(probs * torch.log(probs))
        else:
            # Multi-output: average impurity across outputs
            total_impurity = 0.0
            for i in range(self.n_outputs_):
                y_i = y[:, i].long()
                probs = torch.zeros(self.n_classes_[i], device=self.device, dtype=self.dtype)
                for j, class_label in enumerate(self.classes_[i]):
                    mask = (y_i == class_label.long())
                    probs[j] = sw[mask].sum() / sw_sum
                
                if self.criterion == "gini":
                    total_impurity += (1.0 - torch.sum(probs**2))
                elif self.criterion in ["entropy", "log_loss"]:
                    probs = torch.clamp(probs, min=1e-9)
                    total_impurity += -torch.sum(probs * torch.log(probs))
            return torch.tensor(total_impurity / self.n_outputs_, device=self.device)

        return torch.tensor(0.0, device=self.device)

    def _calculate_leaf_value(self, y, sw):
        sw_sum = sw.sum()
        if self.n_outputs_ == 1:
            y_long = y.flatten().long()
            probs = torch.zeros(self.n_classes_, device=self.device, dtype=self.dtype)
            for i, class_label in enumerate(self.classes_):
                mask = (y_long == class_label.long())
                probs[i] = sw[mask].sum() / sw_sum
            return probs
        else:
            all_probs = []
            for i in range(self.n_outputs_):
                y_i = y[:, i].long()
                probs = torch.zeros(self.n_classes_[i], device=self.device, dtype=self.dtype)
                for j, class_label in enumerate(self.classes_[i]):
                    mask = (y_i == class_label.long())
                    probs[j] = sw[mask].sum() / sw_sum
                all_probs.append(probs)
            return all_probs

    def _find_best_split(self, X, y, sw, allowed_sets=None):
        n_samples, n_features = X.shape
        if n_samples < 2: return None
        
        best_gain = -1.0
        best_split = None
        
        if self.random_state is not None:
            if isinstance(self.random_state, int):
                torch.manual_seed(self.random_state)
            
        feature_indices = torch.randperm(n_features, device=self.device)[:self.max_features_]
        
        if allowed_sets is not None:
            allowed_features = set().union(*allowed_sets)
            feature_indices = [f for f in feature_indices if f.item() in allowed_features]
            if len(feature_indices) == 0:
                return None
            feature_indices = torch.tensor(feature_indices, device=self.device)

        current_impurity = self._calculate_impurity(y, sw)
        
        for feat_idx in feature_indices:
            feat_values = X[:, feat_idx]
            if self.splitter == "random":
                thresholds = feat_values[torch.randint(0, n_samples, (1,))]
            else:
                thresholds = torch.unique(feat_values)
            
            if len(thresholds) <= 1: continue
            
            if self.splitter == "best":
                sorted_vals = torch.sort(thresholds).values
                split_points = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0
            else:
                split_points = thresholds

            for threshold in split_points:
                left_mask = feat_values <= threshold
                right_mask = ~left_mask
                
                n_left = left_mask.sum(); n_right = n_samples - n_left
                sw_left = sw[left_mask].sum(); sw_right = sw[right_mask].sum()
                
                if n_left < self.min_samples_leaf_abs or n_right < self.min_samples_leaf_abs or \
                   sw_left < self.min_weight_leaf_abs or sw_right < self.min_weight_leaf_abs:
                    continue
                
                # Monotonicity check (on positive class prob if binary)
                if self.monotonic_cst is not None and self.n_outputs_ == 1 and self.n_classes_ == 2:
                    m_val = self.monotonic_cst[feat_idx]
                    if m_val != 0:
                        v_left = self._calculate_leaf_value(y[left_mask], sw[left_mask])[1] # prob of class 1
                        v_right = self._calculate_leaf_value(y[right_mask], sw[right_mask])[1]
                        if (m_val == 1 and v_left > v_right) or (m_val == -1 and v_left < v_right):
                            continue

                impurity_left = self._calculate_impurity(y[left_mask], sw[left_mask])
                impurity_right = self._calculate_impurity(y[right_mask], sw[right_mask])
                
                weighted_impurity = (sw_left / sw.sum()) * impurity_left + (sw_right / sw.sum()) * impurity_right
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain and gain >= self.min_impurity_decrease:
                    best_gain = gain
                    best_split = {
                        'feature': feat_idx.item(),
                        'threshold': threshold.item(),
                        'impurity_reduction': gain.item()
                    }
        return best_split

    def _build_tree(self, X, y, sw, depth, allowed_sets=None):
        n_samples = X.shape[0]
        node_impurity = self._calculate_impurity(y, sw)
        weighted_n_samples = sw.sum().item()
        
        # Pure leaf check
        is_pure = False
        if self.n_outputs_ == 1:
            is_pure = len(torch.unique(y)) == 1
        else:
            is_pure = all(len(torch.unique(y[:, i])) == 1 for i in range(self.n_outputs_))

        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split_abs or is_pure:
            return {
                'leaf': True, 'value': self._calculate_leaf_value(y, sw), 
                'n_samples': n_samples, 'impurity': node_impurity, 'weighted_n_samples': weighted_n_samples
            }

        best_split = self._find_best_split(X, y, sw, allowed_sets=allowed_sets)
        if best_split is None:
            return {
                'leaf': True, 'value': self._calculate_leaf_value(y, sw), 
                'n_samples': n_samples, 'impurity': node_impurity, 'weighted_n_samples': weighted_n_samples
            }

        left_mask = X[:, best_split['feature']] <= best_split['threshold']
        right_mask = ~left_mask
        
        # Valid split check
        if left_mask.sum() < self.min_samples_leaf_abs or right_mask.sum() < self.min_samples_leaf_abs or \
           sw[left_mask].sum() < self.min_weight_leaf_abs or sw[right_mask].sum() < self.min_weight_leaf_abs:
            return {
                'leaf': True, 'value': self._calculate_leaf_value(y, sw), 
                'n_samples': n_samples, 'impurity': node_impurity, 'weighted_n_samples': weighted_n_samples
            }

        self.feature_importances_[best_split['feature']] += best_split['impurity_reduction'] * weighted_n_samples

        # Narrow down allowed sets for children
        child_allowed_sets = None
        if allowed_sets is not None:
            child_allowed_sets = [s for s in allowed_sets if best_split['feature'] in s]

        left_child = self._build_tree(X[left_mask], y[left_mask], sw[left_mask], depth + 1, allowed_sets=child_allowed_sets)
        right_child = self._build_tree(X[right_mask], y[right_mask], sw[right_mask], depth + 1, allowed_sets=child_allowed_sets)
        
        return {
            'leaf': False, 'feature': best_split['feature'], 'threshold': best_split['threshold'],
            'impurity_reduction': best_split['impurity_reduction'], 'left': left_child, 'right': right_child,
            'n_samples': n_samples, 'impurity': node_impurity, 'weighted_n_samples': weighted_n_samples,
            'value': self._calculate_leaf_value(y, sw) # Store for pruning
        }

    def _build_tree_best_first(self, X, y, sw, initial_allowed_sets=None):
        import heapq
        n_samples = X.shape[0]
        node_impurity = self._calculate_impurity(y, sw)
        weighted_n_samples = sw.sum().item()
        
        root_split = self._find_best_split(X, y, sw, allowed_sets=initial_allowed_sets)
        if root_split is None:
            return {
                'leaf': True, 'value': self._calculate_leaf_value(y, sw), 
                'n_samples': n_samples, 'impurity': node_impurity, 'weighted_n_samples': weighted_n_samples
            }
            
        root_node = {
            'leaf': False, 'feature': root_split['feature'], 'threshold': root_split['threshold'],
            'impurity_reduction': root_split['impurity_reduction'], 'X': X, 'y': y, 'sw': sw, 'depth': 0,
            'allowed_sets': initial_allowed_sets,
            'n_samples': n_samples, 'impurity': node_impurity, 'weighted_n_samples': weighted_n_samples,
            'value': self._calculate_leaf_value(y, sw)
        }
        
        leaves = []
        heapq.heappush(leaves, (-root_split['impurity_reduction'] * weighted_n_samples, 0, root_node))
        active_nodes = 1
        
        while leaves and active_nodes < self.max_leaf_nodes:
            _, _, node = heapq.heappop(leaves)
            X_n, y_n, sw_n = node['X'], node['y'], node['sw']
            l_mask = X_n[:, node['feature']] <= node['threshold']; r_mask = ~l_mask
            self.feature_importances_[node['feature']] += node['impurity_reduction'] * node['weighted_n_samples']

            # Narrow down allowed sets for children
            child_allowed_sets = None
            if node['allowed_sets'] is not None:
                child_allowed_sets = [s for s in node['allowed_sets'] if node['feature'] in s]

            for mask in [l_mask, r_mask]:
                X_h, y_h, sw_h = X_n[mask], y_n[mask], sw_n[mask]
                child_samples = X_h.shape[0]; child_impurity = self._calculate_impurity(y_h, sw_h)
                child_w_n = sw_h.sum().item()
                
                is_pure = False
                if self.n_outputs_ == 1: is_pure = len(torch.unique(y_h)) == 1
                else: is_pure = all(len(torch.unique(y_h[:, i])) == 1 for i in range(self.n_outputs_))

                split = self._find_best_split(X_h, y_h, sw_h, allowed_sets=child_allowed_sets)
                if split is None or (self.max_depth is not None and node['depth'] + 1 >= self.max_depth) or is_pure:
                    child = {
                        'leaf': True, 'value': self._calculate_leaf_value(y_h, sw_h), 
                        'n_samples': child_samples, 'impurity': child_impurity, 'weighted_n_samples': child_w_n
                    }
                else:
                    child = {
                        'leaf': False, 'feature': split['feature'], 'threshold': split['threshold'],
                        'impurity_reduction': split['impurity_reduction'], 'X': X_h, 'y': y_h, 'sw': sw_h,
                        'depth': node['depth'] + 1, 'allowed_sets': child_allowed_sets,
                        'n_samples': child_samples, 'impurity': child_impurity, 
                        'weighted_n_samples': child_w_n, 'value': self._calculate_leaf_value(y_h, sw_h)
                    }
                    heapq.heappush(leaves, (-split['impurity_reduction'] * child_w_n, id(child), child))
                
                if mask is l_mask: node['left'] = child
                else: node['right'] = child
            
            node.pop('X'); node.pop('y'); node.pop('sw'); node.pop('allowed_sets'); active_nodes += 1

        while leaves:
            _, _, node = heapq.heappop(leaves)
            if not node['leaf']:
                node['leaf'] = True
                node.pop('X'); node.pop('y'); node.pop('sw')
        return root_node

    def _predict_single_proba(self, x, node):
        if node['leaf']: return node['value']
        if torch.isnan(x[node['feature']]):
            return self._predict_single_proba(x, node['left']) if torch.rand(1) > 0.5 else self._predict_single_proba(x, node['right'])
        return self._predict_single_proba(x, node['left']) if x[node['feature']] <= node['threshold'] else self._predict_single_proba(x, node['right'])

    def predict_proba(self, X):
        if not self.is_fit:
            n_outputs = self.n_outputs_ if self.n_outputs_ is not None else 1
            if n_outputs == 1:
                n_classes = self.n_classes_ if self.n_classes_ is not None else 1
                return torch.zeros((X.shape[0], n_classes), device=self.device, dtype=self.dtype)
            n_classes_list = self.n_classes_ if self.n_classes_ is not None else [1] * n_outputs
            return [torch.zeros((X.shape[0], c), device=self.device, dtype=self.dtype) for c in n_classes_list]
        
        X = X.to(self.device).to(self.dtype)
        
        if X.shape[1] == 0:
            if self.n_outputs_ == 1:
                return torch.zeros((X.shape[0], self.n_classes_ if self.n_classes_ is not None else 1), device=self.device, dtype=self.dtype)
            return [torch.zeros((X.shape[0], c if c is not None else 1), device=self.device, dtype=self.dtype) for c in self.n_classes_]
            
        probas = [self._predict_single_proba(x, self._tree_structure) for x in X]
        
        if self.n_outputs_ == 1:
            return torch.stack(probas)
        else:
            return [torch.stack([p[i] for p in probas]) for i in range(self.n_outputs_)]

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """
        Decision function for DecisionTreeClassifier.
        Returns the same as predict_proba for trees.
        """
        return self.predict_proba(X)

    def predict_log_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict logarithm of probabilities.
        """
        proba = self.predict_proba(X)
        if self.n_outputs_ == 1:
            return torch.log(torch.clamp(proba, min=1e-15))
        else:
            return [torch.log(torch.clamp(p, min=1e-15)) for p in proba]

    def predict(self, X):
        probas = self.predict_proba(X)
        if self.n_outputs_ == 1:
            if probas.shape[1] == 0:
                # Fallback if no classes/features
                default_class = self.classes_[0] if self.classes_ is not None and len(self.classes_) > 0 else 0
                return torch.full((X.shape[0],), default_class, device=self.device, dtype=self.classes_.dtype if isinstance(self.classes_, torch.Tensor) else torch.int64)
            return self.classes_[torch.argmax(probas, dim=1)]
        else:
            out = []
            for i in range(self.n_outputs_):
                if probas[i].shape[1] == 0:
                    default_class = self.classes_[i][0] if self.classes_[i] is not None and len(self.classes_[i]) > 0 else 0
                    out.append(torch.full((X.shape[0],), default_class, device=self.device, dtype=self.classes_[i].dtype if isinstance(self.classes_[i], torch.Tensor) else torch.int64))
                else:
                    out.append(self.classes_[i][torch.argmax(probas[i], dim=1)])
            return torch.stack(out, dim=1)

    def _prune_tree(self, node):
        if node['leaf']: return node
        node['left'] = self._prune_tree(node['left'])
        node['right'] = self._prune_tree(node['right'])
        
        if node['left']['leaf'] and node['right']['leaf']:
            err_children = (node['left']['impurity'] * node['left']['weighted_n_samples'] + 
                           node['right']['impurity'] * node['right']['weighted_n_samples'])
            err_node = node['impurity'] * node['weighted_n_samples']
            if (err_node - err_children) / node['weighted_n_samples'] <= self.ccp_alpha:
                return {
                    'leaf': True, 'value': node['value'], 'n_samples': node['n_samples'],
                    'impurity': node['impurity'], 'weighted_n_samples': node['weighted_n_samples']
                }
        return node


class ExtraTreeClassifier(DecisionTreeClassifier):
    """
    An extremely randomized tree classifier.
    """
    def __init__(self,
                 criterion: str = "gini",
                 splitter: str = "random",
                 max_depth: int = None,
                 min_samples_split: Union[int, float] = 2,
                 min_samples_leaf: Union[int, float] = 1,
                 min_weight_fraction_leaf: float = 0.0,
                 max_features: Union[int, float, str] = "sqrt",
                 random_state: int = None,
                 max_leaf_nodes: int = None,
                 min_impurity_decrease: float = 0.0,
                 class_weight: Union[dict, str] = None,
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
            class_weight=class_weight,
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
            if isinstance(self.random_state, int):
                torch.manual_seed(self.random_state)
            
        feature_indices = torch.randperm(n_features, device=self.device)[:self.max_features_]

        if allowed_sets is not None:
            allowed_features = set().union(*allowed_sets)
            feature_indices = [f for f in feature_indices if f.item() in allowed_features]
            if len(feature_indices) == 0:
                return None
            feature_indices = torch.tensor(feature_indices, device=self.device)

        current_impurity = self._calculate_impurity(y, sw)
        
        for feat_idx in feature_indices:
            feat_values = X[:, feat_idx]
            valid_mask = ~torch.isnan(feat_values)
            if not torch.any(valid_mask): continue
            
            v_min, v_max = feat_values[valid_mask].min(), feat_values[valid_mask].max()
            if v_min == v_max: continue
                
            # Random threshold between min and max
            threshold = (torch.rand(1, device=self.device) * (v_max - v_min) + v_min).item()

            left_mask = feat_values <= threshold
            # Randomly assign NaNs
            nan_mask = torch.isnan(feat_values)
            if torch.any(nan_mask):
                left_mask[nan_mask] = torch.rand(nan_mask.sum().item(), device=self.device) > 0.5
            
            right_mask = ~left_mask
            n_left = left_mask.sum(); n_right = n_samples - n_left
            sw_left = sw[left_mask].sum(); sw_right = sw[right_mask].sum()
            
            if n_left < self.min_samples_leaf_abs or n_right < self.min_samples_leaf_abs or \
               sw_left < self.min_weight_leaf_abs or sw_right < self.min_weight_leaf_abs:
                continue

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
