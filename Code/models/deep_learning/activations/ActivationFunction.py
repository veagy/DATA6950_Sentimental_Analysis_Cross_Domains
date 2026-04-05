import torch
import torch.nn as nn
import json
import re
import os
import importlib
import logging
from ....models.deep_learning.activations.utils._utils import Forward_hook, Backward_hook
from typing import Dict, Any, List
__all__ = [
    'Activation',
    'ExpressionParser',
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prefer PyTorch built-ins for common activations (avoids broken registry / heavy imports).
_TORCH_NN_ACTIVATIONS: Dict[str, type] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "mish": nn.Mish,
    "leaky_relu": nn.LeakyReLU,
    "prelu": nn.PReLU,
}


def _normalize_torch_activation_key(name: str) -> str:
    s = name.strip().lower()
    if s.endswith("activation"):
        s = s[: -len("activation")].rstrip("_")
    return s


class Activation(nn.Module):
    """
    Dynamic Activation factory class that loads activation functions from the registry.
    Supports exact name matching and regex searching.
    """
    _registry_data = None
    _name_to_info = {} # Maps class name to (module_path, class_name)
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _root_module = "Code.models.deep_learning.activations"

    def __init__(self, activation: str, *args, **kwargs):
        super().__init__()
        self.debug_mode = kwargs.pop('debug', False)

        torch_key = _normalize_torch_activation_key(activation)
        if torch_key in _TORCH_NN_ACTIVATIONS:
            # layer_config often includes device/dtype; torch activations do not take them.
            for k in ("device", "dtype"):
                kwargs.pop(k, None)
            cls_mod = _TORCH_NN_ACTIVATIONS[torch_key]
            self.op = cls_mod(*args, **kwargs)
            self.activation_name = cls_mod.__name__
            self.debug = self.debug_mode
            if self.debug:
                self.op.register_forward_hook(Forward_hook(name=self.activation_name))
                self.op.register_full_backward_hook(Backward_hook(name=self.activation_name))
            return

        if Activation._registry_data is None:
            self._initialize_registry()
            
        target_info = self._resolve_activation(activation)
        
        if target_info:
            module_path, class_name = target_info
            try:
                module = importlib.import_module(module_path)
                activation_class = getattr(module, class_name)
                self.op = activation_class(*args, **kwargs)
                self.activation_name = class_name
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to load activation {class_name} from {module_path}: {e}")
                raise ImportError(f"Could not load '{class_name}' from '{module_path}'.")
        else:
            # Fallback: Try CustomStringActivationLayer (GLU/Real version default)
            # This handles both raw formulas ("w1*x") and registry names ("my_act")
            # provided the CustomStringActivationLayer logic supports it.
            try:
                # Lazy import to avoid circular dependency
                from ....models.deep_learning.activations.GatedLinearUnits.GLUFamliyActivations import CustomStringActivationLayer
                
                if 'in_features' not in kwargs:
                     raise ValueError(f"Activation '{activation}' not found in registry. If this is a custom string/formula, 'in_features' must be provided in kwargs.")

                act_funcs = kwargs.pop('act_funcs', [])
                self.op = CustomStringActivationLayer(
                    act_operation=activation,
                    act_funcs=act_funcs,
                    *args, 
                    **kwargs
                )
                self.activation_name = f"CustomString({activation})"
                
            except Exception as e:
                # If both fail, raise original error context
                raise ValueError(f"Activation '{activation}' not found in registry and failed to instantiate as CustomString: {e}")

        # Hook registration and Debugging
        self.debug = self.debug_mode # Set earlier
        if self.debug:
            self.op.register_forward_hook(Forward_hook(name=self.activation_name))
            self.op.register_full_backward_hook(Backward_hook(name=self.activation_name))

    def _validate_tensor(self, x: torch.Tensor, stage: str):
        if torch.isnan(x).any():
            logger.warning(f"NaN detected in {stage} of {self.activation_name}")
        if torch.isinf(x).any():
            logger.warning(f"Inf detected in {stage} of {self.activation_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug:
            self._validate_tensor(x, "input")
            
        out = self.op(x)
        
        if self.debug:
            self._validate_tensor(out, "output")
            
        return out

    @classmethod
    def _initialize_registry(cls):
        registry_path = os.path.join(cls._base_dir, "__registry__.json")
        if not os.path.exists(registry_path):
            logger.warning(f"Registry file not found at {registry_path}")
            return

        with open(registry_path, 'r') as f:
            cls._registry_data = json.load(f)
            if isinstance(cls._registry_data, list) and len(cls._registry_data) > 0:
                cls._registry_data = cls._registry_data[0]
        
        cls._build_mapping(cls._registry_data, [])

    @classmethod
    def _build_mapping(cls, data, path_parts):
        """
        Recursively walk the registry and build ClassName -> ModulePath mapping.
        """
        for key, value in data.items():
            if isinstance(value, list):
                # 'key' is likely a filename, 'value' is list of classes
                module_path = cls._resolve_module_path(path_parts + [key])
                for class_name in value:
                    if class_name: # Skip empty strings
                        cls._name_to_info[class_name] = (module_path, class_name)
            elif isinstance(value, dict):
                # 'key' is a category/directory
                cls._build_mapping(value, path_parts + [key])

    @classmethod
    def _resolve_module_path(cls, parts):
        """
        Translates registry hierarchy into a valid python module path, 
        handling known discrepancies and fuzzy directory matching.
        """
        current_abs_path = cls._base_dir
        resolved_parts = []
        
        # Mappings for known top-level discrepancies
        top_level_remap = {
            "GatedLearningUnits": "GatedLinearUnits",
            "ComplexActivations": "Complex",
        }

        for i, part in enumerate(parts):
            target = part
            if i == 0:
                target = top_level_remap.get(part, part)
            
            # Try to find the directory or file with fuzzy matching
            actual_name = cls._find_actual_name(current_abs_path, target, is_last=(i == len(parts) - 1))
            
            if not actual_name and i > 0:
                # If not found, try remapping anyway as a fallback for sub-parts
                remapped_target = top_level_remap.get(target, target)
                if remapped_target != target:
                    actual_name = cls._find_actual_name(current_abs_path, remapped_target, is_last=(i == len(parts) - 1))

            if actual_name:
                resolved_parts.append(actual_name.replace('.py', ''))
                current_abs_path = os.path.join(current_abs_path, actual_name)
            else:
                # Fallback to the target name if not found strictly, might fail at import
                resolved_parts.append(target)
                current_abs_path = os.path.join(current_abs_path, target)

        module_path = f"{cls._root_module}.{'.'.join(resolved_parts)}"
        return module_path

    @classmethod
    def _find_actual_name(cls, base_path, target, is_last=False):
        """
        Finds the actual filename/directory on disk that best matches the target.
        Handles singular/plural, underscores, and typos using difflib.
        """
        if not os.path.exists(base_path):
            return None
            
        entries = os.listdir(base_path)
        
        # 1. Exact match
        for entry in entries:
            name = entry.replace('.py', '') if is_last else entry
            if name == target:
                return entry
        
        # 2. Case-insensitive match
        target_lower = target.lower()
        for entry in entries:
            name = entry.replace('.py', '').lower() if is_last else entry.lower()
            if name == target_lower:
                return entry

        # 3. Robust Fuzzy Match using difflib
        import difflib
        
        # Prepare candidates (strip .py for comparison if looking for a module)
        candidate_map = {}
        for entry in entries:
            name = entry.replace('.py', '') if is_last else entry
            candidate_map[name] = entry
            
        matches = difflib.get_close_matches(target, candidate_map.keys(), n=1, cutoff=0.6)
        if matches:
            return candidate_map[matches[0]]

        return None

    def _resolve_activation(self, pattern: str):
        """
        Resolves a string to the (module_path, class_name) tuple.
        Returns the exact match if found, otherwise searches using regex.
        """
        # 1. Exact match
        if pattern in Activation._name_to_info:
            return Activation._name_to_info[pattern]

        # 1b. Case-insensitive exact match on registry class names (e.g. "gelu" -> "GELUActivation").
        # Prefer longer names first: __registry__.json lists "GELU" next to "GELUActivation" but
        # the short alias may not exist as a Python class in the target module.
        pl = pattern.lower()
        candidates = [name for name in Activation._name_to_info if name.lower() == pl]
        if candidates:
            candidates.sort(key=lambda n: (-len(n), n))
            return Activation._name_to_info[candidates[0]]

        # 2. Regex search
        try:
            regex = re.compile(pattern)
            matches = [name for name in Activation._name_to_info.keys() if regex.search(name)]
            
            if matches:
                # If multiple matches, prioritize exact or shortest match
                matches.sort(key=lambda x: (len(x), x))
                selected = matches[0]
                logger.info(f"Regex '{pattern}' matched multiple activations, selected '{selected}'. Matches: {matches}")
                return Activation._name_to_info[selected]
        except re.error:
            pass
            
        return None

    @classmethod
    def get_available_activations(cls, pattern=None):
        """
        Returns a list of all available activation class names, optionally filtered by regex.
        """
        if cls._registry_data is None:
            cls._initialize_registry()
            

class ExpressionParser:
    """
    Centralized parser for arithmetic expressions in activation functions.
    Handles loading operations from __ops__.json and transpiling user-friendly expressions
    into executable Python/Torch code.
    """
    _ops_map = None
    _ops_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '__ops__.json')

    def __init__(self):
        if ExpressionParser._ops_map is None:
            self._load_ops()

    @classmethod
    def _load_ops(cls):
        if not os.path.exists(cls._ops_file):
            logger.warning(f"Ops file not found at {cls._ops_file}. Using minimal default.")
            cls._ops_map = {}
            return

        try:
            with open(cls._ops_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    cls._ops_map = data[0]
                elif isinstance(data, dict):
                    cls._ops_map = data
                else:
                    cls._ops_map = {}
        except Exception as e:
            logger.error(f"Failed to load ops file: {e}")
            cls._ops_map = {}

    def get_executable_context(self) -> Dict[str, Any]:
        """
        Returns a dictionary of safe operations to be used in eval().
        """
        eps = 1e-8
        safe_div = lambda n, d: n / (d + eps)

        # Base operations
        ops = {
            'add': torch.add, 'subtract': torch.sub, 'multiply': torch.mul, 'divide': torch.div,
            'sin': torch.sin, 'cos': torch.cos, 'tan': torch.tan,
            'asin': torch.asin, 'acos': torch.acos, 'atan': torch.atan,
            'sinh': torch.sinh, 'cosh': torch.cosh, 'tanh': torch.tanh,
            'asinh': torch.asinh, 'acosh': torch.acosh, 'atanh': torch.atanh,
            'exp': torch.exp, 'log': torch.log, 'log2': torch.log2, 'log10': torch.log10,
            'sqrt': torch.sqrt, 'abs': torch.abs, 'sign': torch.sign, 'sgn': torch.sign,
            'pow': torch.pow, 'clamp': torch.clamp,
            'erf': torch.erf, 'erfc': torch.erfc, 
            'ceil': torch.ceil, 'floor': torch.floor, 'round': torch.round, 'trunc': torch.trunc,
            'sigmoid': torch.sigmoid,
            'real': torch.real, 'imag': torch.imag, 'conj': torch.conj, 'angle': torch.angle,
        }
        
        # Conditionally add gamma/lgamma
        if hasattr(torch, 'gamma'):
            ops['gamma'] = torch.gamma
        elif hasattr(torch, 'special') and hasattr(torch.special, 'gamma'):
            ops['gamma'] = torch.special.gamma
            
        if hasattr(torch, 'lgamma'):
            ops['lgamma'] = torch.lgamma
        elif hasattr(torch, 'special') and hasattr(torch.special, 'lgamma'):
            # lgamma is often just lgamma in special too, or multigammaln etc. 
            # We stick to base if possible or special.gammanln (log gamma)
            if hasattr(torch.special, 'gammaln'):
                ops['lgamma'] = torch.special.gammaln

        
        # Extended derived operations for robustness
        extended_ops = {
            'cot': lambda z: safe_div(torch.tensor(1.0, device=z.device), torch.tan(z)),
            'sec': lambda z: safe_div(torch.tensor(1.0, device=z.device), torch.cos(z)),
            'cosec': lambda z: safe_div(torch.tensor(1.0, device=z.device), torch.sin(z)),
            'acot': lambda z: torch.atan(safe_div(torch.tensor(1.0, device=z.device), z)),
            'asec': lambda z: torch.acos(safe_div(torch.tensor(1.0, device=z.device), z)),
            'acosec': lambda z: torch.asin(safe_div(torch.tensor(1.0, device=z.device), z)),
            'coth': lambda z: safe_div(torch.tensor(1.0, device=z.device), torch.tanh(z)),
            'sech': lambda z: safe_div(torch.tensor(1.0, device=z.device), torch.cosh(z)),
            'cosech': lambda z: safe_div(torch.tensor(1.0, device=z.device), torch.sinh(z)),
            'acoth': lambda z: torch.atanh(safe_div(torch.tensor(1.0, device=z.device), z)),
            'asech': lambda z: torch.acosh(safe_div(torch.tensor(1.0, device=z.device), z)),
            'acosech': lambda z: torch.asinh(safe_div(torch.tensor(1.0, device=z.device), z)),
            'exp2': torch.exp2,
            'expm1': torch.expm1,
            'rsqrt': torch.rsqrt
        }
        ops.update(extended_ops)
        return ops

    def transpile(self, expression: str, num_acts: int, extra_ops: List[str] = None) -> str:
        """
        Transpiles a user string into Python code using the loaded ops map.
        """
        # 1. Handle double pipe notation ||x|| -> abs(x)
        while '||' in expression:
            new_expression = re.sub(r'\|\|([^|]+)\|\|', r'abs(\1)', expression)
            if new_expression == expression: break
            expression = new_expression

        # 2. Replace known operations
        # Sort ops by length to avoid partial matches
        ops_keys = list(self.get_executable_context().keys())
        if extra_ops:
            ops_keys.extend(extra_ops)
        # Also include keys from JSON that might map to something else, though mainly we need keys matching our context
        # But to be safe, let's use the context keys as the source of truth for "functions available"
        sorted_ops = sorted(ops_keys, key=len, reverse=True)
        
        for op in sorted_ops:
            # We want to match whole words only, enabling function calls like sin(...) -> ops['sin'](...)
            pattern = rf'\b{re.escape(op)}\b'
            expression = re.sub(pattern, f"ops['{op}']", expression)

        # 3. Handle activations A1, A2...
        def replace_act(match):
            idx = int(match.group(1)) - 1
            if idx < 0: return match.group(0) # Should not happen with regex \d+
            if idx >= num_acts:
                raise ValueError(f"Expression requests A{idx+1}, but only {num_acts} activation functions provided.")
            return f"acts[{idx}]"
        
        expression = re.sub(r'\bA(\d+)\b', replace_act, expression)

        # 4. Handle Inputs x1, x2...
        def replace_x(match):
            idx = int(match.group(1)) - 1
            return f"feats[{idx}]"
        
        expression = re.sub(r'\bx(\d+)\b', replace_x, expression)
        
        # 5. Handle raw input x
        expression = re.sub(r'\bx\b', "raw_x", expression)

        # 6. Handle parameters w1, b1...
        expression = re.sub(r'\bw(\d+)\b', r"custom_params['w\1']", expression)
        expression = re.sub(r'\bb(\d+)\b', r"custom_params['b\1']", expression)

        return expression
