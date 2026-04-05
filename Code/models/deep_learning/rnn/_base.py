"""
Shared base class for RNN modules that require argument expansion across layers.
"""
from ....models.utils import DLModule


class _BaseRNNModule(DLModule):
    """
    Helper base class for handling argument expansion for multi-layer RNNs.
    """
    def _expand_arg(self, arg, layers, bidirectional, is_container=False):
        target_count = layers * (2 if bidirectional else 1)
        if isinstance(arg, str):
             print(f"DEBUG: arg is string: {arg}")
             return [arg] * target_count
        print(f"DEBUG: arg is NOT string: {arg} type={type(arg)}")

        if isinstance(arg, (list, tuple)):
            # If user provided a list that matches target count, use it.
            if len(arg) == target_count:
                # If is_container is True, check if elements are lists/tuples?
                # User requirement: "if the cell have some input arguments in 1D list... their corresponding modules must have 2D list"
                # If arg is [[...], [...]] it matches.
                return arg

            # If length mismatch or if it looks like a single config (1D list) that needs to be replicated?
            # E.g. funcs = ['tanh', 'sigmoid'] for a cell.
            # Module arg should be [['tanh', 'sigmoid'], ['tanh', 'sigmoid']] if 2 layers.
            #
            if is_container:
                # If the outer list doesn't denote layers, but is the config itself?
                # We assume if len != target_count, it might be a single config.
                # However, if len == target_count, it's ambiguous if target_count == len(config).
                # Robustness: Check element type.
                if len(arg) > 0 and isinstance(arg[0], (list, tuple)):
                     # It is already 2D. But len mismatch?
                     # We cycle/extend.
                     pass
                else:
                    # It is 1D. Replicate it to make 2D.
                    return [arg] * target_count

            # Fallback for simple args or already correct args
            if len(arg) == 1:
                return [arg[0]] * target_count

            # Cycle/Extend
            out = []
            for i in range(target_count):
                out.append(arg[i % len(arg)])
            return out
        else:
            return [arg] * target_count

    def _init_layer_args(self, input_size, hidden_size, num_layers, directions, proj_size=0):
        layer_configs = []
        for i in range(num_layers):
            if i == 0:
                l_in = input_size
            else:
                prev_out = proj_size if proj_size is not None and proj_size > 0 else hidden_size
                l_in = prev_out * directions
            layer_configs.append(l_in)
        return layer_configs
