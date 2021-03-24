import functools
import numpy as np
import tensorflow as tf


def _with(**kwargs):
    def _with_decorator(f):
        for (key, value) in kwargs.items():
            values = getattr(f, key, [])
            values.insert(0, value)
            setattr(f, key, values)

        return f
    return _with_decorator


def with_input(name, dtype=None, default=None):
    return _with(
        input_names=name,
        input_types=dtype,
        input_defaults=default)


def with_output(name, dtype=None, shape=None):
    return _with(
        output_names=name,
        output_types=dtype,
        output_shapes=shape)


def _as_func(is_numpy=None, is_tf=None):
    def _as_func_decorator(f, include_inputs=False, *args, **kwargs):
        f_partial = functools.partial(f, *args, **kwargs)

        def _f(inputs):
            input_list = []
            for (name, dtype, default) in zip(f.input_names, f.input_types, f.input_defaults):
                if name in inputs:
                    input_ = inputs[name]
                elif is_numpy:
                    input_ = np.asarray(default)
                elif is_tf:
                    input_ = tf.constant(default, dtype=dtype)

                input_list.append(input_)

            if is_numpy:
                output_list = f_partial(*input_list)
            elif is_tf:
                output_list = tf.numpy_function(f_partial, input_list, f.output_types)

            if include_inputs:
                outputs = inputs
            else:
                outputs = {}

            for (output, name, shape) in zip(output_list, f.output_names, f.output_shapes):
                if is_numpy:
                    outputs[name] = output
                elif is_tf:
                    outputs[name] = tf.ensure_shape(output, shape)

            return outputs
        return _f
    return _as_func_decorator


as_numpy_func = _as_func(is_numpy=True)
as_tensorflow_func = _as_func(is_tf=True)
