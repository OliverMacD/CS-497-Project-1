from typing import List, Any, Tuple, Optional, Union, Sequence
from jaxtyping import Array, Float, Int, PyTree

import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx

import optax as opt

import matplotlib.pyplot as plt

from functools import partial


class DepthwiseConv2D(eqx.Module):
    # [reference](https://github.com/DarshanDeshpande/jax-models/blob/main/jax_models/layers/depthwise_separable_conv.py)

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    kernel_size: Union[int, Sequence[int]] = eqx.field(static=True)
    stride: Union[int, Sequence[int]] = eqx.field(static=True)
    padding: str = eqx.field(static=True)
    depth_multiplier: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    groups: int = eqx.field(static=True)
    key: Any = eqx.field(static=True)

    kernel: Array
    bias: Array

    def __init__(self, in_channels: int, depth_multiplier: int, kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: str, use_bias: bool, key: jr.PRNGKey):
        self.in_channels = in_channels
        self.depth_multiplier = depth_multiplier
        self.out_channels = in_channels * depth_multiplier
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding.upper()
        self.use_bias = False
        self.groups = in_channels
        self.key = key

        # if self.padding.lower() is not "valid" and self.padding is not "same":
        #     raise ValueError("Padding must be either 'valid' or 'same'")

        self.kernel = jr.uniform(
            key,
            shape = (self.in_channels, self.depth_multiplier * self.in_channels, *self.kernel_size)
        )

        if use_bias:
            self.bias = jr.normal(key, shape=(in_channels * depth_multiplier,))
            self.use_bias = True
        else:
            self.bias = jnp.zeros((in_channels * depth_multiplier,))


    def __call__(self, x: Array) -> Array:
        x = jnp.expand_dims(x, axis=0)
        x = jax.lax.conv_general_dilated( # see (https://jax.readthedocs.io/en/latest/notebooks/convolutions.html#dimension-numbers-define-dimensional-layout-for-conv-general-dilated)
            lhs=x,
            rhs=self.kernel,
            window_strides=self.stride,
            padding=self.padding.upper(),
            lhs_dilation=(1,) * len(self.kernel_size),
            rhs_dilation=(1,) * len(self.kernel_size),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            # feature_group_count=x.shape[-1]
            # feature_group_count=1
        )
        x = x.squeeze(axis=0)
        if self.use_bias:
            x = x + self.bias
        return x


class InvertedResidualBlock(eqx.Module):
    # Define an inverted residual block [reference](https://github.com/keras-team/keras/blob/v3.3.3/keras/src/applications/mobilenet_v2.py#L398)

    # static fields get ignored durign training
    in_channels:  int   = eqx.field(static=True)
    expansion:    int   = eqx.field(static=True)
    stride:       int   = eqx.field(static=True)
    alpha:        float = eqx.field(static=True)
    filters:      int   = eqx.field(static=True)
    pw_filters:   int   = eqx.field(static=True)
    block_id:     int   = eqx.field(static=True)

    layers: List[Any]
    

    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def __init__(self, in_channels: int, expansion: int, stride: int, alpha: float, filters: int, block_id: int, key: jr.PRNGKey):
        self.in_channels = in_channels
        self.expansion = expansion
        self.stride = stride
        self.alpha = alpha
        self.filters = filters
        self.block_id = block_id

        pointwise_filters = int(filters * alpha)
        # ensure that the number of filters on the last 1x1 convolution is a multiple of 8
        pointwise_filters = self._make_divisible(pointwise_filters, 8)
        self.pw_filters = pointwise_filters

        # Define the key for the block
        key, conv_key = jr.split(key)
        self.layers = []

        # Define the layers of the block
        if block_id:
            # Expand with a pointwise 1x1 convolution
            self.layers.extend([
                eqx.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels * expansion,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    padding_mode='ZEROS',
                    use_bias=False,
                    key=conv_key
                ),
                eqx.nn.BatchNorm(
                    in_channels * expansion,
                    axis_name='batch',
                    eps=1e-3,
                    momentum=0.99
                ),
                jax.nn.relu6
            ])
        
        self.layers.extend([
            DepthwiseConv2D(
                in_channels=in_channels * expansion,
                depth_multiplier=1,
                kernel_size=(1, 1),
                stride=(stride, stride),
                padding="valid",
                key=conv_key,
                use_bias=False
            ),
            eqx.nn.BatchNorm(
                in_channels * expansion,
                axis_name='batch',
                eps=1e-3,
                momentum=0.99
            ),
            jax.nn.relu6
        ])

        # pointwise 1x1 conv
        self.layers.extend([
            eqx.nn.Conv2d(
                in_channels=in_channels * expansion,
                out_channels=pointwise_filters,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                padding_mode='ZEROS',
                use_bias=False,
                key=conv_key
            ),
            eqx.nn.BatchNorm(
                pointwise_filters,
                axis_name='batch',
                eps=1e-3,
                momentum=0.99
            )
        ])
    
    def __call__(self, x, state):
        input = x

        lc = 0
        
        if self.block_id:
            x = self.layers[0](x)
            x, state = self.layers[1](x, state)
            x = self.layers[2](x)
            lc = 3
        if self.stride == 2:
            correct = (x.shape[0] - (self.in_channels * self.expansion)) // 2
            x = jnp.pad(x, ((correct, correct), (correct, correct), (0, 0)), mode='constant', constant_values=0)

        for _, layer in enumerate(self.layers[lc:]):
            if issubclass(type(layer), eqx.nn.StatefulLayer):
                x, state = layer(x, state)
            else:
                x = layer(x)

        if self.in_channels == self.pw_filters and self.stride == 1:
            x = x + input

        return x, state

class MobileNetV2(eqx.Module):
    # MobileNetV2 model based on the Keras implementation

    layers: List[Any]

    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def __init__(self,
        in_channels: int = 3,
        alpha: float = 1.0,
        include_top: bool = True,
        num_classes: int = 1000,
        classifier_activation: str = 'softmax',
        pooling: Optional[str] = None,
        key: jr.PRNGKey = jr.PRNGKey(0),
        input_size: Sequence[int] = (224, 224),
    ):
        key, conv_key = jr.split(key)
        
        first_block_filters = self._make_divisible(32 * alpha, 8)
        if alpha > 1.0:
            last_block_filters = self._make_divisible(1280 * alpha, 8)
        else:
            last_block_filters = 1280

        first_layer_padding = (
            (input_size[0] - 1) // 2,
            (input_size[1] - 1) // 2
        )

        self.layers = [
            eqx.nn.Conv2d(
                in_channels=in_channels,
                out_channels=first_block_filters,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=first_layer_padding, # equivalent to TF 'same' padding
                padding_mode='ZEROS',
                use_bias=False,
                key=conv_key
            ),
            eqx.nn.BatchNorm(
                input_size=first_block_filters,
                eps=1e-3,
                momentum=0.999,
                axis_name="batch"
            ),
            jax.nn.relu6,
            InvertedResidualBlock(
                in_channels=first_block_filters,
                expansion=1,
                stride=1,
                alpha=alpha,
                filters=16,
                block_id=0,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=16,
                expansion=6,
                stride=2,
                alpha=alpha,
                filters=24,
                block_id=1,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=24,
                expansion=6,
                stride=1,
                alpha=alpha,
                filters=24,
                block_id=2,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=24,
                expansion=6,
                stride=2,
                alpha=alpha,
                filters=32,
                block_id=3,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=32,
                expansion=6,
                stride=1,
                alpha=alpha,
                filters=32,
                block_id=4,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=32,
                expansion=6,
                stride=1,
                alpha=alpha,
                filters=32,
                block_id=5,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=32,
                expansion=6,
                stride=2,
                alpha=alpha,
                filters=64,
                block_id=6,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=64,
                expansion=6,
                stride=1,
                alpha=alpha,
                filters=64,
                block_id=7,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=64,
                expansion=6,
                stride=1,
                alpha=alpha,
                filters=64,
                block_id=8,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=64,
                expansion=6,
                stride=1,
                alpha=alpha,
                filters=64,
                block_id=9,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=64,
                expansion=6,
                stride=1,
                alpha=alpha,
                filters=96,
                block_id=10,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=96,
                expansion=6,
                stride=1,
                alpha=alpha,
                filters=96,
                block_id=11,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=96,
                expansion=6,
                stride=1,
                alpha=alpha,
                filters=96,
                block_id=12,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=96,
                expansion=6,
                stride=2,
                alpha=alpha,
                filters=160,
                block_id=13,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=160,
                expansion=6,
                stride=1,
                alpha=alpha,
                filters=160,
                block_id=14,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=160,
                expansion=6,
                stride=1,
                alpha=alpha,
                filters=160,
                block_id=15,
                key=key
            ),
            InvertedResidualBlock(
                in_channels=160,
                expansion=6,
                stride=1,
                alpha=alpha,
                filters=320,
                block_id=16,
                key=key
            ),
            eqx.nn.Conv2d(
                in_channels=320,
                out_channels=last_block_filters,
                kernel_size=(1, 1),
                use_bias=False,
                key=conv_key
            ),
            eqx.nn.BatchNorm(
                input_size=last_block_filters,
                eps=1e-3,
                momentum=0.999,
                axis_name="batch"
            ),
            jax.nn.relu6
        ]

        if include_top:
            self.layers.extend([
                eqx.nn.AvgPool2d(kernel_size=(7, 7)), # TODO: replace with global average pooling
                jnp.ravel,
                eqx.nn.Linear(
                    in_features=81920,
                    out_features=num_classes,
                    key=key
                )
            ])
            if classifier_activation == 'softmax':
                self.layers.append(jax.nn.log_softmax)

        else:
            if pooling == 'avg':
                self.layers.append(eqx.nn.AvgPool2d(kernel_size=(7, 7))) # TODO: replace with global average pooling
            elif pooling == 'max':
                self.layers.append(eqx.nn.MaxPool2d(kernel_size=(7, 7))) # TODO: replace with global max pooling

    def __call__(self, x, state):
        for layer in self.layers:
            if issubclass(type(layer), eqx.nn.StatefulLayer) or isinstance(layer, InvertedResidualBlock):
                x, state = layer(x, state)
            else:
                x = layer(x)
        return x, state
