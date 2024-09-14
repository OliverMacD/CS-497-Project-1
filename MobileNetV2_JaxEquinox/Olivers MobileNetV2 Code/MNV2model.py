from typing import List, Any, Tuple, Optional, Union, Sequence
from jaxtyping import Array, Float, Int, PyTree

import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx

import optax as opt

import matplotlib.pyplot as plt

from functools import partial

#####################################################

# Define a Depthwise Separable Convolution Layer
class DepthwiseSeparableConv(eqx.Module):
    depthwise: eqx.nn.Conv2d
    pointwise: eqx.nn.Conv2d

    def __init__(self, in_channels, out_channels, stride, key):
        dw_key, pw_key = jax.random.split(key)
        self.depthwise = eqx.nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=(3, 3), 
            stride=stride,  
            padding = 1,
            groups=in_channels,
            key=dw_key
        )
        self.pointwise = eqx.nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding = 1,
            key=pw_key
        )

    def __call__(self, x):
        x = self.depthwise(x)
        x = jax.nn.relu(x)
        x = self.pointwise(x)
        return x
    
#####################################################

# Define a Bottleneck Block
class Bottleneck(eqx.Module):
    _stride: int = eqx.field(static=True)

    conv1: eqx.nn.Conv2d
    depthwise_conv: DepthwiseSeparableConv
    conv3: eqx.nn.Conv2d
    use_residual: bool

    def __init__(self, in_channels, out_channels, stride, expand_ratio, use_residual, key: jr.PRNGKey):
        self._stride=stride
        keys = jr.split(key, 3)
        hidden_dim = in_channels * expand_ratio
        self.conv1 = eqx.nn.Conv2d(in_channels, hidden_dim, kernel_size=(1, 1), key=keys[0])
        self.depthwise_conv = [DepthwiseSeparableConv(hidden_dim, hidden_dim, stride=1, key=keys[1]),
                               DepthwiseSeparableConv(hidden_dim, hidden_dim, stride=2, key=keys[1])]

        if(stride == 1):
            self.conv3 = eqx.nn.Conv2d(in_channels, in_channels, stride=stride, padding=0, kernel_size=(1, 1), key=keys[2])
        else:
            self.conv3 = eqx.nn.Conv2d(hidden_dim, out_channels, stride=stride, padding=0, kernel_size=(1, 1), key=keys[2])

        self.use_residual = use_residual

    def __call__(self, x):
        residual = x
        x = self.conv1(x)
        x = jax.nn.relu(x)
        if self._stride == 1:
            x = self.depthwise_conv[0](x)
            x = jax.nn.relu(x)
            x = self.conv3(x)

            return x + residual
        else:
            x = self.depthwise_conv[1](x)
            x = jax.nn.relu(x)
            x = self.conv3(x)
            return x
        
#####################################################

# Define the MobileNetV2
class MobileNetV2(eqx.Module):
    in_channels: int = eqx.field(static=True)
    
    first_conv: eqx.nn.Conv2d
    bottlenecks: list
    last_conv: eqx.nn.Conv2d
    pool: eqx.nn.AvgPool2d
    classifier: eqx.nn.Conv2d

    def __init__(self, in_channels, num_classes, key):
        keys = jax.random.split(key, 10)
        self.in_channels = in_channels

        self.first_conv = eqx.nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=2, key=keys[0])

        # Bottleneck blocks configuration
        bottleneck_configs = [
            # (in_channels, out_channels, stride, expand_ratio, n_repeats)
            (32, 16, 1, 1, 1),   # First block, no expansion, 1 repetition
            (16, 24, 2, 6, 2),   # Second block, 2x stride, 3 repetitions
            (24, 32, 2, 6, 3),   # Third block, 2x stride, 4 repetitions
            (32, 64, 2, 6, 4),   # Fourth block, 2x stride, 5 repetitions
            (64, 96, 1, 6, 3),   # Fifth block, stride 1, 4 repetitions
            (96, 160, 2, 6, 3),  # Sixth block, 2x stride, 4 repetitions
            (160, 320, 1, 6, 1), # Seventh block, stride 1, 1 repetition
        ]

        self.bottlenecks = []
        current_key = keys[1]

        for config in bottleneck_configs:
            in_channels, out_channels, stride, expand_ratio, n_repeats = config

            # Add the first block in the stage with the specified stride
            self.bottlenecks.append(
                Bottleneck(in_channels, out_channels, stride, expand_ratio, use_residual=(stride == 1), key=current_key)
            )
            current_key = jax.random.split(current_key, 1)[0]
            print(f'In: {in_channels}, Out: {out_channels}')

            # Add the remaining blocks with stride = 1
            for i in range(n_repeats):
                self.bottlenecks.append(
                    Bottleneck(out_channels, out_channels, stride=1, expand_ratio=expand_ratio, use_residual=True, key=current_key)
                )
                print(f'In: {out_channels}, Out: {out_channels}')
                current_key = jax.random.split(current_key, 1)[0]

        self.last_conv = eqx.nn.Conv2d(24, 1280, kernel_size=(1, 1), key=keys[2])
        self.pool = eqx.nn.AvgPool2d(kernel_size=(7, 7))
        self.classifier = eqx.nn.Conv2d(1280, num_classes, kernel_size=(1,1),key=keys[3])

    def __call__(self, x):
        x = self.first_conv(x)
        x = jax.nn.relu(x)

        for bottleneck in self.bottlenecks:
            x = bottleneck(x)

        x = self.last_conv(x)
        x = jax.nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = self.classifier(x)
        return x
