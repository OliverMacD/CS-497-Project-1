import torch  # https://pytorch.org
import torchvision  # https://pytorch.org

from typing import List, Any, Tuple, Optional, Union, Sequence
from jaxtyping import Array, Float, Int, PyTree

import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx

import optax as opt

import matplotlib.pyplot as plt

from functools import partial

import MNV2model as MNV # Model

# Training hyperparameters
LEARNING_RATE = 1e-3
N_EPOCHS = 300
BATCH_SIZE = 32
PRINT_EVERY = 30
SEED = 42

# Key generation
key = jax.random.PRNGKey(SEED)

##############################################################

# Lets test with MNIST

# Load the MNIST dataset [reference](https://docs.kidger.site/equinox/examples/mnist/#the-dataset)
normalise_data = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_dataset = torchvision.datasets.MNIST(
    "MNIST",
    train=True,
    download=True,
    transform=normalise_data,
)
test_dataset = torchvision.datasets.MNIST(
    "MNIST",
    train=False,
    download=True,
    transform=normalise_data,
)
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)

# we aren't using a validation set here, but that's easy enough to fix

# Checking our data a bit (by now, everyone knows what the MNIST dataset looks like)
dummy_x, dummy_y = next(iter(trainloader))
dummy_x = dummy_x.numpy()
dummy_y = dummy_y.numpy()
print(dummy_x.shape)  # 64x1x28x28
print(dummy_y.shape)  # 64
print(dummy_y)

##########################################################################################

model, state = eqx.nn.make_with_state(MNV.MobileNetV2)(in_channels=1, num_classes=10, key=key)
#print(model)

##########################################################################################

# For MobileNetV2

def loss(
    model: MNV.MobileNetV2, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    # Our input has the shape (BATCH_SIZE, 1, 28, 28), but our model operations on
    # a single input input image of shape (1, 28, 28).
    #
    # Therefore, we have to use jax.vmap, which in this case maps our model over the
    # leading (batch) axis.
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)


def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    # y are the true targets, and should be integers 0-9.
    # pred_y are the log-softmax'd predictions.
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


# Example loss
loss_value = loss(model, dummy_x, dummy_y)
print(loss_value.shape)  # scalar loss
# Example inference
output = jax.vmap(model)(dummy_x)
print(output.shape)  # batch of predictions

