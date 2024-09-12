from typing import List, Any, Tuple, Optional, Union, Sequence
from jaxtyping import Array, Float, Int, PyTree

import sys

import json

import jax
import jax.numpy as jnp
import jax.random as jr

import torch  # https://pytorch.org
import torchvision  # https://pytorch.org

import equinox as eqx

import optax as opt

import matplotlib.pyplot as plt

from functools import partial

from MobileNetV2_Jax import MobileNetV2, InvertedResidualBlock, DepthwiseConv2D


@eqx.filter_jit
def loss(
    model: MobileNetV2,  state: eqx.nn.State, x: Float[Array, "batch 3 224 224"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    batch_model = jax.vmap(
        model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )
    pred_y, state = batch_model(x, state)
    return cross_entropy(y, pred_y), state

@eqx.filter_jit
def inference_loss(
    model: MobileNetV2, x: Float[Array, "batch 3 224 224"], y: Int[Array, " batch"]
):
    pred_y, _ = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)

def cross_entropy(
    y: Int[Array, "batch"], pred_y: Float[Array, "batch 100"]
) -> Float[Array, ""]:
    # y are the true targets, and should be integers 0-99.
    # pred_y are the log-softmax'd predictions.
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1).squeeze()
    return -jnp.mean(pred_y)

@eqx.filter_jit
def compute_accuracy(
    model: MobileNetV2, state, x: Float[Array, "batch 3 224 224"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    inference_model = eqx.nn.inference_mode(model)
    inference_model = eqx.Partial(inference_model, state=state)

    pred_y, _ = jax.vmap(inference_model)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)

def evaluate(model: MobileNetV2, state, testloader: torch.utils.data.DataLoader):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    inference_model = eqx.nn.inference_mode(model)
    inference_model = eqx.Partial(inference_model, state=state)

    avg_loss = 0
    avg_acc = 0
    x_s = []
    y_s = []
    for x, y in testloader:
        x = x.numpy()
        x_s.append(x)
        y = y.numpy()
        y_s.append(y)
    
    @eqx.debug.assert_max_traces(max_traces=1)
    def body_fn(i, carry):
        # print("compiling body_fn")
        l, a, x, y = carry
        l = l + inference_loss(inference_model, x[i], y[i])
        a = a + compute_accuracy(model, state, x[i], y[i])
        return (l, a, x, y)
    
    if x_s[-1].shape[0] < testloader.batch_size:
        avg_loss, avg_acc, _, _ = jax.lax.fori_loop(0, len(x_s) - 1, body_fn, (0., 0., jnp.array(x_s[:-1]), jnp.array(y_s[:-1])))
        avg_loss += inference_loss(inference_model, x_s[-1], y_s[-1])
        avg_acc += compute_accuracy(model, state, x_s[-1], y_s[-1])
    else:
        avg_loss, avg_acc, _, _ = jax.lax.fori_loop(0, len(x_s), body_fn, (0., 0., jnp.array(x_s), jnp.array(y_s)))
    
    return avg_loss / len(testloader), avg_acc / len(testloader)

def train(
    model: MobileNetV2,
    state: eqx.nn.State,
    optim: Any,
    trainloader: torch.utils.data.DataLoader,
    valloader: torch.utils.data.DataLoader,
    n_epochs: int,
    print_every: int,
) -> Tuple[MobileNetV2, eqx.nn.State, Any]:
    
    # only train parameters, filter out non-arrays and static fields
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def make_step(model, state, opt_state, x, y):
        ls, grads = eqx.filter_value_and_grad(loss, has_aux=True)(model, state, x, y) # loss is already vmap'd, so no need to vmap here
        loss_value, state = ls
        # return model, state, opt_state, loss_value, grads
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, state, opt_state, loss_value
    
    def infiniteTrainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(n_epochs), infiniteTrainloader()):
        
        x = x.numpy()
        y = y.numpy()
        
        model, state, opt_state, loss_value = make_step(model, state, opt_state, x, y)
        if step % print_every == 0 or step == n_epochs - 1:
            val_loss, val_accuracy = evaluate(model, state, valloader)
            print(f"Step {step}, Loss: {loss_value:.5e}")
            print(f"\tVal Loss: {val_loss:.5e}, Val Accuracy: {val_accuracy:.3f}")

    return model, state

if __name__ == '__main__':
    # Training hyperparameters
    LEARNING_RATE = 3e-4
    IN_CHANNELS = 3
    N_CLASSES = 100
    BATCH_SIZE = 32
    N_EPOCHS = 25
    PRINT_EVERY = 5
    SEED = 42
    hyperparams = {
        "learning_rate": LEARNING_RATE,
        "epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "in_channels": IN_CHANNELS,
        "num_classes": N_CLASSES,
        "include_top": True,
        "input_size": (224, 224),
        "seed": SEED
    }

    # Key generation
    key = jax.random.PRNGKey(SEED)

    # Load the MNIST dataset [reference](https://docs.kidger.site/equinox/examples/mnist/#the-dataset)
    print("Loading dataset...")
    process_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR100(
        "CIFAR100",
        train=True,
        download=True,
        transform=process_data,
    )
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [45000, 5000])
    test_dataset = torchvision.datasets.CIFAR100(
        "CIFAR100",
        train=False,
        download=True,
        transform=process_data,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # Initialize the model
    print("Initializing model...")
    model, state = eqx.nn.make_with_state(MobileNetV2)(in_channels=IN_CHANNELS, num_classes=N_CLASSES, key=key, include_top=True, input_size=(224, 224))

    print("Beginning training...\n")
    optim = opt.adam(LEARNING_RATE)
    model, state = train(model, state, optim, trainloader, valloader, N_EPOCHS, PRINT_EVERY)

    test_loss, test_accuracy = evaluate(model, state, testloader) # TODO: Fix evaluate to use vmap or a jax for loop. SLOW
    print(f"\nTest Loss: {test_loss:.5e}, Test Accuracy: {test_accuracy:.3f}")


    def save(filename, hyperparams, model):
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, model)


    save("/bsuhome/christopherdaghe/repos/CS-497-Project-1/MobilNetV2_JaxEquinox/CIFAR100_Models/model.eqx", hyperparams, model)
