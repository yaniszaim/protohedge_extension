"""
Deep Hedging Base (PyTorch version)
-----------------------------------
Replacement for TensorFlow utilities in base.py
"""

from cdxbasics.logger import Logger
from cdxbasics.config import Config, Int, Float  # NOQA
from cdxbasics.prettydict import PrettyOrderedDict as pdct  # NOQA
from cdxbasics.util import isAtomic

from collections.abc import Mapping
import numpy as np
import torch
import math
import inspect
import datetime

_log = Logger(__file__)

# -------------------------------------------------
# Torch setup
# -------------------------------------------------

TORCH_VERSION = torch.__version__

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_GPU = torch.cuda.device_count()
NUM_CPU = 1  # torch does not expose this the same way TF does
TORCH_ENVIRONMENT_MESSAGE = (
    f"PyTorch version {TORCH_VERSION} running on "
    f"{NUM_CPU} CPUs and {NUM_GPU} GPUs"
)

dh_dtype = torch.float32


"""
DIM_DUMMY
Used to infer batch dimension size in some layers
"""
DIM_DUMMY = "_dimension_dummy"


# -------------------------------------------------
# Torch <--> Numpy
# -------------------------------------------------

import torch
import numpy as np
from collections.abc import Mapping

def torchCast(x, dtype=torch.float32, device="cpu", native=False):

    # already torch tensor
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)

    # numpy array
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=dtype, device=device)

    # scalar
    if isinstance(x, (float, int)):
        return torch.tensor(x, dtype=dtype, device=device)

    # dict
    if isinstance(x, Mapping):
        d = {
            k: torchCast(v, dtype=dtype, device=device)
            for k, v in x.items()
        }
        return d if native or type(x) == dict else x.__class__(d)

    # list
    if isinstance(x, list):
        return [
            torchCast(v, dtype=dtype, device=device)
            for v in x
        ]

    # tuple
    if isinstance(x, tuple):
        return tuple(
            torchCast(v, dtype=dtype, device=device)
            for v in x
        )

    raise TypeError(
        f"Cannot convert type {type(x)} to torch tensor"
    )


def torch_dict(dtype=dh_dtype, **kwargs):
    """
    dictionary of tensors
    """
    return torchCast(kwargs, dtype=dtype)


def npCast(x, dtype=None):
    """
    recursively convert torch -> numpy
    """

    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype)

    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=dtype)

    if isAtomic(x):
        return np.array(x, dtype=dtype)

    if isinstance(x, dict):
        d = {k: npCast(x[k], dtype=dtype) for k in x}

        return d if type(x) == dict else x.__class__(d)

    if isinstance(x, list):
        l = [npCast(v, dtype=dtype) for v in x]

        return l if type(l) == list else x.__class__(l)

    return np.asarray(x, dtype=dtype)


# -------------------------------------------------
# initialization helpers
# -------------------------------------------------

def torch_glorot_value(shape):
    """
    equivalent of tf.keras.initializers.GlorotUniform
    """

    fan_in = shape[0]

    limit = math.sqrt(6 / fan_in)

    x = torch.empty(shape, dtype=dh_dtype)

    torch.nn.init.uniform_(x, -limit, limit)

    return x


# -------------------------------------------------
# tensor reshaping utilities
# -------------------------------------------------

def torch_back_flatten(tensor, dim):

    if tensor.ndim <= dim:
        return tensor

    new_shape = list(tensor.shape[:dim-1]) + [-1]

    return tensor.reshape(new_shape)


def torch_make_dim(tensor, dim):

    if tensor.ndim > dim:

        return torch_back_flatten(tensor, dim)

    while tensor.ndim < dim:

        tensor = tensor.unsqueeze(-1)

    return tensor


# -------------------------------------------------
# numpy utilities (unchanged)
# -------------------------------------------------

def np_unique_tol(x, tol=1e-8, is_sorted=False):

    x = np.sort(x) if not is_sorted else np.asarray(x)

    dx = x[1:] - x[:-1]

    ixs = np.full((len(x),), True, dtype=np.bool_)

    ixs[:-1] = dx > tol

    x = x[ixs]

    assert len(x) > 0

    return x


# -------------------------------------------------
# weighted statistics
# -------------------------------------------------

def mean(P, w, axis=None):

    assert P.shape == w.shape

    return np.sum(P * w, axis=axis)


def var(P, w, axis=None):

    m = mean(P, w, axis)

    return np.sum(P * ((w - m) ** 2), axis=axis)


def std(P, w, axis=None):

    return math.sqrt(var(P, w, axis))


def err(P, w, axis=None):

    return std(P, w, axis) / math.sqrt(float(P.shape[0]))


# -------------------------------------------------
# validation utilities
# -------------------------------------------------

def assert_iter_not_is_nan(d, name=""):

    for k in d:

        v = d[k]

        n = name + "." + k if name else k

        if isinstance(v, Mapping):

            assert_iter_not_is_nan(v, n)

        else:

            assert not np.isnan(v).any(), f"NaN detected in {n}"


# -------------------------------------------------
# formatting utilities
# -------------------------------------------------

def fmt_now():

    return datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# -------------------------------------------------
# optimizer factory (simplified)
# -------------------------------------------------

def create_optimizer(train_config):

    lr = train_config("learning_rate", 1e-3)

    name = train_config("optimizer", "adam")

    if name.lower() == "adam":

        return torch.optim.Adam

    if name.lower() == "sgd":

        return torch.optim.SGD

    raise ValueError(f"Unknown optimizer {name}")
