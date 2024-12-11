from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    # Reshape the input tensor
    reshaped_input = (
        input.contiguous()
        .view(
            batch, channel, height // kh, kh, width // kw, kw
        )  # Split height and width into blocks
        .permute(0, 1, 2, 4, 3, 5)  # Rearrange axes to bring kernel dimensions together
    )

    # Combine kernel dimensions into a single axis
    reshaped_input = reshaped_input.contiguous().view(
        batch, channel, height // kh, width // kw, kh * kw
    )

    # Compute new height and width for the reshaped tensor
    new_height = height // kh
    new_width = width // kw

    return reshaped_input, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D average pooling on an input tensor.

    Args:
    ----
        input: Tensor of size batch x channel x height x width.
        kernel: Tuple (kernel_height, kernel_width) specifying pooling size.

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width after pooling.
    """
    batch, channel, height, width = input.shape
    input, new_height, new_width = tile(input, kernel)
    return input.mean(4).view(batch, channel, new_height, new_width)


reduce_max = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Find the indices of the maximum values along a dimension.

    Args:
    ----
        input: Tensor to compute the argmax on.
        dim: Dimension to reduce.

    Returns:
    -------
        Tensor with boolean mask where maximum values are `True`.
    """
    out = reduce_max(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for the max"""
        ctx.save_for_backward(input, dim)
        return reduce_max(input, int(dim.item()))

    # int(dim[0])
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the max"""
        input, dim = ctx.saved_values
        return grad_output * argmax(input, int(dim.item())), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction along a dimension."""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor."""
    return input.exp() / input.exp().sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor."""
    return input - (input - max(input, dim)).exp().sum(dim).log() - max(input, dim)


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D max pooling on an input tensor."""
    batch, channel, height, width = input.shape
    out, new_height, new_width = tile(input, kernel)
    out = max(out, len(out.shape) - 1)
    return out.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off."""
    if ignore:
        return input
    else:
        return input * (rand(input.shape) > rate)
