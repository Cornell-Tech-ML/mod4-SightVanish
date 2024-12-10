from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


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
    # TODO: Implement for Task 4.3.
    
    input = (
        input.contiguous()
        .view(batch, channel, height, int(width / kw), kw)
        .permute(0, 1, 3, 2, 4)
    )
    input = input.contiguous().view(
        batch, channel, int(width / kw), int(height / kh), kh * kw
    )
    return (input, int(height / kh), int(width / kw))


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
    out = FastOps.reduce(operators.max, float('-inf'))(input, dim)
    return out == input
    
class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        # TODO: Implement for Task 4.4.
        # raise NotImplementedError("Need to implement for Task 4.4")
        ctx.save_for_backward(input, int(dim.item()))
        return FastOps.reduce(operators.max, float('-inf'))(input, int(dim.item()))

    # int(dim[0])
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        # TODO: Implement for Task 4.4.
        # raise NotImplementedError("Need to implement for Task 4.4")
        input, dim = ctx.saved_values
        return grad_output * argmax(input, dim), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))

def softmax(input: Tensor, dim: int) -> Tensor:
    return input.exp() / input.exp().sum(dim=dim)

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    return input - (input - max(input, dim)).exp().sum(dim).log() - max(input, dim)

def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    batch, channel, height, width = input.shape
    out, new_height, new_width = tile(input, kernel)
    out = max(out, len(out.shape) - 1)
    return out.view(batch, channel, new_height, new_width)

def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    if ignore:
        return input
    else:
        return input * (rand(input.shape) > rate)