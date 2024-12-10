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
    """Tiled average pooling 2D

    Args:
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
    -------
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")
    input, new_height, new_width = tile(input, kernel)
    return input.mean(4).view(
        batch, channel, new_height, new_width
    )  # 4 comes from len(out_shape)
    
def argmax(input: Tensor, dim: int) -> Tensor:
    pass
    
