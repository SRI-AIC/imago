import math

"""
Computation of convolution and encoding schedules.

This will be dynamic, but for now we'll have to index by height
and width of the input field.
"""

class ConvInfo(object):
    def __init__(self, num_in, num_out, kernel_size=4, stride=2, padding=0, dilation=1):
        self.num_in, self.num_out = num_in, num_out
        self.kernel_size, self.stride, self.padding, self.dilation = kernel_size, stride, padding, dilation 

def get_encoder_schedules(num_inputs, num_hidden, height, width):
    shape = (height, width)
    return {
    (12, 12): [ ConvInfo(num_inputs, num_hidden, kernel_size=4, stride=1),
               ConvInfo(num_hidden, num_hidden, kernel_size=4, stride=1),
               ConvInfo(num_hidden, num_hidden, kernel_size=4, stride=1),
               ConvInfo(num_hidden, num_hidden, kernel_size=2, stride=1) ],

    (64, 64): [ ConvInfo(num_inputs, num_hidden, kernel_size=4, stride=2),
               ConvInfo(num_hidden, num_hidden, kernel_size=4, stride=2),
               ConvInfo(num_hidden, num_hidden, kernel_size=4, stride=2),
               ConvInfo(num_hidden, num_hidden, kernel_size=4, stride=2) ],

        # MLP domains (e.g., Cartpole)
        (1, 1): [
            ConvInfo(num_inputs, num_hidden, kernel_size=1, stride=1),
            ConvInfo(num_hidden, num_hidden, kernel_size=1, stride=1),
        ]
    }[shape]

# Schedules for constructing the hierarchical Z encoding
# Note: Only up to n-1 latent z encoders, as the last convolutional layer already
# goes to computing the embedding.  Note this must match the encoder schedule
# The num_in should correspond to channel inputs from the respective encoder layer.
# Currently these are equal, but future setup will allow change this.
def get_z_schedules(num_in, num_out, height, width):
    shape = (height, width)
    return {(12, 12): [ ConvInfo(num_in, num_out, kernel_size=4, stride=1, padding=0, dilation=1),
                ConvInfo(num_in, num_out, kernel_size=4, stride=1, padding=0, dilation=1),
                ConvInfo(num_in, num_out, kernel_size=2, stride=1, padding=0, dilation=1)],
                
    (64, 64): [ ConvInfo(num_in, num_out, kernel_size=4, stride=2, padding=0, dilation=1),
                ConvInfo(num_in, num_out, kernel_size=4, stride=2, padding=0, dilation=1),
                ConvInfo(num_in, num_out, kernel_size=4, stride=2, padding=0, dilation=1)],

    (1,1): [
        ConvInfo(num_in, num_out, kernel_size=1, stride=1, padding=0),
        ConvInfo(num_in, num_out, kernel_size=1, stride=1, padding=0)
    ]
    }[shape]


def get_decoder_schedules(height, width, n_targets,
                          n_latent=256, n_hidden=32):
    shape = (height, width)
    return {
        (64, 64): [
            ConvInfo(n_latent, n_hidden * 4, kernel_size=4, stride=2),
            ConvInfo(n_hidden * 4, n_hidden * 2, kernel_size=4, stride=2),
            ConvInfo(n_hidden * 2, n_hidden, kernel_size=4, stride=2),
            ConvInfo(n_hidden, n_targets, kernel_size=6, stride=2)
            ],

        (12, 12): [
            ConvInfo(n_latent, n_hidden * 4, kernel_size=4, stride=1),
            ConvInfo(n_hidden * 4, n_hidden * 2, kernel_size=4, stride=1),
            ConvInfo(n_hidden * 2, n_hidden, kernel_size=4, stride=1),
            ConvInfo(n_hidden, n_targets, kernel_size=2, stride=1)
            ],
        (1, 1): [
            ConvInfo(n_latent, n_hidden, kernel_size=1, stride=1),
            ConvInfo(n_hidden, n_targets, kernel_size=1, stride=1)
        ]
        }[shape]


"""
Used for computing convolutions
"""


def conv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

def convtransp_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + kernel_size[0] + pad[0]
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + kernel_size[1] + pad[1]

    return h, w


def print_64x64():
    """ Convenience routine, printing out 64x64 schedule 
    """
    # 64x64, used by PySc2
    curr = (64, 64)
    for i in range(4):
        next = conv_output_shape(curr, kernel_size=4, stride=2, padding=0, dilation=1)
        print("{}: {}->{}".format(i, curr, next))
        curr = next


def print_12x12():
    """ Convenience routine, for printing out 12x12 schedule
    """
    # 12x12, used by Canniballs
    curr = (12, 12)
    for i in range(3):
        next = conv_output_shape(curr, kernel_size=4, stride=1, padding=0, dilation=1)
        print("{}: {}->{}".format(i, curr, next))
        curr = next
    next = conv_output_shape(curr, kernel_size=2, stride=1, padding=0, dilation=1)
    print("{}: {}->{}".format(i, curr, next))
