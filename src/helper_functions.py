


import torch


def calc_dim(height:int, width:int, conv_layer:torch.nn):
    """Calculate the output dimensions for convolutional layers."""
    
    ## Case checking - Sometimes the values can be returned as int instead of a tuple
    kernel_h, kernel_w = (
        (conv_layer.kernel_size, conv_layer.kernel_size) if isinstance(conv_layer.kernel_size, int) 
        else conv_layer.kernel_size
    )
    
    padding_h, padding_w = (
        (conv_layer.padding, conv_layer.padding) if isinstance(conv_layer.padding, int) 
        else conv_layer.padding
    )
    
    stride_h, stride_w = (
        (conv_layer.stride, conv_layer.stride) if isinstance(conv_layer.stride, int) 
        else conv_layer.stride
    )
    
    dilation_h, dilation_w = (
        (conv_layer.dilation, conv_layer.dilation) if isinstance(conv_layer.dilation, int) 
        else conv_layer.dilation
    )
    
    ## Calculate the output dimensions - Src: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
    output_height = int( (height + 2*padding_h - dilation_h*(kernel_h - 1) - 1)/stride_h + 1 )
    output_width = int( (width + 2*padding_w - dilation_w*(kernel_w-1) - 1)/stride_w +1 )
    
    return (output_height, output_width)