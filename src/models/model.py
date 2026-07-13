import torch.nn as nn
from src.helper_functions import calc_dim

class Model(nn.Module):
    def __init__(self, input_channels:int=1, input_height:int=28, input_width:int=28, num_classes:int=10):
        super().__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes

        ### Loss Function ###
        self.loss_fn = nn.CrossEntropyLoss()

        #### Convolutional Layers ###

        # Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=8,
            kernel_size=3,
            padding=1
        )
        self.activation1 = nn.LeakyReLU() # Doesn't change the output dimensions
        out_height, out_width = calc_dim(self.input_height, self.input_width, self.conv1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces the output dimensions by half
        out_height, out_width = calc_dim(out_height, out_width, self.maxpool1)


        # Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.activation2 = nn.LeakyReLU() # Doesn't change the output dimensions
        out_height, out_width = calc_dim(out_height, out_width, self.conv2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces the output dimensions by half
        out_height, out_width = calc_dim(out_height, out_width, self.maxpool2)

        ### Fully Connected Layers ###
        self.flatten = nn.Flatten()
        self.flattened_dim = out_height * out_width * self.conv2.out_channels

        # Clasification Layer 1
        self.dense1 = nn.Linear(
            in_features=self.flattened_dim,
            out_features=128
        )

        # Clasification Layer 2
        self.dense2 = nn.Linear(
            in_features=128,
            out_features=self.num_classes
        )

        

if __name__ == "__main__":
    model = Model()
    print(model)