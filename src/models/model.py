import torch
import torch.nn as nn
import torchinfo
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
        self.activation3 = nn.LeakyReLU()  # Doesn't change the output dimensions

        # Clasification Layer 2
        self.dense2 = nn.Linear(
            in_features=128,
            out_features=self.num_classes
        )

    def forward(self, x):
        # Convolutional Layers
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.activation2(x)
        x = self.maxpool2(x)

        # Fully Connected Layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.activation3(x)
        x = self.dense2(x)

        return x


    def get_loss_function(self, **kwargs):
        """Return the loss function used by this network"""
        return self._loss_function(**kwargs)
    

    def show_summary(self):
        """Show the model summary information."""
        return torchinfo.summary(
            self, 
            input_size=(self.input_channels, self.input_height, self.input_width),
            batch_dim=0,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu").type,
            verbose=0) 


if __name__ == "__main__":

    # Test the model summary
    temp_model = Model()
    print(temp_model.show_summary())