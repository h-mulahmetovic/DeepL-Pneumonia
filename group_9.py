import torch.nn as nn

input_dim=(256,256) # set the input dimension of the images to 256x256
channel_dim=1 # 1 for greyscale, 3 for RGB

class group_9(nn.Module):
    """
    A convolutional neural network (CNN) model for binary classification.

    This class defines a neural network designed for classifying grayscale chest x-ray images 
    into two categories: healthy and pneumatic. It consists of two convolutional layers followed 
    by a series of fully connected layers. The network includes dropout layers for regularization 
    and ReLU activations for non-linearity.

    Pneumonia causes inflammation of the lungs' air sacs, these air sacs get filled up with fluid. 
    This fluid is denser than the air in the lungs, because of this it absorbs more x-rays. Since
    the fluid absorbs the x-rays, they do not reach the detector, which makes them show up as white spots.
    We want the model to detect these opaque spots in the lungs, they will be detected by the convolutional 
    layers, we have multiple layers with different kernel sizes, this allows the model to pick up on both
    small and larger scale patterns in the lungs. After the convolutional layers we flatten the data and 
    feed it into 3 hidden layers. We have chosen to use 3 hidden layers since this is the maximum amount of
    layers that we can accurately train given the size of our dataset. 
    
    We use dropout layers between the hidden
    layers in order to avoid overfitting, the dropout rate is set to 0.3, this value has been chosen by experimentation
    and review of the results. 
    
    The input dimensions of the model are 256x256, on this image size, features
    are still distinguishable enough that the model can classify them, increasing the image size would
    lead to longer training times, so this value has been chosen in a trade-off between accuracy and efficiency.

    Architecture:
    - Convolutional Layer (conv1): Applies 32 convolutional filters of size 5x5 to the input.
    - Convolutional Layer (conv2): Applies 64 convolutional filters of size 3x3.
    - Max Pooling Layer (maxpool1): Reduces spatial dimensions by pooling with a 2x2 kernel.
    - Fully Connected Layer (fc1): Transforms the flattened output to 512 units.
    - Fully Connected Layer (fc2): Further transforms the data to 128 units.
    - Fully Connected Layer (fc3): Reduces the data to 64 units.
    - Fully Connected Layer (fc4): Outputs 2 units representing the final class scores.
    - Dropout Layers: Applied after each fully connected layer to prevent overfitting.
    - ReLU Activations: Introduce non-linearity after each convolutional and fully connected layer.
    - Softmax Activation: Outputs probability distribution over the two classes.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        maxpool1 (nn.MaxPool2d): The max pooling layer.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        fc3 (nn.Linear): The third fully connected layer.
        fc4 (nn.Linear): The final fully connected layer.
        dropout (nn.Dropout): Dropout layer for regularization.
        relu (nn.ReLU): ReLU activation function.
        softmax (nn.Softmax): Softmax activation function for output probabilities.
    """

    def __init__(self):
        """
        Initializes the network layers and components.
        
        Sets up the convolutional layers, max pooling, fully connected layers, dropout layers,
        and activation functions according to the architecture defined for the network.
        """
        super(group_9, self).__init__()
        
        # Two convolutional layers to detect patterns in the images
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        # The convolutional layers are pooled to make the model more 
        # robust against translation, and 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Correct the input size for fc1 based on the calculations
        self.fc1 = nn.Linear(64 * 125*125, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width) where
                              batch_size is the number of images in a batch, 1 is the number
                              of input channels (e.g., grayscale), and height and width are
                              the dimensions of the images.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2) where each value represents
                          the probability of the input belonging to one of the two classes.
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return self.softmax(x)