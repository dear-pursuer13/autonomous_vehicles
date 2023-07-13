"""
Co-Author: Danya Liu 
E-mail: danya.liu@tum.de
Date: 28.05.2023
Project: Task 2
"""

"""
  *  @copyright (c) 2020 Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *  @file    network_model.py
  *  @author  Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *
  *  @brief Class declaration for building the network model.  
 """

import torch
import torch.nn as nn
from torchsummary import summary
"""
* @brief Class declaration for building the training network model.
* @param None.
* @return The built netwwork.
"""
class model_cnn(nn.Module):
    """
    * @brief Initializes the class varaibles
    * @param None.
    * @return None.
    """
    def __init__(self):
        super(model_cnn, self).__init__()

        # 5 Convolutions layer

        # Together with batch normalization to improve the stability of this network
        self.conv_0 = nn.Conv2d(3, 24, 5, stride=2)
        self.bn_0 = nn.BatchNorm2d(24)
        self.conv_1 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.bn_1 = nn.BatchNorm2d(36)
        self.conv_2 = nn.Conv2d(36, 48, kernel_size=5, stride=2) #384 kernels, size 3x3
        self.bn_2 = nn.BatchNorm2d(48)
        self.conv_3 = nn.Conv2d(48, 64, kernel_size=3) # 384 kernels size 3x3
        self.bn_3 = nn.BatchNorm2d(64)
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3) # 256 kernels, size 3x3
        self.bn_4 = nn.BatchNorm2d(64)

        # Four fully-connection layer to get the final result

        self.fc0 = nn.Linear(1152, 100)
        self.fc1 = nn.Linear(100, 50)

        # Add the lstm to complete the generalization process
        self.lstm = nn.LSTM(input_size=50, hidden_size=64, num_layers=1, batch_first=True)

        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 1)

        # Set the activation function
        self.elu = nn.ELU()

    """ 
    * @brief Function to build the model.
    * @parma The image to train.
    * @return The trained prediction network.
    """
    def forward(self, input):
        input = self.elu(self.bn_0(self.conv_0(input)))
        input = self.elu(self.bn_1(self.conv_1(input)))
        input = self.elu(self.bn_2(self.conv_2(input)))
        input = self.elu(self.bn_3(self.conv_3(input)))
        input = self.elu(self.bn_4(self.conv_4(input)))

        # Flatten the input
        input = input.reshape(input.size(0), -1)  

        input = self.elu(self.fc0(input))
        input = self.elu(self.fc1(input))
        # Add extra dimension for LSTM input
        input = input.unsqueeze(1)  

        # LSTM layer
        output, _ = self.lstm(input)
        # Extract the output at the last time step
        lstm_output = output[:, -1, :]  
        
        input = nn.functional.elu(self.fc2(lstm_output))
        input = self.fc3(input)

        return input
