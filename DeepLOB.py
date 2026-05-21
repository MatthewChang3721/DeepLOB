import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class DeepLOB(nn.Module):
    def __init__(self, num_features: int, num_classes: int = 3):
        super().__init__()

        # Convolutional layers
        self.causal_pad = nn.ZeroPad2d((0, 0, 3, 0))  # Pad only on the top for time dimension

        self.conv_time1 = nn.Conv2d(1, 16, kernel_size=(4, 1))
        self.conv_time2 = nn.Conv2d(16, 16, kernel_size=(4, 1))
        self.conv_time3 = nn.Conv2d(16, 16, kernel_size=(4, 1))
        self.conv_time4 = nn.Conv2d(16, 16, kernel_size=(4, 1))
        self.conv_feat1 = nn.Conv2d(16, 16, kernel_size=(1, num_features))
        self.conv_time5 = nn.Conv2d(16, 16, kernel_size=(4, 1))
        self.conv_time6 = nn.Conv2d(16, 16, kernel_size=(4, 1))

        # Inception layer would be defined here (not implemented in this simple version)
        self.incp_Path1_1 = nn.Conv2d(16, 32, kernel_size=(1,1))
        self.incp_Path1_pad = nn.ZeroPad2d((0, 0, 2, 0))  # Pad only on the top for time dimension
        self.incp_Path1_2 = nn.Conv2d(32, 32, kernel_size=(3,1))

        self.incp_Path2_1 = nn.Conv2d(16, 32, kernel_size=(1,1))
        self.incp_Path2_pad = nn.ZeroPad2d((0, 0, 4, 0))  # Pad only on the top for time dimension
        self.incp_Path2_2 = nn.Conv2d(32, 32, kernel_size=(5,1))

        self.incp_Path3_pad = nn.ZeroPad2d((0, 0, 2, 0))  # Pad only on the top for time dimension
        self.incp_Path3_1 = nn.MaxPool2d(kernel_size=(3,1), stride=(1,1))
        self.incp_Path3_2 = nn.Conv2d(16, 32, kernel_size=(1,1))

        # LSTM layer with 64 hidden units
        self.lstm = nn.LSTM(input_size=96, hidden_size=64, batch_first=True, bidirectional=False)

        # Fully connected layer for classification
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, time_steps, num_features]
        x = x.unsqueeze(1)            # [batch, 1, time_steps, num_features]

        # Convolution Blocks
        # Time Convolutions Block1 (4*1)@16
        x = self.causal_pad(x)  # Apply causal padding before the time convolution
        x = F.leaky_relu(self.conv_time1(x), negative_slope=0.01)
       
        # Time Convolutions Block2 (4*1)@16
        x = self.causal_pad(x)  # Apply causal padding before the time convolution
        x = F.leaky_relu(self.conv_time2(x), negative_slope=0.01)
        
        # Time Convolutions Block3 (4*1)@16
        x = self.causal_pad(x)  # Apply causal padding before the time convolution
        x = F.leaky_relu(self.conv_time3(x), negative_slope=0.01)
        
        # Time Convolutions Block4 (4*1)@16
        x = self.causal_pad(x)  # Apply causal padding before the time convolution
        x = F.leaky_relu(self.conv_time4(x), negative_slope=0.01)
        
        # Feature Convolution (1*10)@16
        x = F.leaky_relu(self.conv_feat1(x), negative_slope=0.01)

        # Time Convolutions Block5 (4*1)@16
        x = self.causal_pad(x)  # Apply causal padding before the time convolution
        x = F.leaky_relu(self.conv_time5(x), negative_slope=0.01)

        # Time Convolutions Block6 (4*1)@16  RF = 1 + 6 * (4-1) = 19
        x = self.causal_pad(x)  # Apply causal padding before the time convolution
        x = F.leaky_relu(self.conv_time6(x), negative_slope=0.01)
        # Output shape after conv_time6: [batch, 16, time_steps:200, features:1] 

        # Inception Block
        # Path 1: (1*1)@32 -> (3*1)@32
        x_path1 = F.leaky_relu(self.incp_Path1_1(x), negative_slope=0.01)
        x_path1 = self.incp_Path1_pad(x_path1)
        x_path1 = F.leaky_relu(self.incp_Path1_2(x_path1), negative_slope=0.01)

        # Path 2: (1*1)@32 -> (5*1)@32
        x_path2 = F.leaky_relu(self.incp_Path2_1(x), negative_slope=0.01)
        x_path2 = self.incp_Path2_pad(x_path2)
        x_path2 = F.leaky_relu(self.incp_Path2_2(x_path2), negative_slope=0.01)

        # Path 3: MaxPool(3*1) -> (1*1)@32
        x_path3 = self.incp_Path3_pad(x)
        x_path3 = self.incp_Path3_1(x_path3)
        x_path3 = F.leaky_relu(self.incp_Path3_2(x_path3), negative_slope=0.01)

        x = torch.cat([x_path1, x_path2, x_path3], dim=1)  # Concatenate along the channel dimension
        # Output shape after concatenation: [batch, 96, time_steps:200, features:1]

        # LSTM Block
        # Reshape from [batch, 96, time_steps, features:1] to [batch, time_steps, features:96]
        x = x.squeeze(3).transpose(1, 2)
        output, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]  # Take the last hidden state from the LSTM and reshape to [batch, hidden_size]

        # Dense
        x = self.head(x)        
        
        return x

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()  # Set model to training mode
    total_loss = 0.0

    pbar = tqdm(train_loader, desc="Training", unit="batch")
        
    for i, (data, label) in enumerate(pbar):
        data, label = data.to(device), label.to(device).long().squeeze()  # Ensure labels are the correct shape and type

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        
        optimizer.step()

        total_loss += loss.item()

        current_avg_loss = total_loss / (i + 1)
        pbar.set_postfix({'loss': f'{current_avg_loss:.4f}'})

    avg_loss = total_loss / len(train_loader)

    return avg_loss