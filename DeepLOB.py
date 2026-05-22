import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

class TimeSeriesDataset(Dataset):
    def __init__(self, data: torch.Tensor, window_size: int = 200, target_size: int = 1):
        self.data = data
        self.window_size = window_size
        self.target_size = target_size

    def __len__(self):
        return len(self.data) - self.window_size - self.target_size + 1
    
    def __getitem__(self, idx):
        x_start = idx
        x_end = idx + self.window_size
        X = self.data[x_start:x_end, :-1]  # Features

        y_start = x_end
        y_end = x_end + self.target_size
        Y = self.data[y_start:y_end, -1]  # Target label

        return X, Y

def create_dataloader(start_files: int, num_files: int, window_size: int = 200, target_size: int = 1, batch_size: int = 32, shuffle: bool = True):
    from pathlib import Path
    all_daily_ds = []

    csv_files = sorted(Path('Data/Normalized_data').glob('normalized_*.csv'))[start_files:start_files + num_files]
    print(f"Found {len(csv_files)} CSV files for this dataloader.")

    label_counts_overall = np.zeros(3, dtype=int)  # To count occurrences of each label across all files
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        data['price_move_label'] = data['price_move_label'] + 1  # Shift labels to be 0, 1, 2 instead of -1, 0, 1
        label_counts_overall += data['price_move_label'].value_counts().sort_index().values
        tc_data = torch.tensor(data.values, dtype=torch.float32)

        # Denfensively check length of df is sufficient for windowing
        if len(data) < window_size + target_size:
            print(f"Warning: {csv_file} has insufficient data ({len(data)} rows). Skipping this file.")
            continue

        ts_dataset = TimeSeriesDataset(tc_data, window_size=window_size, target_size=target_size)
        all_daily_ds.append(ts_dataset)
    
    label_names = ['Down', 'Neutral', 'Up']

    final_dataset = ConcatDataset(all_daily_ds)

    # DataLoader
    data_loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=shuffle)

    print(f'\nTotal days used for training: {len(all_daily_ds)}')
    print(f'Total samples in final dataset: {len(final_dataset)}')

    X_sample, Y_sample = final_dataset[0]
    print(f'\nSample X shape: {X_sample.shape}, Sample Y shape: {Y_sample.shape}')

    print(f'\nBatch Size: {data_loader.batch_size}')
    print(f'Total number of batches per epoch: {len(data_loader)}')

    for batch_idx, (data, label) in enumerate(data_loader):
        print(f"\nData shape: {data.shape}") 
        print(f"Label shape: {label.shape}")
        break
    label_df = pd.DataFrame(label_counts_overall, index=label_names, columns=['Count'])
    print(f"\nLabel distribution across all training files:\n{label_df}")

    return data_loader, label_df

class DeepLOB(nn.Module):
    def __init__(self, num_features: int, num_classes: int = 3):
        super().__init__()

        # Convolutional layers
        self.causal_pad = nn.ZeroPad2d((0, 0, 3, 0))  # Pad only on the top for time dimension

        self.conv_feat1 = nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2))
        self.conv_time1 = nn.Conv2d(16, 16, kernel_size=(4, 1))
        self.conv_time2 = nn.Conv2d(16, 16, kernel_size=(4, 1))
        self.conv_feat2 = nn.Conv2d(16, 16, kernel_size=(1, 2), stride=(1,2))
        self.conv_time3 = nn.Conv2d(16, 16, kernel_size=(4, 1))
        self.conv_time4 = nn.Conv2d(16, 16, kernel_size=(4, 1))
        self.conv_feat3 = nn.Conv2d(16, 16, kernel_size=(1, num_features))
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
        # Space Convolutions Block1 (1*2)@16 stride(1*2)
        x = F.leaky_relu(self.conv_feat1(x), negative_slope=0.01)

        # Time Convolutions Block1 (4*1)@16
        x = self.causal_pad(x)  # Apply causal padding before the time convolution
        x = F.leaky_relu(self.conv_time1(x), negative_slope=0.01)
       
        # Time Convolutions Block2 (4*1)@16
        x = self.causal_pad(x)  # Apply causal padding before the time convolution
        x = F.leaky_relu(self.conv_time2(x), negative_slope=0.01)
        
        # Space Convolutions Block2 (1*2)@16 stride(1*2)
        x = F.leaky_relu(self.conv_feat2(x), negative_slope=0.01)

        # Time Convolutions Block3 (4*1)@16
        x = self.causal_pad(x)  # Apply causal padding before the time convolution
        x = F.leaky_relu(self.conv_time3(x), negative_slope=0.01)
        
        # Time Convolutions Block4 (4*1)@16
        x = self.causal_pad(x)  # Apply causal padding before the time convolution
        x = F.leaky_relu(self.conv_time4(x), negative_slope=0.01)
        
        # Feature Convolution (1*10)@16
        x = F.leaky_relu(self.conv_feat3(x), negative_slope=0.01)

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

class FocalLoss(nn.Module):
    def __init__(self, weight = None, gamma = 2.0, reduction = 'mean'):
        super(FocalLoss,self).__init__()
        self.weight = weight
        self.gamma = gamma
        self. reduction = reduction

    def forward(self, inputs, targets):
        ce_loss =  F.cross_entropy(inputs, targets, weight = self.weight, reduction = self.reduction)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1-pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()  # Set model to training mode
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", unit="batch")
        
    for i, (data, label) in enumerate(pbar):
        data, label = data.to(device), label.to(device).long().squeeze()  # Ensure labels are the correct shape and type

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        total_loss += loss.item()

        current_avg_loss = total_loss / (i + 1)
        current_avg_acc = 100 * correct / total
        pbar.set_postfix({'loss': f'{current_avg_loss:.4f}', 'acc': f'{current_avg_acc:.2f}%'})

    avg_loss = total_loss / len(train_loader)

    return avg_loss, 100 * correct/total

def validate_engine(model, val_loader, criterion, device):
    model.eval() # Switch to evluation
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad(): # stop gradient
        for data, label in val_loader:
            data, label = data.to(device), label.to(device).long().squeeze()
            all_labels.append(label)

            output = model(data)
            loss = criterion(output, label)
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            all_preds.append(predicted)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            
    return total_loss / len(val_loader), 100 * correct / total, all_preds, all_labels   