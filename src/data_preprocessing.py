import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class NILMDataset(Dataset):
    """PyTorch Dataset for NILM data"""
    def __init__(self, mains_sequences, appliance_sequences):
        self.mains_sequences = mains_sequences
        self.appliance_sequences = appliance_sequences
        
    def __len__(self):
        return len(self.mains_sequences)
    
    def __getitem__(self, idx):
        mains_seq = self.mains_sequences[idx]
        appliance_seq = self.appliance_sequences[idx]
        
        # Convert to PyTorch tensors
        mains_seq = torch.FloatTensor(mains_seq)
        appliance_seq = torch.FloatTensor(appliance_seq)
        
        return mains_seq, appliance_seq

def load_data(data_dir, appliance_name):
    """Load data for a specific appliance"""
    mains_path = os.path.join(data_dir, "house_1_mains.csv")
    appliance_path = os.path.join(data_dir, f"house_1_{appliance_name}.csv")
    
    if not os.path.exists(mains_path) or not os.path.exists(appliance_path):
        raise FileNotFoundError(f"Data files not found at {data_dir}")
    
    # Load data
    mains_data = pd.read_csv(mains_path)
    appliance_data = pd.read_csv(appliance_path)
    
    return mains_data, appliance_data

def create_sequences(mains_data, appliance_data, seq_length=299, stride=10):
    """Create sliding window sequences"""
    mains_power = mains_data["power"].values
    appliance_power = appliance_data["power"].values
    
    # Calculate number of sequences
    n_samples = len(mains_power)
    n_sequences = (n_samples - seq_length) // stride + 1
    
    # Initialize arrays
    mains_sequences = np.zeros((n_sequences, seq_length, 1))
    appliance_sequences = np.zeros((n_sequences, seq_length, 1))
    
    # Create sequences
    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + seq_length
        
        mains_sequences[i, :, 0] = mains_power[start_idx:end_idx]
        appliance_sequences[i, :, 0] = appliance_power[start_idx:end_idx]
    
    return mains_sequences, appliance_sequences

def normalize_data(train_data, val_data, test_data):
    """Normalize data using StandardScaler"""
    # Reshape for scaling
    train_flat = train_data.reshape(-1, train_data.shape[-1])
    scaler = StandardScaler()
    train_flat_normalized = scaler.fit_transform(train_flat)
    
    # Reshape back
    train_normalized = train_flat_normalized.reshape(train_data.shape)
    
    # Apply same transformation to validation and test
    val_flat = val_data.reshape(-1, val_data.shape[-1])
    val_normalized = scaler.transform(val_flat).reshape(val_data.shape)
    
    test_flat = test_data.reshape(-1, test_data.shape[-1])
    test_normalized = scaler.transform(test_flat).reshape(test_data.shape)
    
    return train_normalized, val_normalized, test_normalized, scaler

def preprocess_data(appliance_name, data_dir, seq_length=299, batch_size=32, stride=10):
    """Main function to preprocess data for NILM"""
    print(f"Preprocessing data for {appliance_name}")
    
    # Load data
    mains_data, appliance_data = load_data(data_dir, appliance_name)
    
    # Create sequences
    mains_sequences, appliance_sequences = create_sequences(
        mains_data, appliance_data, seq_length=seq_length, stride=stride
    )
    
    # Split data (70% train, 15% validation, 15% test)
    train_idx, temp_idx = train_test_split(
        np.arange(len(mains_sequences)), test_size=0.3, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42
    )
    
    # Extract data for each split
    train_mains = mains_sequences[train_idx]
    train_app = appliance_sequences[train_idx]
    
    val_mains = mains_sequences[val_idx]
    val_app = appliance_sequences[val_idx]
    
    test_mains = mains_sequences[test_idx]
    test_app = appliance_sequences[test_idx]
    
    # Normalize data
    train_mains_norm, val_mains_norm, test_mains_norm, mains_scaler = normalize_data(
        train_mains, val_mains, test_mains
    )
    
    train_app_norm, val_app_norm, test_app_norm, app_scaler = normalize_data(
        train_app, val_app, test_app
    )
    
    # Create PyTorch datasets
    train_dataset = NILMDataset(train_mains_norm, train_app_norm)
    val_dataset = NILMDataset(val_mains_norm, val_app_norm)
    test_dataset = NILMDataset(test_mains_norm, test_app_norm)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Train set: {len(train_dataset)} sequences")
    print(f"Validation set: {len(val_dataset)} sequences")
    print(f"Test set: {len(test_dataset)} sequences")
    
    # Create preprocessing info dictionary
    preprocessing_info = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "mains_scaler": mains_scaler,
        "appliance_scaler": app_scaler
    }
    
    return preprocessing_info
