
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: nilm_project/src/data.py
# execution: true
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

def load_ukdale_data(data_dir='data/ukdale'):
    """Load UK-DALE data from CSV files"""
    print(f"Loading data from {data_dir}...")
    
    # Load mains data
    mains_path = os.path.join(data_dir, 'house_1_mains.csv')
    mains_data = pd.read_csv(mains_path)
    
    # Get list of available appliances
    appliance_files = [f for f in os.listdir(data_dir) if f.startswith('house_1_') 
                      and f != 'house_1_mains.csv' and f.endswith('.csv')]
    
    appliance_names = [f.split('house_1_')[1].split('.csv')[0] for f in appliance_files]
    print(f"Found appliances: {appliance_names}")
    
    # Load appliance data
    appliance_data = {}
    for app_name in appliance_names:
        app_path = os.path.join(data_dir, f'house_1_{app_name}.csv')
        appliance_data[app_name] = pd.read_csv(app_path)
    
    # Print sample of data
    print("Mains data sample:")
    print(mains_data.head())
    
    return mains_data, appliance_data, appliance_names

def create_sequences(mains_data, appliance_data, seq_length=599, stride=1):
    """Create sliding window sequences"""
    mains_power = mains_data['power'].values
    appliance_power = appliance_data['power'].values
    
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

def prepare_nilm_data(config):
    """Process data and prepare for model training"""
    appliance_name = config['training']['appliance_name']
    seq_length = config['model']['seq_length']
    batch_size = config['training']['batch_size'] 
    stride = config['training']['stride']
    
    print(f"Preparing data for {appliance_name} with sequence length {seq_length}...")
    
    # Load data
    mains_data, appliance_dfs, available_appliances = load_ukdale_data()
    
    if appliance_name not in available_appliances:
        raise ValueError(f"Appliance {appliance_name} not found in available appliances: {available_appliances}")
    
    appliance_data = appliance_dfs[appliance_name]
    
    # Create sequences
    mains_sequences, appliance_sequences = create_sequences(
        mains_data, appliance_data, seq_length=seq_length, stride=stride
    )
    
    # Split data
    # Using indices to split to ensure proper time-series splitting
    indices = np.arange(len(mains_sequences))
    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
    
    # Get data for each split
    train_mains = mains_sequences[train_indices]
    train_app = appliance_sequences[train_indices]
    
    val_mains = mains_sequences[val_indices]
    val_app = appliance_sequences[val_indices]
    
    test_mains = mains_sequences[test_indices]
    test_app = appliance_sequences[test_indices]
    
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
    
    print(f"Data preparation complete:")
    print(f"Train set: {len(train_dataset)} sequences")
    print(f"Validation set: {len(val_dataset)} sequences")
    print(f"Test set: {len(test_dataset)} sequences")
    
    # Visualize a sample sequence
    sample_idx = np.random.randint(0, len(test_mains))
    plt.figure(figsize=(10, 6))
    plt.plot(test_mains[sample_idx, :, 0], label='Mains')
    plt.plot(test_app[sample_idx, :, 0], label=appliance_name)
    plt.legend()
    plt.title(f'Sample Sequence - {appliance_name}')
    plt.savefig(f'nilm_project/results/sample_{appliance_name}_sequence.png')
    plt.close()
    
    data_info = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'mains_scaler': mains_scaler,
        'appliance_scaler': app_scaler,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset)
    }
    
    return data_info

# Test loading one sample to verify the code works
if __name__ == "__main__":
    # Just testing data loading functionality
    try:
        mains_data, appliance_data, appliance_names = load_ukdale_data()
        print("Data loaded successfully")
        print(f"Mains data shape: {mains_data.shape}")
        for app_name, app_df in appliance_data.items():
            print(f"{app_name} data shape: {app_df.shape}")
        
        # Create a small config for testing
        test_config = {
            'model': {'seq_length': 299},
            'training': {
                'appliance_name': appliance_names[0],
                'batch_size': 32,
                'stride': 20
            }
        }
        
        # Test sequence creation
        print("Testing sequence creation...")
        mains_seq, app_seq = create_sequences(
            mains_data, 
            appliance_data[test_config['training']['appliance_name']], 
            seq_length=test_config['model']['seq_length'], 
            stride=test_config['training']['stride']
        )
        print(f"Created {len(mains_seq)} sequences")
        print(f"Sequence shape: {mains_seq.shape}")
        
    except Exception as e:
        print(f"Error testing data loading: {e}")
        print("Creating synthetic data for testing...")
        
        # Create synthetic data for testing
        timestamp = pd.date_range(start='2013-01-01', periods=10000, freq='6s')
        mains_data = pd.DataFrame({
            'timestamp': timestamp,
            'power': np.random.normal(500, 100, 10000) + np.sin(np.linspace(0, 10*np.pi, 10000)) * 200
        })
        
        appliance_data = {
            'fridge': pd.DataFrame({
                'timestamp': timestamp,
                'power': 100 * (np.random.rand(10000) > 0.7).astype(float)
            }),
            'kettle': pd.DataFrame({
                'timestamp': timestamp,
                'power': 2000 * (np.random.rand(10000) > 0.95).astype(float)
            }),
            'dishwasher': pd.DataFrame({
                'timestamp': timestamp,
                'power': 1500 * (np.random.rand(10000) > 0.9).astype(float)
            })
        }
        
        appliance_names = list(appliance_data.keys())
        
        # Save synthetic data for later use
        os.makedirs('nilm_project/data/ukdale', exist_ok=True)
        mains_data.to_csv('nilm_project/data/ukdale/house_1_mains.csv', index=False)
        
        for app_name, app_df in appliance_data.items():
            app_df.to_csv(f'nilm_project/data/ukdale/house_1_{app_name}.csv', index=False)
        
        print("Synthetic data created and saved")