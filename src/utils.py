
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: nilm_project/src/utils.py
# execution: true
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_model(model, path):
    """Save PyTorch model"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """Load weights into model"""
    model.load_state_dict(torch.load(path))
    return model

def plot_learning_curves(train_losses, val_losses, save_path):
    """Plot and save learning curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_prediction_example(y_true, y_pred, save_path):
    """Plot example of prediction vs ground truth"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Ground Truth')
    plt.plot(y_pred, label='Prediction')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time Steps')
    plt.ylabel('Power')
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean(np.square(y_true - y_pred))
    rmse = np.sqrt(mse)
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }

def save_metrics(metrics, path):
    """Save metrics to file"""
    with open(path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Metrics saved to {path}")

# Create and check directory exists
def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)