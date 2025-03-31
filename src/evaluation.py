import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(
    model,
    test_loader,
    device,
    app_scaler,
    appliance_name,
    results_dir
):
    """
    Evaluate a trained NILM model
    """
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate on test set
    model.eval()
    all_preds = []
    all_targets = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move data to device
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Store predictions and targets
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
    
    # Concatenate batches
    predictions = np.concatenate(all_preds, axis=0)
    ground_truth = np.concatenate(all_targets, axis=0)
    
    # Reshape for denormalization
    predictions_flat = predictions.reshape(-1, 1)
    ground_truth_flat = ground_truth.reshape(-1, 1)
    
    # Denormalize
    predictions_denorm = app_scaler.inverse_transform(predictions_flat).reshape(predictions.shape)
    ground_truth_denorm = ground_truth_flat.reshape(ground_truth.shape)
    
    # Calculate metrics
    mae = mean_absolute_error(ground_truth_denorm.flatten(), predictions_denorm.flatten())
    rmse = np.sqrt(mean_squared_error(ground_truth_denorm.flatten(), predictions_denorm.flatten()))
    
    # Calculate scientific notation
    mae_sci = format(mae, ".2e")
    
    # Print results
    print(f"Evaluation results for {appliance_name}")
    print(f"MAE: {mae}")
    print(f"MAE (scientific notation): {mae_sci}")
    print(f"RMSE: {rmse}")
    
    # Plot sample predictions
    num_samples = min(5, len(predictions))
    plt.figure(figsize=(15, 10))
    
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i+1)
        
        # Get sample index
        idx = np.random.randint(0, len(predictions))
        
        # Plot sequences
        plt.plot(ground_truth_denorm[idx, :, 0], "b-", label="Ground Truth")
        plt.plot(predictions_denorm[idx, :, 0], "r-", label="Prediction")
        
        plt.title(f"Sample {i+1}: {appliance_name}")
        plt.ylabel("Power (W)")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{appliance_name}_predictions.png"))
    
    metrics = {
        "appliance": appliance_name,
        "mae": mae,
        "mae_scientific": mae_sci,
        "rmse": rmse
    }
    
    return metrics
