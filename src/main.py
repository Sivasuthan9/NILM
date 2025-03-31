import os
import json
import torch
import numpy as np
from model import TransformerNILM, NILMLoss
from data_preprocessing import preprocess_data
from training import train_model
from evaluation import evaluate_model

def main(config_path):
    """Main function to run the NILM pipeline"""
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Extract parameters
    appliance_name = config["training"]["appliance_name"]
    data_dir = config.get("data_dir", "../../data/ukdale")
    results_dir = config.get("results_dir", "../results")
    models_dir = config.get("models_dir", "../models")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    data_info = preprocess_data(
        appliance_name=appliance_name,
        data_dir=data_dir,
        seq_length=config["model"]["seq_length"],
        batch_size=config["training"]["batch_size"],
        stride=config["training"]["stride"]
    )
    
    # Create model
    model = TransformerNILM(
        input_dim=config["model"]["input_dim"],
        output_dim=config["model"]["output_dim"],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        num_encoder_layers=config["model"]["num_encoder_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        seq_length=config["model"]["seq_length"]
    ).to(device)
    
    print(f"Created model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Initialize loss function and optimizer
    criterion = NILMLoss(alpha=0.7, beta=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    
    # Train model
    best_model_path, history = train_model(
        model=model,
        train_loader=data_info["train_loader"],
        val_loader=data_info["val_loader"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config["training"]["epochs"],
        patience=config["training"]["patience"],
        save_dir=models_dir,
        appliance_name=appliance_name
    )
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_loader=data_info["test_loader"],
        device=device,
        app_scaler=data_info["appliance_scaler"],
        appliance_name=appliance_name,
        results_dir=results_dir
    )
    
    # Save metrics
    with open(os.path.join(results_dir, f"{appliance_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

if __name__ == "__main__":
    # Use the configuration file
    config_path = "../config/config.json"
    
    if not os.path.exists(config_path):
        # Create a default configuration
        default_config = {
            "model": {
                "input_dim": 1,
                "output_dim": 1,
                "d_model": 128,
                "nhead": 8,
                "num_encoder_layers": 4,
                "dim_feedforward": 512,
                "dropout": 0.1,
                "seq_length": 299
            },
            "training": {
                "appliance_name": "dishwasher",
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 50,
                "patience": 10,
                "stride": 20
            },
            "data_dir": "../../data/ukdale",
            "results_dir": "../results",
            "models_dir": "../models"
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=4)
    
    # Run the pipeline
    main(config_path)
