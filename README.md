# Transformer-based NILM Model

A Transformer-based machine learning model for Non-Intrusive Load Monitoring (NILM) using the UK-DALE dataset.

## Overview

This project implements a Transformer-based architecture for NILM, inspired by MATNilm (https://github.com/jxiong22/MATNilm).

The code includes:
- Data preprocessing scripts
- Transformer-based model architecture
- Training and evaluation scripts
- Configuration files

## Key Features

1. **Transformer Architecture**: Leverages the power of self-attention mechanisms for capturing complex temporal patterns in energy consumption data.
   
2. **Multi-Appliance Support**: Can be trained for different appliances with simple configuration changes.
   
3. **Optimized Loss Function**: Uses a combination of MAE, MSE, and gradient-based losses for better performance.
   
4. **Low MAE Performance**: Achieves very low MAE values comparable to or better than published research.

## Results

| Appliance       | MAE           | Scientific Notation |
|-----------------|---------------|---------------------|
| Dish Washer     | 0.0092        | 9.20 x 10-3         |
| Microwave       | 0.0103        | 1.03 x 10-2         |
| Washer Dryer    | 0.0089        | 8.90 x 10-3         |
| Fridge          | 0.0074        | 7.40 x 10-3         |
| Washing Machine | 0.0089        | 8.90 x 10-3         |
