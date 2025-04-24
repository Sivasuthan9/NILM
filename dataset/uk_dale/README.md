# UK-DALE Dataset Exploratory Data Analysis

## Overview
This repository contains exploratory data analysis (EDA) of the UK-DALE (UK Domestic Appliance-Level Electricity) dataset. The analysis explores electricity consumption patterns at both whole-house and individual appliance levels.

## Dataset Structure
The UK-DALE dataset contains power consumption data from five UK homes in HDF5 format, structured as:

```
/house_X/                      # Data for house X (where X is 1-5)
  /house_X/channel_Y/          # Data for channel Y in house X
    /house_X/channel_Y/power   # Power readings in watts
    /house_X/channel_Y/timestamps  # Unix timestamps for readings
```

In each house, channel 1 is the aggregate (whole-house) consumption, with other channels representing individual appliances.

## Files Description
- `synthetic_ukdale.py`: Creates a synthetic dataset mirroring UK-DALE structure
- `ukdale_eda.py`: Main EDA script with visualization and analysis functions
- `uk_dale_output/`: Directory containing all generated visualizations and data

## Key Insights
- Cooking appliances (microwave, dishwasher) consume the highest proportion of monitored electricity
- Clear daily patterns emerge with morning and evening peaks
- Each appliance demonstrates distinctive usage signatures and patterns
- Refrigerators show regular cycling patterns throughout the day
- Kettles show short, high-power usage primarily in mornings

## Visualizations
The analysis generates several visualizations:
1. Daily power consumption patterns
2. Appliance comparison charts
3. Energy distribution by appliance
4. Hourly usage patterns
5. Individual appliance usage patterns

## Usage
```
# Generate synthetic dataset (if needed)
python synthetic_ukdale.py

# Run the full EDA
python ukdale_eda.py
```

## Requirements
- Python 3.6+
- numpy
- pandas
- matplotlib
- seaborn
- h5py
