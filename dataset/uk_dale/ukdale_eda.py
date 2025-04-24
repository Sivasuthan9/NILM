import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from datetime import datetime, timedelta

def load_data(file_path):
    """Load and process the UK-DALE dataset."""
    with h5py.File(file_path, 'r') as f:
        houses = list(f.keys())
        house_data = {}
        
        for house_name in houses:
            house = f[house_name]
            house_id = int(house_name.split('_')[1])
            
            channels = list(house.keys())
            house_data[house_id] = {
                'channels': {},
                'metadata': dict(house.attrs)
            }
            
            for channel_name in channels:
                channel = house[channel_name]
                channel_id = int(channel_name.split('_')[1])
                appliance_name = channel.attrs['name']
                
                power = channel['power'][:]
                timestamps = channel['timestamps'][:]
                
                if len(power) > 0:
                    stats = {
                        'min': np.min(power),
                        'max': np.max(power),
                        'mean': np.mean(power),
                        'median': np.median(power),
                        'std': np.std(power),
                        'non_zero': np.sum(power > 0),
                        'percent_on': np.mean(power > 0) * 100,
                        'start_time': datetime.fromtimestamp(timestamps[0]),
                        'end_time': datetime.fromtimestamp(timestamps[-1]),
                        'duration_days': (timestamps[-1] - timestamps[0]) / (24 * 3600),
                        'sample_rate': np.median(np.diff(timestamps)) if len(timestamps) > 1 else None
                    }
                else:
                    stats = {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0}
                
                house_data[house_id]['channels'][channel_id] = {
                    'name': appliance_name,
                    'stats': stats,
                    'sample': {
                        'power': power[:min(5000, len(power))],
                        'timestamps': timestamps[:min(5000, len(timestamps))]
                    }
                }
    
    return house_data

def calculate_energy(house_data):
    """Calculate energy consumption for each appliance."""
    energy_consumption = {}
    
    for house_id, house in house_data.items():
        energy_consumption[house_id] = {}
        
        for channel_id, channel in house['channels'].items():
            # Skip empty channels
            if len(channel['sample']['power']) == 0 or np.all(channel['sample']['power'] == 0):
                energy_consumption[house_id][channel['name']] = 0
                continue
                
            # Calculate energy in kWh based on mean power and duration
            mean_power = channel['stats']['mean']
            duration_hours = channel['stats']['duration_days'] * 24
            energy_kwh = mean_power * duration_hours / 1000
            
            energy_consumption[house_id][channel['name']] = energy_kwh
    
    return energy_consumption

def plot_daily_consumption(house_data, output_dir):
    """Plot daily power consumption patterns."""
    house1 = house_data[1]
    agg_channel = house1['channels'][1]
    
    timestamps = [datetime.fromtimestamp(ts) for ts in agg_channel['sample']['timestamps']]
    power = agg_channel['sample']['power']
    
    agg_df = pd.DataFrame({
        'timestamp': timestamps,
        'power': power
    })
    
    one_day = agg_df[agg_df['timestamp'] < (agg_df['timestamp'][0] + timedelta(days=1))]
    
    plt.figure(figsize=(12, 6))
    plt.plot(one_day['timestamp'], one_day['power'])
    plt.title('House 1: Aggregate Power Consumption (24 hours)', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Power (Watts)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/daily_power_consumption.png')
    
    return one_day

def plot_appliance_comparison(house_data, output_dir):
    """Compare power consumption of different appliances."""
    house1 = house_data[1]
    appliances_to_compare = [1, 2, 3, 4]  # Aggregate, fridge, kettle, microwave
    
    fig, axes = plt.subplots(len(appliances_to_compare), 1, figsize=(12, 10), sharex=True)
    
    for i, channel_id in enumerate(appliances_to_compare):
        if channel_id in house1['channels']:
            channel = house1['channels'][channel_id]
            name = channel['name']
            
            timestamps = [datetime.fromtimestamp(ts) for ts in channel['sample']['timestamps'][:1000]]
            power = channel['sample']['power'][:1000]
            
            axes[i].plot(timestamps, power)
            axes[i].set_title(f'{name.capitalize()} Power Consumption', fontsize=12)
            axes[i].set_ylabel('Power (W)', fontsize=10)
            axes[i].grid(True)
    
    plt.xlabel('Time', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/appliance_comparison.png')

def plot_energy_distribution(energy_consumption, output_dir):
    """Plot energy consumption by appliance."""
    if 1 in energy_consumption:
        # Remove aggregate from the comparison
        house1_energy = {k: v for k, v in energy_consumption[1].items() if k != 'aggregate'}
        
        if house1_energy:
            sorted_appliances = sorted(house1_energy.items(), key=lambda x: x[1], reverse=True)
            appliances = [item[0] for item in sorted_appliances]
            energy = [item[1] for item in sorted_appliances]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(appliances, energy)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}',
                        ha='center', va='bottom', rotation=0)
            
            plt.title('Energy Consumption by Appliance (House 1)', fontsize=14)
            plt.xlabel('Appliance', fontsize=12)
            plt.ylabel('Energy (kWh)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/energy_consumption.png')
            
            # Calculate percentage of total
            total_energy = sum(house1_energy.values())
            energy_percent = {k: (v/total_energy)*100 for k, v in house1_energy.items()}
            
            plt.figure(figsize=(10, 6))
            sorted_percent = sorted(energy_percent.items(), key=lambda x: x[1], reverse=True)
            appliances = [item[0] for item in sorted_percent]
            percentages = [item[1] for item in sorted_percent]
            
            bars = plt.bar(appliances, percentages)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%',
                        ha='center', va='bottom', rotation=0)
            
            plt.title('Percentage of Total Energy by Appliance (House 1)', fontsize=14)
            plt.xlabel('Appliance', fontsize=12)
            plt.ylabel('Percentage of Total (%)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/energy_percentage.png')
            
            return sorted_percent

def plot_hourly_patterns(house_data, output_dir):
    """Analyze hourly usage patterns."""
    if 1 in house_data and 1 in house_data[1]['channels']:
        agg_channel = house_data[1]['channels'][1]
        
        timestamps = agg_channel['sample']['timestamps']
        power = agg_channel['sample']['power']
        
        hours = [(datetime.fromtimestamp(ts).hour + 
                 datetime.fromtimestamp(ts).minute/60) for ts in timestamps]
        
        hourly_df = pd.DataFrame({
            'hour': hours,
            'power': power
        })
        
        hourly_avg = hourly_df.groupby(pd.cut(hourly_df['hour'], bins=24)).mean()
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(24), hourly_avg['power'])
        plt.title('Average Hourly Power Consumption (House 1)', fontsize=14)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Average Power (W)', fontsize=12)
        plt.xticks(range(24))
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/hourly_consumption.png')

def analyze_appliance_patterns(house_data, output_dir):
    """Analyze individual appliance usage patterns."""
    appliances_to_analyze = ['kettle', 'fridge', 'microwave']
    
    for appliance_name in appliances_to_analyze:
        for channel_id, channel in house_data[1]['channels'].items():
            if channel['name'] == appliance_name:
                timestamps = channel['sample']['timestamps']
                power = channel['sample']['power']
                
                hours = [(datetime.fromtimestamp(ts).hour + 
                         datetime.fromtimestamp(ts).minute/60) for ts in timestamps]
                
                app_df = pd.DataFrame({
                    'hour': hours,
                    'power': power,
                    'is_on': power > 10  # Consider appliance on if power > 10W
                })
                
                hourly_usage = app_df.groupby(pd.cut(app_df['hour'], bins=24)).mean()
                
                plt.figure(figsize=(10, 6))
                plt.bar(range(24), hourly_usage['is_on'] * 100)
                plt.title(f'{appliance_name.capitalize()} Usage Probability by Hour', fontsize=14)
                plt.xlabel('Hour of Day', fontsize=12)
                plt.ylabel('Usage Probability (%)', fontsize=12)
                plt.xticks(range(24))
                plt.grid(True, axis='y')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/{appliance_name}_usage_pattern.png')
                break

def run_ukdale_eda(data_file):
    """Run the complete EDA pipeline."""
    # Create output directory
    output_dir = "uk_dale_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    print("Loading and processing data...")
    house_data = load_data(data_file)
    
    # Calculate energy consumption
    print("Calculating energy consumption...")
    energy_consumption = calculate_energy(house_data)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_daily_consumption(house_data, output_dir)
    plot_appliance_comparison(house_data, output_dir)
    energy_dist = plot_energy_distribution(energy_consumption, output_dir)
    plot_hourly_patterns(house_data, output_dir)
    analyze_appliance_patterns(house_data, output_dir)
    
    # Print some key insights
    if energy_dist:
        print("\nTop energy consuming appliances in House 1:")
        for app, pct in energy_dist[:3]:
            print(f"  {app.capitalize()}: {pct:.1f}%")
    
    print(f"Analysis complete! Results saved to {output_dir}")
    
    return {
        'house_data': house_data,
        'energy_consumption': energy_consumption,
        'energy_distribution': energy_dist
    }

if __name__ == "__main__":
    # Check if synthetic dataset exists, otherwise create it
    data_file = "uk_dale_data/uk_dale_synthetic.h5"
    if not os.path.exists(data_file):
        from synthetic_ukdale import create_synthetic_ukdale
        data_file = create_synthetic_ukdale()
    
    # Run EDA
    run_ukdale_eda(data_file)
