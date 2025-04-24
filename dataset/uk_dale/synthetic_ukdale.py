import os
import h5py
import numpy as np
from datetime import datetime

def create_synthetic_ukdale():
    """Create a synthetic UK-DALE dataset with realistic structure."""
    os.makedirs("uk_dale_data", exist_ok=True)
    
    # Define appliances
    appliances = {
        1: "aggregate", 2: "fridge", 3: "kettle", 4: "microwave",
        5: "washing machine", 6: "dishwasher", 7: "computer", 8: "television"
    }
    
    # Create synthetic sample
    sample_file = "uk_dale_data/uk_dale_synthetic.h5"
    
    with h5py.File(sample_file, 'w') as f:
        # Create houses 1-5
        for house_num in range(1, 6):
            house = f.create_group(f"house_{house_num}")
            house.attrs["name"] = f"House {house_num}"
            house.attrs["device_model"] = "Current Cost IAMs"
            
            # Add channels (appliances)
            channels_count = min(len(appliances), 9 - house_num)
            
            for channel_num in range(1, channels_count + 1):
                channel = house.create_group(f"channel_{channel_num}")
                channel.attrs["name"] = appliances.get(channel_num, f"unknown_{channel_num}")
                
                # Duration and sample rates
                duration_days = {1: 350, 2: 30, 3: 100, 4: 70, 5: 40}.get(house_num, 10)
                start_time = datetime(2013, 4, 1).timestamp()
                sample_rate = 1 if house_num == 1 and channel_num == 1 else 6
                
                # Generate timestamps
                timestamps = np.arange(
                    start_time, 
                    start_time + (duration_days * 24 * 60 * 60), 
                    sample_rate
                )
                
                # Generate synthetic power data
                if channel_num == 1:  # Aggregate
                    hour_of_day = (((timestamps - start_time) / 3600) % 24).astype(int)
                    base_load = 100 + np.random.normal(0, 10, size=len(timestamps))
                    morning_peak = 300 * ((hour_of_day >= 7) & (hour_of_day <= 9)).astype(float)
                    evening_peak = 500 * ((hour_of_day >= 18) & (hour_of_day <= 23)).astype(float)
                    random_spikes = np.random.exponential(1, size=len(timestamps)) * 100
                    spike_mask = np.random.random(size=len(timestamps)) < 0.05
                    spikes = random_spikes * spike_mask
                    power_data = base_load + morning_peak + evening_peak + spikes
                elif appliances.get(channel_num) == "fridge":
                    cycles = np.sin(np.arange(len(timestamps)) / 500) > 0.7
                    power_data = cycles.astype(float) * 80 + np.random.normal(0, 3, size=cycles.shape)
                elif appliances.get(channel_num) == "kettle":
                    power_data = np.zeros(len(timestamps))
                    for day in range(duration_days):
                        uses_per_day = np.random.randint(5, 15)
                        for _ in range(uses_per_day):
                            start_idx = np.random.randint(
                                day * 24 * 60 * 60 // sample_rate,
                                (day + 1) * 24 * 60 * 60 // sample_rate - 100
                            )
                            duration_idx = 180 // sample_rate
                            if start_idx + duration_idx < len(power_data):
                                power_data[start_idx:start_idx + duration_idx] = 2000 + np.random.normal(0, 50, size=duration_idx)
                else:  # Generic appliance
                    power_data = np.zeros(len(timestamps))
                    for day in range(duration_days):
                        for _ in range(5):
                            start_idx = np.random.randint(
                                day * 24 * 60 * 60 // sample_rate,
                                (day + 1) * 24 * 60 * 60 // sample_rate - 1800
                            )
                            duration_idx = 1800 // sample_rate
                            if start_idx + duration_idx < len(power_data):
                                avg_power = {
                                    "microwave": 1200,
                                    "computer": 150,
                                    "television": 100,
                                    "dishwasher": 700,
                                    "washing machine": 1000
                                }.get(appliances.get(channel_num), 200)
                                power_data[start_idx:start_idx + duration_idx] = avg_power + np.random.normal(0, avg_power * 0.1, size=duration_idx)
                
                # Limit data size for sample
                max_points = 10000 if house_num == 1 and channel_num == 1 else 5000
                if len(timestamps) > max_points:
                    timestamps = timestamps[:max_points]
                    power_data = power_data[:max_points]
                
                # Store data
                channel.create_dataset("power", data=power_data)
                channel.create_dataset("timestamps", data=timestamps)
    
    return sample_file

if __name__ == "__main__":
    file_path = create_synthetic_ukdale()
    print(f"Created synthetic UK-DALE dataset at: {file_path}")
