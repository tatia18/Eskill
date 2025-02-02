import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your data (time_rel and velocity columns)
df = pd.read_csv('/Users/tatiatsiklauri/Desktop/space_apps_2024_seismic_detection/data/lunar/test/data/S12_GradeB/xa.s12.00.mhz.1969-12-16HR00_evid00006.csv')

# STA/LTA Parameters
short_window = 50   # Short-term window size (in data points)
long_window = 500   # Long-term window size (in data points)
threshold_on = 4.0  # Detection threshold (for spike onset)
threshold_off = 1.5 # End of event threshold (for spike offset)

# Calculate the STA/LTA characteristic function
def classic_sta_lta(velocity, short_window, long_window):
    sta = np.cumsum(velocity ** 2)
    sta[short_window:] = sta[short_window:] - sta[:-short_window]
    lta = np.cumsum(velocity ** 2)
    lta[long_window:] = lta[long_window:] - lta[:-long_window]
    
    # Avoid division by zero
    lta[lta == 0] = 1e-10
    sta_lta_ratio = sta[long_window - 1:] / lta[long_window - 1:]
    
    return sta_lta_ratio

# Apply STA/LTA to the velocity data
velocity = df['velocity(m/s)'].values
sta_lta_ratio = classic_sta_lta(velocity, short_window, long_window)

# Detect seismic events based on the STA/LTA ratio
seismic_event_onset = np.where(sta_lta_ratio > threshold_on)[0]
seismic_event_offset = np.where(sta_lta_ratio < threshold_off)[0]

# To avoid issues where offset is not found after onset, we'll ensure proper pairing
events = []
for onset in seismic_event_onset:
    # Find the first offset after the current onset
    offset_candidates = seismic_event_offset[seismic_event_offset > onset]
    if len(offset_candidates) > 0:
        offset = offset_candidates[0]
        events.append((onset, offset))

# Plot results
plt.figure(figsize=(12, 8))

# Plot velocity and highlight seismic events
plt.subplot(2, 1, 1)
plt.plot(df['time_rel(sec)'], velocity, label='Velocity (m/s)')
plt.title('Seismic Event Detection using STA/LTA')
plt.xlabel('Relative Time (s)')
plt.ylabel('Velocity (m/s)')

# Mark detected events on velocity plot in red
for event in events:
    plt.axvspan(df['time_rel(sec)'][event[0]], df['time_rel(sec)'][event[1]], color='red', alpha=0.5, label='Seismic Activity')

# Plot STA/LTA characteristic function
plt.subplot(2, 1, 2)
plt.plot(df['time_rel(sec)'][long_window-1:], sta_lta_ratio, label='STA/LTA Ratio', color='g')
plt.axhline(y=threshold_on, color='r', linestyle='--', label='Threshold On')
plt.axhline(y=threshold_off, color='b', linestyle='--', label='Threshold Off')
plt.title('STA/LTA Characteristic Function')
plt.xlabel('Relative Time (s)')
plt.ylabel('STA/LTA Ratio')
plt.legend()

plt.tight_layout()
plt.show()

# Extract the maximum velocity from the detected seismic events
max_velocity = max(abs(df['velocity(m/s)']))  # Taking absolute value of velocity

# Assume a constant C based on empirical values (adjust as needed for accuracy)
C = 5.5  # This constant should be calibrated for your region/data

# Estimate magnitude
magnitude = np.log10(max_velocity) + C

print(f"Estimated Magnitude: {magnitude}")

# Convert detected events to a DataFrame
detected_events = pd.DataFrame({
    'Event_Onset_Time(sec)': [df['time_rel(sec)'].iloc[event[0]] for event in events],
    'Event_Offset_Time(sec)': [df['time_rel(sec)'].iloc[event[1]] for event in events]
})

# Save detected events to a CSV file
csv_file_path = "/Users/tatiatsiklauri/Desktop/detected_seismic_events.csv"  # Adjust path as needed
detected_events.to_csv(csv_file_path, index=False)

print(f"Detected seismic events saved to {csv_file_path}")