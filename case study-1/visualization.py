import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(file_path, start_idx=2000, end_idx=3000):
    df = pd.read_csv(file_path)
    feature_columns = [
        'Air Temperature', 'Dew Point', 'Rel. Hum', 'Atm. Press',
        'Solar Radiation', 'Wind Speed', 'Wind Gust'
    ]
    
    df_slice = df.iloc[start_idx:end_idx]
    
    plt.style.use('default')
    ACTUAL_COLOR = '#0000CC'
    PREDICT_COLOR = '#CC0000'
    GRID_COLOR = '#E6E6E6'
    BASE_FONT_SIZE = 46
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': BASE_FONT_SIZE,
        'axes.titlesize': BASE_FONT_SIZE,
        'axes.labelsize': BASE_FONT_SIZE,
        'xtick.labelsize': BASE_FONT_SIZE,
        'ytick.labelsize': BASE_FONT_SIZE,
        'legend.fontsize': int(BASE_FONT_SIZE * 0.83),
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': GRID_COLOR,
        'axes.axisbelow': True
    })
    
    fig = plt.figure(figsize=(30, 16))
    

    row1_bottom = 0.6
    row2_bottom = 0.005
    row_height = 0.38
    row1_width = 0.21
    row1_spacing = 0.1
    row1_left_margin = 0.05
    row2_width = 0.21
    row2_spacing = 0.10
    right_shift_offset = 0.13
    

    total_row2_width = 3 * row2_width + 2 * row2_spacing
    row2_left_margin = (1.0 - total_row2_width) / 2 + right_shift_offset
    
    row1_positions = [ [row1_left_margin + i * (row1_width + row1_spacing), row1_bottom, row1_width, row_height] for i in range(4) ]
    row2_positions = [ [row2_left_margin + i * (row2_width + row2_spacing), row2_bottom, row2_width, row_height] for i in range(3) ]
    all_positions = row1_positions + row2_positions
    
    axes = [fig.add_axes(pos) for pos in all_positions]
    
    y_labels = ['Temperature (°C)', 'Dew Point (°C)', 'Rel. Humidity (%)', 'Atm. Pressure (kPa)',
                'Solar Radiation (W/m²)', 'Wind Speed (m/s)', 'Wind Gust (m/s)']
    subplot_titles = ['(a) Air Temperature', '(b) Dew Point', '(c) Relative Humidity',
                      '(d) Atmospheric Pressure', '(e) Solar Radiation', '(f) Wind Speed', '(g) Wind Gust']
    custom_y_limits = [(-15, 30), (-15, 30), (20, 140), (98, 107), (-20, 620), (-3, 20), (-3, 20)]
    feature_tick_spacing = [15, 15, 40, 3, 200, 10, 10]
    
    for idx, feature in enumerate(feature_columns):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        ax.set_facecolor('white')
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, color=GRID_COLOR, alpha=0.3)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.25, color=GRID_COLOR, alpha=0.15)
        ax.minorticks_on()
        
        actual_data = df_slice[f'{feature}_Actual'].values
        predicted_data = df_slice[f'{feature}_Predicted'].values
        
        ax.plot(actual_data, color=ACTUAL_COLOR, linewidth=4, label='Actual', zorder=3)
        ax.plot(predicted_data, color=PREDICT_COLOR, linestyle='--', linewidth=4, label='Predicted', zorder=3)
        
        ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
        ax.set_xlim([0, len(df_slice)])
        ax.set_ylim(custom_y_limits[idx])
        ax.yaxis.set_major_locator(plt.MultipleLocator(feature_tick_spacing[idx]))
        ax.set_xlabel('Timestamp')
        ax.set_ylabel(y_labels[idx])
        ax.text(0.5, -0.33, subplot_titles[idx], transform=ax.transAxes, 
                fontsize=BASE_FONT_SIZE, ha='center', va='center')
    
    #plt.savefig('weather_predictions.pdf', dpi=3000, bbox_inches='tight')
    plt.show()