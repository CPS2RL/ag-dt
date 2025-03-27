import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import random

class FaultInjector:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def inject_synchronized_random_faults(self, columns: List[str], start_idx: int, end_idx: int,
                                          density: float = 0.3,
                                          intensities: List[float] = [-1.5, 1.5],
                                          min_interval: int = 5 
                                          ) -> pd.DataFrame:
        df_subset = self.df.iloc[start_idx:end_idx].copy()
        num_rows = end_idx - start_idx
        target_fault_count = int(num_rows * density)
        fault_indices = []
        available_indices = set(range(num_rows))

        while len(fault_indices) < target_fault_count and available_indices:
            if not available_indices:
                break

            idx = random.choice(list(available_indices))
            fault_indices.append(idx)

            for i in range(max(0, idx - min_interval), min(num_rows, idx + min_interval + 1)):
                available_indices.discard(i)

        for column in columns:
            values = df_subset[column].values
            for idx in fault_indices:
                intensity = random.choice(intensities)
                values[idx] = values[idx] * (1 + intensity)
            df_subset[column] = values

        # Create fault labels
        labels = ['clean'] * num_rows
        for idx in fault_indices:
            labels[idx] = 'random'
        df_subset['Class'] = labels

        return df_subset

    
    def inject_synchronized_malfunction_fault(self, columns: List[str], start_idx: int, end_idx: int,
                                          noise_intensity: float = 4.5
                                          ) -> pd.DataFrame:
        df_subset = self.df.iloc[start_idx:end_idx].copy()
        variances = {col: np.var(df_subset[col].values) for col in columns}
        
        for column in columns:
            values = df_subset[column].values
            variance = variances[column]
            for i in range(len(values)):
                noise = np.random.normal(0, np.sqrt(variance)) * noise_intensity
                values[i] = values[i] + noise
            df_subset[column] = values
            
        df_subset['Class'] = 'malfunction'
        
        return df_subset
    
    def inject_synchronized_drift_fault(self, columns: List[str], start_idx: int, end_idx: int,
                                    intensities: List[float] = [-4,4],
                                    noise_intensity: float = 1.25
                                    ) -> pd.DataFrame:
        df_subset = self.df.iloc[start_idx:end_idx].copy()
        intensity = random.choice(intensities)
        for column in columns:
            values = df_subset[column].values
            variance = np.var(values)
            offset = values[0] * intensity
            for i in range(len(values)):
                noise = np.random.normal(1, 3*np.sqrt(variance)) * noise_intensity
                values[i] = values[i] + noise + offset
            df_subset[column] = values
        df_subset['Class'] = 'drift'
        return df_subset

    def inject_synchronized_bias_fault(self, columns: List[str], start_idx: int, end_idx: int,
                                   noise_intensity: float = 3.0,
                                   intensities: List[float] = [2]
                                   ) -> pd.DataFrame:
        df_subset = self.df.iloc[start_idx:end_idx].copy()
        intensity = intensities[0]

        for column in columns:
            values = df_subset[column].values
            original_mean = np.mean(values)
            new_mean = original_mean * intensity

            for i in range(len(values)):
                values[i] = new_mean

            df_subset[column] = values

        df_subset['Class'] = 'bias'
        return df_subset

    def inject_all_faults(self, config: Dict) -> pd.DataFrame:

        result_df = self.df.copy()
        result_df['Class'] = 'clean'

        # Handle all fault types
        for fault_type, fault_config in config.items():
            for interval in fault_config['intervals']:
                start_idx = interval['start_idx']
                end_idx = interval['end_idx']

                if fault_type == 'random':
                    fault_df = self.inject_synchronized_random_faults(
                        fault_config['columns'],
                        start_idx, end_idx
                    )
                elif fault_type == 'malfunction':
                    fault_df = self.inject_synchronized_malfunction_fault(
                        fault_config['columns'],
                        start_idx, end_idx
                    )
                elif fault_type == 'drift':
                    fault_df = self.inject_synchronized_drift_fault(
                        fault_config['columns'],
                        start_idx, end_idx
                    )
                elif fault_type == 'bias':
                    fault_df = self.inject_synchronized_bias_fault(
                        fault_config['columns'],
                        start_idx, end_idx
                    )

                interval_mask = (result_df.iloc[start_idx:end_idx]['Class'] == 'clean').values

                for column in fault_config['columns']:
                    current_values = result_df.iloc[start_idx:end_idx, result_df.columns.get_loc(column)].values
                    fault_values = fault_df[column].values

                    current_values[interval_mask] = fault_values[interval_mask]
                    result_df.iloc[start_idx:end_idx, result_df.columns.get_loc(column)] = current_values

                # Update class labels
                current_labels = result_df.iloc[start_idx:end_idx, result_df.columns.get_loc('Class')].values
                fault_labels = fault_df['Class'].values
                current_labels[interval_mask] = fault_labels[interval_mask]
                result_df.iloc[start_idx:end_idx, result_df.columns.get_loc('Class')] = current_labels

        return result_df




    def generate_fault_intervals(self, total_rows: int, fault_percentage,
                               min_interval_size: int = 70, max_interval_size: int = 120) -> Dict:

        target_fault_rows = int(total_rows * fault_percentage)
        remaining_rows = target_fault_rows
        available_indices = set(range(total_rows))
        intervals = []

        # Distribute faults among different types including random
        fault_type_distribution = {
            'random': 0.5,     
            'malfunction': 0.2, 
            'drift': 0.15,      
            'bias': 0.15         
        }

        fault_intervals = {
            'random': [],
            'malfunction': [],
            'drift': [],
            'bias': []
        }

        while remaining_rows > 0 and available_indices:
            interval_size = min(
                random.randint(min_interval_size, max_interval_size),
                remaining_rows
            )

            valid_start_indices = []
            for i in range(total_rows - interval_size + 1):
                if all(j in available_indices for j in range(i, i + interval_size)):
                    valid_start_indices.append(i)

            if not valid_start_indices:
                break

            start_idx = random.choice(valid_start_indices)
            end_idx = start_idx + interval_size

            for i in range(start_idx, end_idx):
                available_indices.remove(i)

            intervals.append({'start_idx': start_idx, 'end_idx': end_idx})
            remaining_rows -= interval_size

        random.shuffle(intervals)
        total_faulty_rows = sum(interval['end_idx'] - interval['start_idx'] for interval in intervals)

        current_position = 0
        for fault_type, proportion in fault_type_distribution.items():
            target_rows = int(total_faulty_rows * proportion)
            accumulated_rows = 0

            while current_position < len(intervals) and accumulated_rows < target_rows:
                interval = intervals[current_position]
                interval_size = interval['end_idx'] - interval['start_idx']
                fault_intervals[fault_type].append(interval)
                accumulated_rows += interval_size
                current_position += 1

        return fault_intervals

    def inject_mixed_faults(self, columns_config: Dict[str, List[str]],
                           fault_percentage) -> pd.DataFrame:

        total_rows = len(self.df)
        fault_intervals = self.generate_fault_intervals(total_rows, fault_percentage)

        config = {
            'random': {
                'columns': columns_config['random'],
                'intervals': fault_intervals['random']
            },
            'malfunction': {
                'columns': columns_config['malfunction'],
                'intervals': fault_intervals['malfunction']
            },
            'drift': {
                'columns': columns_config['drift'],
                'intervals': fault_intervals['drift']
            },
            'bias': {
                'columns': columns_config['bias'],
                'intervals': fault_intervals['bias']
            }
        }

        return self.inject_all_faults(config)

def main():
    df = pd.read_csv('all_quincy-train.csv')

    fault_columns = df.drop(columns=['timestamp_utc', 'datetime'])

    columns_config = {'random': fault_columns,'malfunction': fault_columns,'drift': fault_columns,'bias': fault_columns}

    injector = FaultInjector(df)

    result_df = injector.inject_mixed_faults(
        columns_config,
        fault_percentage=0.1
    )

    result_df.to_csv('quincy_faulty.csv', index=False)



    fault_distribution = result_df['Class'].value_counts()
    print("\nFault Distribution:")
    for fault_type, count in fault_distribution.items():
        print(f"  {fault_type}: {count} rows ({count/len(result_df)*100:.2f}%)")


    faulty_rows = len(result_df[result_df['Class'] != 'clean'])
    total_rows = len(result_df)
    print(f"\nOverall fault percentage: {(faulty_rows/total_rows)*100:.2f}%")

if __name__ == "__main__":
    main()