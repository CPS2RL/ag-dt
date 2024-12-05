## Requirements

Python>=3.8 or later

tensorflow>=2.8.0

numpy>=1.21.0

pandas>=1.3.0

scikit-learn>=0.24.0

matplotlib>=3.4.0

seaborn>=0.11.0

memory-profiler>=0.60.0

psutil>=5.8.0

## Inconsistency Classification
Data inconsistency is the most challenging for any kind of data analysis. It can alter any kind of decision making process. To address this challege, we have simulated several sensor faults (Random, Malfunction, Drift, Bias) to simulate inconsistecy in our weather time series data. Then we classfiy the inconsistencies using various deep learning tecniques.


In the code, we have implemented 9 different machine learning models. The models are CNN-based (TCN, ResNet), RNN-based (LSTM, Bi-LSTM, GRU), Transformer-based (TST, Informer) and Hybrid (TST-AE, LSTM-AE).
All of are use for time-series inconsistency classification and fruit surface temperature prediction with and without faults.  

