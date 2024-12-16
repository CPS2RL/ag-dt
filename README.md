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


## Models We are using
In the code, we have implemented 9 different machine learning models. The models are CNN-based (TCN, ResNet), RNN-based (LSTM, Bi-LSTM, GRU), Transformer-based (TST, Informer) and Hybrid (TST-AE, LSTM-AE). All of are use for time-series inconsistency classification and fruit surface temperature prediction with and without faults. 

## Fruit Surface Temperature as A Case Study
Measuring FST is cruical for agricultural systmes to ensure efficient farming and resource usage. For exapmles, apples can be exposed to excessive heat from solar radiation. In order to maintain a specific growth of agricultural products maintaining the FST becomes a key factor for decision making. FST can be measured from weather attributes like air temperature, dew point, solar radiation and wind speed using energy balance model and using these state-of-the art deep learning models, we can analyze and predict the FST in real-time. It can help farmers to monitor crop health and contribute to their decision making process.

## Digital Twin
Digital twin is one of the growing technology for automation and its simulation aspects can be help to create a virutal connection with the physical plants. It can enhance the decision-making process by integrating data driven analysis. We use this digital twin concept and try to connect it with our physical weather stations.

Select an appropriate model for a dataset and run `All_models_training.ipynb` to train a model.
Run `Inference Time and Memory.ipynb` for measuring system performance interms of inference time and memory usage.
Run `fst-prediction-all-models.ipynb` for any prediction behavior analysis with and without inconsistent data.
Run `DT_connection.ipynb` for connect a weather statition through API (for our case, Zentra Cloud) and get output from the trained models. 
