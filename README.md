## Inconsistency Classification
Data inconsistency is the most challenging for any kind of data analysis. It can alter any kind of decision making process. To address this challege, we have simulated several sensor faults (Random, Malfunction, Drift, Bias) to simulate inconsistecy in our weather time series data. Then we classfiy the inconsistencies using various deep learning tecniques. Use `generation.py` under data\Inconsistency Generation to introduce inconsistencies in the dataset.


## Models We are using
In the code, we have implemented 9 different machine learning models. The models are CNN-based (TCN, ResNet), RNN-based (LSTM, Bi-LSTM, GRU), Transformer-based (TST, Informer) and Hybrid (TST-AE, LSTM-AE). All of are use for time-series inconsistency classification and fruit surface temperature prediction with, without faults and after data imputation. 

## Fruit Surface Temperature as A Case Study
Measuring FST is cruical for agricultural systmes to ensure efficient farming and resource usage. For exapmles, apples can be exposed to excessive heat from solar radiation. In order to maintain a specific growth of agricultural products maintaining the FST becomes a key factor for decision making. FST can be measured from weather attributes like air temperature, dew point, solar radiation and wind speed using energy balance model and using these state-of-the art deep learning models, we can analyze and predict the FST in real-time. It can help farmers to monitor crop health and contribute to their decision making process.

Select an appropriate model for a dataset and configure `train.py` to train the model.  
Configure `main.py` to evaluate the trained model.  
Run `docker-compose up --build` in the command prompt to measure system performance in terms of inference time and memory usage in a containerized environment.  
Use **Case Study-1** for predicting weather attributes.  
Use **Case Study-2** for behavioral prediction analysis with and without inconsistent data.  
Run `dt.py` to connect to a weather station through the API (Zentra Cloud in this case) and get output from the trained models.  

