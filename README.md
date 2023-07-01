# H2former
This repo is the official implementation for Multi-Scale Adaptive Graph Neural Network for Multivariate Time Series Forecasting.

## 1.1 The framework of H2former
![framework](https://github.com/shangzongjiang/H2former/fig/main.pdf)
# 2 Prerequisites

* Python 3.8.5
* PyTorch 1.13.1
* math, sklearn, numpy, torch_geometric
* # 3 Datasets
To evaluate the performance of H2former, we conduct experiments on six public benchmark datasets：ETT, Weather, Traffic, Electricity, and Exchange-Rate.
## 3.1 Weather
This dataset contains 21 meteorological measurements data form the Weather Station of the Max Planck Biogeochemistry, which are sampled every 10 minutes.
## 3.2 Traffic
This dataset contains the road occupancy rates of 862 sensors in San Francisco Bay Area freeways, which are sampled hourly.
## 3.3 Electricity
This dataset contains the electricity consumption of 321 clients from the UCI Machine Learning Repository, which are sampled hourly.
## 3.4 Exchange-Rate
This dataset contains the exchange rate date from 8 foreign countries, which are sampled daily.
## 3.5 ETT
This dataset contains the oil temperature and load data collected by electricity transformers, including ETTh and ETTm, which are sampled hourly and every 15 minutes respectively
# 4 Running
## 4.1 Install all dependencies listed in prerequisites

## 4.2 Download the dataset

## 4.3 Training
```python
# Train on Weather
python long_range_main.py -data weather -input_size 168 -predict_step 168 -root_path ./data/ETT/ -data_path weather.csv -CSCM Conv_Construct
# Train on Electricity
python train.py -data elect -input_size 168 -predict_step 168 -root_path ./data/Electricity/ -data_path electricity.csv -CSCM Conv_Construct
# Train on ETTh1
python train.py -data ETTh1 -input_size 168 -predict_step 168 -root_path ./data/ETT/ -data_path ETTh1.csv -CSCM Conv_Construct
# Train on ETTm1
python train.py -data ETTm1 -input_size 168 -predict_step 168 -root_path ./data/ETT/ -data_path ETTm1.csv -CSCM Conv_Construct
# Train on Traffic
python train.py -data traffic -input_size 168 -predict_step 168 -root_path ./data/Traffic/ -data_path traffic.csv -CSCM Conv_Construct
# Train on Exchange-Rate
python train.py -data exchange_rate -input_size 168 -predict_step 168 -root_path ./data/exchange/ -data_path exchange_rate.csv -CSCM Conv_Construct
```