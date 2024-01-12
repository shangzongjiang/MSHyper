# MSHyper
This repo is the official implementation for Multi-Scale Adaptive Graph Neural Network for Multivariate Time Series Forecasting.

## 1.1 The framework of MSHyper
![framework](https://github.com/shangzongjiang/MSHyper/blob/main/fig/figure1_last7.pdf)
# 2 Prerequisites

* Python 3.8.5
* PyTorch 1.13.1
* math, sklearn, numpy, torch_geometric
* # 3 Datasets
To evaluate the performance of H2former, we conduct experiments on five public benchmark datasets： [Weather](https://www.bgc-jena.mpg.de/wetter/), [ETT](https://github.com/MAZiqing/FEDformer), [Traffic](http://pems.dot.ca.gov/), and [Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014).
## 3.1 Weather
This dataset contains 21 meteorological measurements data form the Weather Station of the Max Planck Biogeochemistry, which are sampled every 10 minutes.
## 3.2 Traffic
This dataset contains the road occupancy rates of 862 sensors in San Francisco Bay Area freeways, which are sampled hourly.
## 3.3 Electricity
This dataset contains the electricity consumption of 321 clients from the UCI Machine Learning Repository, which are sampled hourly.
## 3.5 ETT
This dataset contains the oil temperature and load data collected by electricity transformers, including ETTh and ETTm, which are sampled hourly and every 15 minutes respectively
# 4 Running
## 4.1 Install all dependencies listed in prerequisites

## 4.2 Download the dataset

## 4.3 Training
```python
# Train on Weather
python train.py -data weather -input_size 168 -predict_step 168 -root_path ./data/ETT/ -data_path weather.csv -CSCM Conv_Construct
# Train on Electricity
python train.py -data elect -input_size 168 -predict_step 168 -root_path ./data/Electricity/ -data_path electricity.csv -CSCM Conv_Construct
# Train on ETTh1
python train.py -data ETTh1 -input_size 168 -predict_step 168 -root_path ./data/ETT/ -data_path ETTh1.csv -CSCM Conv_Construct
# Train on ETTm1
python train.py -data ETTm1 -input_size 168 -predict_step 168 -root_path ./data/ETT/ -data_path ETTm1.csv -CSCM Conv_Construct
# Train on Traffic
python train.py -data traffic -input_size 168 -predict_step 168 -root_path ./data/Traffic/ -data_path traffic.csv -CSCM Conv_Construct
```
# 5 Main results
![main_results](https://github.com/shangzongjiang/MSHyper/blob/main/fig/table1.pdf)(https://github.com/shangzongjiang/MSHyper/blob/main/fig/table2.pdf)
