# <div align="center"> MSHyper: Multi-Scale Hypergraph Transformer for Long-Range Time Series Forecasting
✨ This repo is the official implementation of Multi-Scale Hypergraph Transformer for Long-Range Time Series Forecasting.

# 1 The framework of MSHyper
The core of MSHyper is to build multi-scale hypergraph structures, which can explicity model high-order interactions between temporal patterns of different scales. MSHyper consists of three parts: **Multi-Scale Feature Extraction (MFE) Module**, **Hypergraph and Hyperedge Graph Construction (H-HGC) Module**, and **Tri-Stage Message Passing (TMP) Mechanism**. The overall framework of MSHyper is shown as follows:
![framework](https://github.com/shangzongjiang/MSHyper/blob/main/figures/Framework.png)
# 2 Prerequisites

* Python 3.8.5
* PyTorch 1.13.1
* math, sklearn, numpy, torch_geometric
* # 3 Datasets && Description

To evaluate the performance of MSHyper, we conduct experiments on eight public benchmark datasets: [ETT(ETTh1, ETTh2, ETTm1, and ETTm2)](https://github.com/MAZiqing/FEDformer), [Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014), [Flight](https://drive.google.com/drive/folders/1JSZByfM0Ghat3g_D3a-puTZ2JsfebNWL), [Weather](https://www.bgc-jena.mpg.de/wetter/), and [Exchange-Rate]([http://pems.dot.ca.gov/](https://github.com/MAZiqing/FEDformer)). The detailed descriptions about the eight datasets are given as follows:
* ![dataset-statistics](https://github.com/shangzongjiang/MSHyper/blob/main/figures/dataset%20statistics.png)

## 3.1 ETT(ETTh1, ETTh2, ETTm1, and ETTm2)
This dataset contains the oil temperature and load data collected by electricity transformers, including ETTh and ETTm, which are sampled hourly and every 15 minutes, respectively.
## 3.2 Electricity
This dataset contains the electricity consumption of 321 clients from the UCI Machine Learning Repository, which are sampled hourly.
## 3.3 Flight
This dataset contains changes in flight data from 7 major European airports provided by OpenSky, which is sampled hourly.
## 3.4 Weather
This dataset contains 21 meteorological measurements data form the Weather Station of the Max Planck Biogeochemistry, which are sampled every 10 minutes.
## 3.5 Exchange-Rate
This dataset contains the exchange-rate data from 8 foreign countries, which is sampled daily.
📦 You can download the all datasets from [datasets](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download). **All the datasets are well pre-processed** and can be used directly.
# 4 Running
## 4.1 Install all dependencies listed in prerequisites

## 4.2 Download the dataset

## 4.3 Training
🚀 We provide the experiment scripts of MSHyper on all dataset under the folder `./scripts`. You can obtain the full results by running the following command:
```
# Train on ETTh1
sh ./scripts/Long-range/ETTh1.sh
# Train on ETTh2
sh ./scripts/Long-range/ETTh2.sh
# Train on ETTm1
sh ./scripts/Long-range/ETTm1.sh
# Train on ETTm2
sh ./scripts/Long-range/ETTm2.sh
# Train on Electricity
sh ./scripts/Long-range/electricity.sh
# Train on Flight
sh ./scripts/Long-range/flight.sh
# Train on Weather
sh ./scripts/Long-range/weather.sh

```
or obtain specific results by runinng the following command:
```python
# Train on Weather
python run_longExp.py -data weather -input_size 96 -predict_step 96 -root_path ./data/ETT/ -data_path weather.csv -CSCM Conv_Construct
# Train on Electricity
python run_longExp.py -data elect -input_size 96 -predict_step 96 -root_path ./data/Electricity/ -data_path electricity.csv -CSCM Conv_Construct
# Train on ETTh1
python run_longExp.py -data ETTh1 -input_size 96 -predict_step 96 -root_path ./data/ETT/ -data_path ETTh1.csv -CSCM Conv_Construct
# Train on ETTm1
python run_longExp.py -data ETTm1 -input_size 96 -predict_step 96 -root_path ./data/ETT/ -data_path ETTm1.csv -CSCM Conv_Construct
# Train on Traffic
python run_longExp.py -data traffic -input_size 96 -predict_step 96 -root_path ./data/Traffic/ -data_path traffic.csv -CSCM Conv_Construct
```
# 5 Main results
**🏆 MSHyper achieves consistent state-of-the-art performance on all benchmarks**, covering a large variety of series with different frequencies, variate numbers and real-world scenarios.
## 5.1 Multivariate long-range time series forecasting results
Multivariate long-range time series forecasting results on eight real-world datasets. The input length is set as I=96, and the prediction length O is set as 96, 192, 336, and 720. The best results are bolded and the second best results are underlined.
![Multivariate](https://github.com/shangzongjiang/MSHyper/blob/main/figures/Multivariate%20Result.png) 
## 5.2 Univariate long-range time series forecasting results
Univariate long-range time series forecasting results on ETT dataset. The input length is set as I=96, and the prediction length O is set as 96, 192, 336, and 720. The best results are bolded and the second best results are underlined.
![Univariate](https://github.com/shangzongjiang/MSHyper/blob/main/figures/Univariate%20Results.png)
# Citation 
😀If you find this repo useful, please cite our paper.
```
@article{shang2024mshyper,
  title={Mshyper: Multi-scale hypergraph transformer for long-range time series forecasting},
  author={Zongjiang, Shang and Ling, Chen},
  journal={arXiv preprint arXiv:2401.09261},
  year={2024}
}
```

