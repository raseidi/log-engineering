# Event Feature Engineering

Repository for reproducing the [Empowering Predictive Process Monitoring through time-related Features](.) paper.

---

Content: 

- [Event Feature Engineering](#event-feature-engineering)
- [Setup](#setup)
- [Data Preprocessing](#data-preprocessing)
  - [1. Download benchmarked data](#1-download-benchmarked-data)
  - [2. Run script for preprocessing](#2-run-script-for-preprocessing)
  - [3. List of features](#3-list-of-features)
- [Training](#training)
  - [1. Tabular/shallow algorithms](#1-tabularshallow-algorithms)
  - [2. LSTM](#2-lstm)
- [Results](#results)
- [Citation](#citation)

---

# Setup

TODO
<!-- ```bash
pip install -r requirements.txt
```

or alternatively

```bash
conda install env.yml
``` -->

# Data Preprocessing

## 1. Download benchmarked data

Clone this [repository](https://github.com/hansweytjens/predictive-process-monitoring-benchmarks) and run the code for BPI12 and BPI20.

## 2. Run script for preprocessing

You can rather run the script to preprocess all the event logs at once by running

```bash
chmod +x create_datasets.sh
./create_datasets.sh
```

Or preprocess an individual event log

```python3
python3 create_dataset.py --dataset <dataset_name> 
```

## 3. List of features

| **id** 	|        **name**       	|  **Description**  	|
|:------:	|:---------------------:	|:-----------------:	|
|    0   	|          min          	|      np.min()     	|
|    1   	|          max          	|      np.max()     	|
|    2   	|          mean         	|     np.mean()     	|
|    3   	|         median        	|    np.median()    	|
|    4   	|          mode         	|    stats.mode()   	|
|    5   	|          std          	|      np.std()     	|
|    6   	|        variance       	|      np.var()     	|
|    7   	|           q1          	|  np.percentile()  	|
|    8   	|           q3          	|  np.percentile()  	|
|    9   	|          iqr          	|    stats.iqr()    	|
|   10   	|     geometric_mean    	|   stats.gmean()   	|
|   11   	|     geometric_std     	|    stats.gstd()   	|
|   12   	|     harmonic_mean     	|   stats.hmean()   	|
|   13   	|        skewness       	|    stats.skew()   	|
|   14   	|        kurtosis       	|  stats.kurtosis() 	|
|   15   	| coefficient_variation 	| stats.variation() 	|
|   16   	|        entropy        	|  stats.entropy()  	|
|   17   	|          hist         	|   np.histogram()  	|
|   18   	|     skewness_hist     	|    stats.skew()   	|
|   19   	|     kurtosis_hist     	|  stats.kurtosis() 	|
<!-- TODO: improve description -->

# Training

## 1. Tabular/shallow algorithms 

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

## 2. LSTM

TODO

# Results

TODO `results/`

# Citation

TODO