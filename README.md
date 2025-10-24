# Leveraging: Replication Data

This repository contains the replication material for the paper "Leveraging Temporal Patterns in Forecasting".

## Overview
The repository includes the necessary scripts and data to replicate the results presented in the paper. The primary script, `main.py` trains the models and obtains the predictions.

## Requirements
- **Python version:** 3.8.5
- Required libraries: Install dependencies using:
  ```bash
  pip install -r requirements.txt

## Running the Model
To reproduce the results, execute the following command in your terminal:
```bash
python main.py
```
Then run main_random.py to obtain results with random cluster assignments. 
```bash
python main_random.py
```
The results are produces by running
```bash
python results.py
```

## Expected Runtime
The script should take approximately 5 days to run.

## Directory Structure
- main.py: Main script.
- main_random.py: Trains models with random clusters.
- results.py: Produces the findings discussed in the paper. 
- functions.py and functions_deep_learning.py: Functions needed to run the model. 
- Datasets/: Contains input data required.
- Results/: Stores the predictions and generated images.
