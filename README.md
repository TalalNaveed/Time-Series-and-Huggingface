# Time Series Forecasting with Transformers

This repository contains a time series forecasting project using a Transformer model implemented in PyTorch and Hugging Face's Trainer API.

---

## Dataset

- **Source:** [UCI Household Power Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)  
- **Description:** The dataset contains **over 2 million rows** of electricity consumption measurements (global active power) from a single household, recorded every minute.  
- **Preprocessing:**  
  - Missing values filled with forward-fill.  
  - Data resampled to **hourly frequency**.  
  - Sliding window of **24 hours** used for forecasting the next hour.

---

## Model

- **Architecture:** Transformer-based model for time series forecasting.  
- **Details:**  
  - Input: last 24 hours of global active power.  
  - Transformer Encoder with 3 layers and 8 attention heads.  
  - Output: predicted global active power for the next hour.  
  - Loss: Mean Squared Error (MSE).

- **Why Transformer:** Captures long-term dependencies better than traditional RNN or CNN architectures, as discussed in "A comprehensive survey of deep learning for time series forecasting: architectural diversity and open challenges."

---

## Training

- **Framework:** Hugging Face Trainer API.  
- **Training/Validation Split:** 90% train, 10% test.  
- **Hyperparameters:**  
  - Learning rate: 1e-4  
  - Batch size: 64  
  - Epochs: 3  

---

## Results

- **Evaluation Metric:** MSE (~0.267 on test set)  
- **Visualization:**  

  <img width="547" height="413" alt="image" src="https://github.com/user-attachments/assets/20797142-f905-40d3-8d0b-cccf22dcd31d" />


- The model predictions follow the overall trend of electricity consumption with some underestimation of spikes.


