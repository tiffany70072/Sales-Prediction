# Sales-Prediction

## Introduction
### Data after preprocess:
| File name | shape | content |
| exist.npy | 499*433*103 tensor | 1, if sales data for product exists on this device in this week; 0, no data here |
| amount.npy | 499*433*103 tensor | real sales amount |
| exist_first.npy | 499*433*103 tensor | 1, if data for product was sold on this device for the first time' 0, for other situations |
| price.npy | 499*433*103 tensor | real price |
| number.npy | 499*433*103 tensor | real sales count, = amount / price |

### Evaluation
For test_id = 55 to 100:
    training data = data[1:test_id-1]
    testinf data = data[test_id]
    
## Model
### Tensor factorzaion
3 dimensional tensor: device * product * time
File name: tensor_factorization.py

### Binary classifier
File name: classifier.py
File name: prediction_two_steps.py

### (New) Matrix factorization (with all first-time product) 
Filter out all data, if they appear on a device for the first time, as real value.
File name: mf_factorization_machine.py

### Factorization machine
Add other features into matrix factorization.

File name: mf_factorization_machine.py


## Baseline model
### Matrix factorization (with weighted sum of sales amount)
Weighted sum the past sales amount as the real value in testing week.
Four different weighted sum method, including exponential moving average.
There is a coefficient, alpha, in each last three weighted sum method.
They are set as 0.4, 1.0, 2.0.

File name: matrix_factorization.py

### modified_FPMC
Change the last two layers in FPMC so this algorithm can predict value instead of probability.

File name: modified_FPMC.py


