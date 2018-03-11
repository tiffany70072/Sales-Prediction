# Sales-Prediction

## Introduction
### Data after preprocess
* 499 devices
433 products
103 weeks 

| File name | shape | content |
| --------- | ----- | ------- |
| exist.npy | 499 * 433 * 103 tensor | 1, if sales data for product exists on this device in this week; 0, no data here |
| amount.npy | 499 * 433 * 103 tensor | real sales amount |
| exist_first.npy | 499 * 433 * 103 tensor | 1, if data for product was sold on this device for the first time; 0, other situations |
| price.npy | 499 * 433 * 103 tensor | real price |
| number.npy | 499 * 433 * 103 tensor | real sales count, = amount / price |

### Evaluation
    For test_id = 55 to 100: 
        training_data = data[1:test_id-1] 
        testing_data = data[test_id]
        train_model...
        MAE[test_id] = mean(absolute_error_of_testing_data) 
        MSE[test_id] = mean(square_error_of_testing_data)) 
    MAE = sum(MAE[test_id] * number_of_data[test_id]) 
    MSE = sum(MSE[test_id] * number_of_data[test_id]) 
    
## Model
### Tensor factorzaion
* 3 dimensional tensor: device * product * time
* Three regularization terms <br />
    * Weight regularization <br />
    * Temporal regularization: the prediction of two adjacent weeks should be similar. <br />
    * Non-negative term: the prediction should not be negative. <br />
* File name: tensor_factorization.py

### Binary classifier
* classifier's structure: <br />
  x1 = neural_network(input quantitive features) <br />
  x2 = neural_network(input one-hot features) <br />
  prediction = neural_network(concatenate(x1, x2))
* Combination: <br />
1. Use classifier to predict the sold out probability is zero or not <br />
2. If the output is not zero, then use tensor factorizaion to predict the value <br />
* File name: classifier.py <br />
  File name: prediction_two_steps.py (used for combination)

### (New) Matrix factorization (with all first-time product) 
* Filter: <br />
Only use the data, if the product appears on a device for the first time, as real value. <br />
Without considering which week of these data.
* 2 dimensional matrix: device * product (for the first time product in all weeks)
* File name: mf_factorization_machine.py

### Factorization machine
* Add other features into matrix factorization. <br />
Example: the sales amount of the same product when it was sold on another device for the first time. <br />
Example: the sales amount of other products when they were sold on the same device for the first time. <br />
* File name: mf_factorization_machine.py

## Baseline model
### Matrix factorization (with weighted sum of sales amount)
* Totally different with the (new) matrix factorization. <br />
Weighted sum the past sales amount as the real value in testing week. <br />
* 2 dimensional matrix: device * product (in the last week)
* Four different weighted sum method, including exponential moving average. <br />
There is a coefficient, alpha, in each last three weighted sum method. <br />
They are set as 0.4, 1.0, 2.0.
* File name: matrix_factorization.py

### Modified_FPMC
* Change the last two layers in FPMC so this algorithm can predict value instead of probability or rank.
* 3 dimensional tensor: device, item last week, item this week
* File name: modified_FPMC.py


