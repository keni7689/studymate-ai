# Machine Learning Basics

## Gradient Descent
Gradient descent is an optimization algorithm that minimizes a loss function. It starts with random weights, computes the gradient of the loss, and moves the weights a small step in the opposite direction of the gradient. The size of each step is the learning rate. If the learning rate is too large the algorithm can overshoot the minimum; if it is too small, training becomes very slow. Think of walking downhill in fog: you feel the slope under your feet and always step downward.

## Overfitting
Overfitting happens when a model learns the training data too well, including its noise, and performs poorly on new data. Signs of overfitting are a very low training error but a high validation error. Common fixes include collecting more data, using regularization such as L2 or dropout, early stopping, and using a simpler model.

## Cross-Validation
Cross-validation splits the data into several folds. The model is trained on some folds and tested on the remaining fold, and this is repeated so every fold is used for testing once. It gives a more reliable estimate of model performance.
