# High-Dimensional Linear Regression with Weight Decay

This Python script performs linear regression using PyTorch in a high-dimensional setting, demonstrating the effect of weight decay on overfitting. The script generates synthetic data and trains a linear regression model while evaluating the performance on both training and validation sets with various weight decay values.

## Features
- **High-Dimensional Linear Regression**: Trains a linear regression model with hundreds of input features.
- **Weight Decay Regularization**: Investigates how different weight decay values influence the training and validation loss to prevent overfitting.
- **Synthetic Data Generation**: Simulates noisy linear data for model training and validation.
- **Loss Tracking**: Computes and stores the training and validation losses for each weight decay value.

## Requirements
- `torch`: For deep learning and tensor operations.

## Script Overview

### Parameters:
- `NUM_TRAIN`: Number of training samples (default: 20).
- `NUM_VAL`: Number of validation samples (default: 100).
- `NUM_INPUTS`: Number of input features (default: 200).
- `MAX_EPOCHS`: Maximum number of training epochs (default: 10).
- `LEARNING_RATE`: Learning rate for the optimizer (default: 0.02).
- `WEIGHT_DECAY_RANGE`: Number of weight decay values to test (default: 21).

### Functions:
1. **`generate_data(num_train, num_val, num_inputs)`**:
   - Generates synthetic data with `num_inputs` features, returning training and validation sets.
   
2. **`train_model(Xtrain, ytrain, Xval, yval, weight_decay, max_epochs, learning_rate)`**:
   - Trains a linear regression model using stochastic gradient descent (SGD) with a specified `weight_decay` and returns the training and validation losses.

3. **`main()`**:
   - Executes the data generation, model training, and evaluation over a range of weight decay values. Tracks the losses and identifies the best-performing weight decay value based on the difference between training and validation losses.

### Output:
- The script prints the training and validation losses for each epoch.
- After testing various weight decay values, it displays the best-performing weight decay value based on minimal loss difference and the associated losses for both training and validation.
  
## Usage
Run the script as follows:

```bash
python high_dimensional_regression.py
```

You can adjust constants like the number of training samples, validation samples, input features, and learning rate directly in the script.

## License
This project is licensed under the MIT License.
