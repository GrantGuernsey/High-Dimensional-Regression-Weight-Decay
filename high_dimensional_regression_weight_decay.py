# Referenced from DL6A.py, CS5173/6073, cheng, 2024
"""
This script performs high-dimensional linear regression using PyTorch, with an option to apply weight decay.
The script generates synthetic data for training and validation, then trains a linear regression model 
with different weight decay values to observe the effect on overfitting.
Attributes:
    num_train (int): Number of training samples.
    num_val (int): Number of validation samples.
    num_inputs (int): Number of input features.
    max_epochs (int): Number of training epochs.
    n (int): Total number of samples (training + validation).
    X (torch.Tensor): Input features tensor.
    noise (torch.Tensor): Noise tensor added to the output.
    w (torch.Tensor): Initial weights tensor.
    b (float): Initial bias.
    y (torch.Tensor): Output tensor with noise.
    Xtrain (torch.Tensor): Training input features.
    ytrain (torch.Tensor): Training output.
    Xval (torch.Tensor): Validation input features.
    yval (torch.Tensor): Validation output.
    all_loss_v (list): List to store validation losses for different weight decays.
    all_loss_t (list): List to store training losses for different weight decays.
    diff (list): List to store the difference between validation and training losses.
"""

import torch

# Constants
NUM_TRAIN = 20
NUM_VAL = 100
NUM_INPUTS = 200
MAX_EPOCHS = 10
LEARNING_RATE = 0.02
WEIGHT_DECAY_RANGE = 21

# Generate synthetic data
def generate_data(num_train, num_val, num_inputs):
    n = num_train + num_val
    X = torch.randn(n, num_inputs)
    noise = torch.randn(n, 1) * 0.01
    w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    y = torch.matmul(X, w) + b + noise
    return X[:num_train], y[:num_train], X[num_train:], y[num_train:]

# Train the model and return losses
def train_model(Xtrain, ytrain, Xval, yval, weight_decay, max_epochs, learning_rate):
    model = torch.nn.Linear(NUM_INPUTS, 1)
    loss_fun = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    for epoch in range(max_epochs):
        model.train()
        y_pred_t = model(Xtrain)
        loss_t = loss_fun(y_pred_t, ytrain)
        
        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_pred_v = model(Xval)
            loss_v = loss_fun(y_pred_v, yval)
        
        print(f"Epoch {epoch+1}/{max_epochs} - Training Loss: {loss_t.item():.4f}, Validation Loss: {loss_v.item():.4f}")
    
    return loss_t.item(), loss_v.item()

# Main function
def main():
    Xtrain, ytrain, Xval, yval = generate_data(NUM_TRAIN, NUM_VAL, NUM_INPUTS)
    all_loss_v = []
    all_loss_t = []
    diff = []
    
    for weight_decay in range(WEIGHT_DECAY_RANGE):
        print(f"\nWeight Decay: {weight_decay}")
        loss_t, loss_v = train_model(Xtrain, ytrain, Xval, yval, weight_decay, MAX_EPOCHS, LEARNING_RATE)
        all_loss_v.append(loss_v)
        all_loss_t.append(loss_t)
        diff.append(loss_v - loss_t)
    
    best_index = diff.index(min(diff))
    print(f"\nBest weight decay index: {best_index}")
    print(f"Training Loss: {all_loss_t[best_index]:.4f}, Validation Loss: {all_loss_v[best_index]:.4f}")
    print(f"No weight decay - Training Loss: {all_loss_t[0]:.4f}, Validation Loss: {all_loss_v[0]:.4f}")

if __name__ == "__main__":
    main()