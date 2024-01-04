import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from instruction_gen import *
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def plot_predictions(train_data, 
                     train_labels,
                     test_data,
                     test_labels,
                     predictions=None):
  """
  Plots training data, test data and compares predictions. 
  """

  plt.figure(figsize=(10,7))
 
  #training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")

  #test data in green
  plt.scatter(test_data, test_labels, c="r", s=4, label="Testing Data")


  # Are there predictions?
  if predictions is not None:
    # Plot the predictions if they exist
    plt.scatter(test_data, predictions, c="g", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 4});


# build simple fully-connected nn with hidden size >= 2 * input size
class Simple_1L_NNModel(nn.Module):

    def __init__(self, in_features, hidden_size, out_features):
        
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)

    
    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        return self.fc2(x)
    

class Simple_2L_NNModel(nn.Module):

    def __init__(self, in_features, hidden_size, out_features):
        
        # TODO: NOTE THAT OUT_FEATURES = INFEATURES + 1 FOR R TYPE INSTRUCTIONS TO PREVENT OVERFLOW!

        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_features)


    
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.out(x)

inst_bits = 12 #inst bits + num1 (3 bits) + num2 (3 bits)
num_bits = 3 # 
dataset_len = 1024
hidden_size = 32
out_size = 4 # num_bits + 1
inst_data, res_data = gen_R("add", num_bits, dataset_len)

#Create an instance of the model, loss, and optimizer (this is a subclass of nn.Module)
#Set Random Seed
torch.manual_seed(42)
model_Simple = Simple_2L_NNModel(inst_bits, hidden_size, out_size)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_Simple.parameters(), lr=0.01)

# Create our train test split

X = torch.tensor(inst_data, dtype=torch.float32)
y = torch.tensor(res_data, dtype=torch.float32)

split = int(0.8 * len(X))
print(split)
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

print(X_train[:10])
print(X_train.dtype, y_train.dtype)

# epochs == one loop through the data... is a hyperparameter cause we set it ourselves
epochs = 1000

epoch_count = []
loss_values = []
test_loss_values = []

### Training
# 0. Loop through the data
for epoch in range(epochs):
    # Set the model to training mode - sets all params that require gradients to require gradients
    model_Simple.train()

    # 1. Forward Pass
    y_pred = model_Simple(X_train)
    print(y_pred)
    y_pred = torch.round(y_pred)
    # 2. Calculate Loss
    loss = loss_fn(y_pred, y_train) # (input, target)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backpropogation on the loss with repsect to the paramters of the model
    loss.backward()

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step() # by defailt the optimizer changes will accumulate over time ... so zero_grad() allows us to reset the gradients between each epoch


    ### Testing 
    model_Simple.eval() # turns off different settings not needed for evaling/testing
    with torch.inference_mode(): # turns off gradient tracking & a couple more things
        # 1. Do the forward pass
        test_pred = model_Simple(X_test)

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:

        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
    print(f"Epoch {epoch} | Loss: {loss} | Test loss: {test_loss}")

    


eval_sample = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32)
eval_output = model_Simple(eval_sample)
print(eval_output)






