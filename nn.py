import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from instruction_gen import *
import sklearn
from sklearn.model_selection import train_test_split


# build simple fully-connected nn with hidden size >= 2 * input size
class Simple_1L_NNModel(nn.Module):

    def __init__(self, in_features, hidden_size, out_features):
        
        # TODO: NOTE THAT OUT_FEATURES = INFEATURES + 1 FOR R TYPE INSTRUCTIONS TO PREVENT OVERFLOW!

        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)

    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class Simple_2L_NNModel(nn.Module):

    def __init__(self, in_features, hidden_size, out_features):
        
        # TODO: NOTE THAT OUT_FEATURES = INFEATURES + 1 FOR R TYPE INSTRUCTIONS TO PREVENT OVERFLOW!

        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_features)


    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# TODO: NOTE THAT OUT_FEATURES = IN_FEATURES + 1 FOR R TYPE INSTRUCTIONS TO PREVENT OVERFLOW!

num_bits = 3
dataset_len = 1024
hidden_size = 32
model = Simple_1L_NNModel(num_bits, hidden_size, num_bits+1)

df_R = df_gen_R("add", num_bits, dataset_len)

print(len(df_R["inst_bits"][0]))
print()
print(df_R.head())

# Train, Test, Split
X, y = df_R["inst_bits"], df_R["result_bits"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# concert everything to tensors (for you, shouldnt matter if int or float)
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)

# choose criterion and optimizer (NOTE: THESE ARE PROLLY NOT OPTIMAL)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# train the model!
from torch.optim import Adam
from tqdm import tqdm

def train(model, train_data, val_data, learning_rate, epochs):

    train, val = R_Dataset(train_data), R_Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True) # used a RandomSampler(dataset, **) rather than a SequentialSampler(dataset, *...*)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda: 

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_input = train_input.to(device)
                train_label = train_label.to(device)


                output = model(train_input)
                print(output, train_label)
                batch_loss = criterion(output, train_label.long()) #.long() converts the train_labels into a numpy.int64
                total_loss_train += batch_loss.item()
                
                #acc = (output.argmax(dim=1) == train_label).sum().item() 
                #total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_input = train_input.to(device)
                    val_label = train_label.to(device)
                    output = model(val_input)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    #acc = (output.argmax(dim=1) == val_label).sum().item()
                    #total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')
                  
    
train(model, df_train, df_val, LR, EPOCHS)