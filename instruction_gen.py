import numpy as np
import torch
import pandas as pd
import random


def DecimalToBinary(num, num_bits): 
    lst = [] 
    while num: 
        # if (num & 1) = 1 
        if (num & 1): 
            lst.append(1)
        # if (num & 1) = 0 
        else: 
            lst.append(0)
        # right shift by 1 
        num >>= 1

    if len(lst) < num_bits:
        lst.extend([0 for _ in range(num_bits - len(lst))])
    return reverse(lst) 
  
# function to reverse the string 
def reverse(lst): 
    return lst[::-1]

ops_R = {"add": [0,0,0, 0,0,0], "sub": [0,0,0, 0,1,0]}

def instruction_gen_R(operation, num1, num2, num_bits, ops = ops_R): # convert inputs into an binary-encoded R-type instruction
    # define a unique operation by funct3 + first 3 bits of funct7
    inst_bits = list(ops[operation])

    # convert each num into its binary form of max_length 
    if num1 >= pow(2,num_bits) or num2 >= pow(2, num_bits):
        print("num1 or num2 out of bounds for given num_bits")
        return
    
    inst_bits.extend(DecimalToBinary(num1, num_bits))
    inst_bits.extend(DecimalToBinary(num2, num_bits))

    return inst_bits

def result_gen_R(operation, num1, num2, num_bits, ops=ops_R):
    if operation not in ops:
        print("inValid  operation")
        return
    
    if operation == "add":
        return DecimalToBinary(num1 + num2, num_bits)

    else:
        return DecimalToBinary(num1 - num2, num_bits)
    
   
num1 = 5
num2 = 15
num_bits = 4
#print(instruction_gen_R("add", num1, num2, num_bits))

# create a Dataset class that generates data from deterministic random numbers (set a seed)
def gen_R(operation, num_bits, dataset_len, ops=ops_R):
    # generate random numbers deterministically:
    
    inst_data = []
    res_data = []
    random.seed(42)

    for _ in range(dataset_len):
        num1 = random.randint(0, pow(2,num_bits) - 1)
        num2 = random.randint(0, pow(2, num_bits) - 1)

        inst_bits = instruction_gen_R(operation, num1, num2, num_bits, ops)
        # u need num_bits + 1 bits to fully represent the addition of two num_bit size ints without overflow
        result_bits = result_gen_R(operation, num1, num2, num_bits + 1, ops)


        inst_data.append(inst_bits) 
        res_data.append(result_bits)

    return inst_data, res_data

class R_Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.inst = [inst_bits for inst_bits in df["inst_bits"]]
        self.results = [res_bits for res_bits in df["result_bits"]]
        
    
    def __len__(self):
        return len(self.results)


    def __getitem__(self, idx):

        inst = self.inst[idx]
        res = self.results[idx]

        return inst, res