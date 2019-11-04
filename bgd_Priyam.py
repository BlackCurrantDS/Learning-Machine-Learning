# coding: utf-8

# In[198]:


import argparse, sys
import pandas as pd
import numpy as np
from numpy import genfromtxt
import os

def CalulateError(X, y, weight):
    e = np.power(((X @ weight.T) - y), 2) 
    return np.sum(e) 



def compute_gd(X,y,weight,l,stop_when):
    prev_error = 0
    i=0
    
   

    error = CalulateError(X, y, weight)
    results = []
    while abs(prev_error-error) > stop_when:
        
        prev_error = error
        weight = weight + l * np.sum(( y-X @ weight.T) * X, axis=0)
        error = np.sum(np.power(((X @ weight.T) - y), 2) )
        results.append((i+1, list(np.round(weight,decimals=4).tolist()[0]),round(error,4)))
        i = i+1
        
    
    return results 
    
def b_gd(threshold,data,learningRate):
    data_given = np.genfromtxt(data, delimiter=',') 
    X = data_given[:, :-1] 
    matrix_x_ones = np.ones([X.shape[0], 1]) 
    X = np.concatenate([matrix_x_ones, X],1) 
    y = data_given[:, -1].reshape(-1,1) 
    n = X.shape[1]
    weight = np.zeros((1, n))
    l = learningRate
    stop_when = threshold
    initial_error =  CalulateError(X, y, weight)
    new_data = compute_gd(X,y,weight,l,stop_when)
   
    df = pd.DataFrame(data=new_data)
    
   
    
    initial_row = pd.DataFrame({0: '0', 1 : [list(weight.tolist()[0])], 2: initial_error }, index = [0])
    
    df = pd.concat([initial_row, df]).reset_index(drop = True)
    df = pd.concat([df[0],df[1].apply(pd.Series), df[2]], axis = 1)
    
    df.to_csv("bgd_out_Priyam.csv",header=False, index=False)
    
    return df
    
if __name__ == "__main__":
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument('--data', required=True,help='input file')
    parser.add_argument('--learningRate', required=True,help='Foo the program')
    parser.add_argument('--threshold', required=True,help='Foo the program')

    args=parser.parse_args()
    data = args.data
    #file_name = os.path.basename(data)

    b_gd(float(args.threshold),data,float(args.learningRate))

