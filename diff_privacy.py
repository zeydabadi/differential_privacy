#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:11:25 2018

@author: Mahmoud Zeydabadinezhad
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

def diff_privacy(File, Ep):
    """
    Inputs: 
        - File: Path to the CSV file
        - ep:   The value of epsilon
    
    Outputs: 
        - The original histogram and error plot.
    """
    print("User picked epsilon is: ", Ep)
    #print(type(Ep))
    df = pd.read_csv(File)
    MinAge = df.Age.min()
    MaxAge = df.Age.max()
    Bins = range(MinAge,MaxAge+1,5)
    counts, bins = np.histogram(df.Age, bins=Bins)
    #print(counts)
    #print(bins)
    def laplaceMechanism(x, epsilon):
        x +=  np.random.laplace(0, epsilon)
        return x
    
    laplaceMechanism_v = np.vectorize(laplaceMechanism)
    err_list  = []
    new_count = []
    user_counts = laplaceMechanism_v(counts,1/float(Ep)) # Noisy histogram counts calculated with user's epsilon
    #print(len(user_counts))
    user_error  = np.mean(np.abs(counts - user_counts)) # mean absolute error calculated for user's epsilon
    print("Differential privacy error for the user's epsilon is:", user_error)
    
    EP = np.linspace(0.1,1,10)
    #print(len(Bins))
    for ep in EP:
        countsPrime = laplaceMechanism_v(counts,1/ep)
        new_count.append(countsPrime)
        error = counts - countsPrime
        err_list.append(np.mean(np.abs(error)))
        
        
    fig, ax = plt.subplots(1, 3, figsize=(20, 18))
    axes = ax.flatten()
    axes[0].hist(df.Age, bins=Bins)
    axes[0].set_title('Age histogram',fontsize='large')
    axes[0].set_ylabel('Number of people', color='blue')
    axes[0].set_xlabel('Age', color='blue')
    axes[0].grid(True)
    axes[0].legend(loc='upper right')
    
    axes[1].set_title('Histogram count after adding noise',fontsize='large')
    axes[1].bar(Bins[1::],user_counts, 10, color="red")
    axes[1].set_ylabel('Number of people', color='blue')
    axes[1].set_xlabel('Age', color='blue')
    axes[1].grid(True)
    
    axes[2].set_title('Error plot',fontsize='large')
    axes[2].plot(EP,err_list)
    axes[2].set_ylabel('Absolute error')
    axes[2].set_xlabel('epsilon')
    axes[2].set_xticks(EP)
    axes[2].grid(True)
    plt.show()
    
if __name__== "__main__":
    if len(sys.argv) == 3:
        diff_privacy(sys.argv[1],sys.argv[2])
    else:
        sys.exit("\nUsage: diff_privacy.py path_to_CSV_file epsiln\n\n\n\n")
