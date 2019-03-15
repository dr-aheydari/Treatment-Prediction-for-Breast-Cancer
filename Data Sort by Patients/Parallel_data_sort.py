#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:14:59 2019

@author: aliheydari
"""

""" 
parallel computing in Python
Instance: Splitting the data with a parallel code

"""

import multiprocessing as mp
from numba import autojit, prange
import re
import numpy as np
import pandas as pd


# Parallel Processing 
@autojit
def SortData(data,array_length,patient_sorted):
    """
     So my logic is that for each patient, it will go through the data
    if anywhere in each row, any string matches the string literal, then it
    writes it to the "pat_sorted" file.
    """
    
    for i in range(0, array_length):
        
        # I was stuck becasue the "line" would not reset! So we need this 
        data.seek(0);
        
        for line in data:
        
            # if it matches the string anywhre in the line
            if re.match('(.*){}(.*)'.format(patients[i]), line):
        
                    patient_sorted.write(line)
                    
                  
    return patient_sorted
                



print("Number of processors: ", mp.cpu_count())

# open all the files
data = open("NC_BRCA.maf", "r+")
patient_sorted = open("patients_sorted.maf", "w+")
patient_ids = open( "PatientIDs.txt", "r+" )
patients = []

# to remove the end of the line characters to it becomes the same as the data
for line in patient_ids:
        line = line.rstrip("\n")
        patients.append(line)

    
    

# get patient array size
array_length = len(patients)
print(array_length);

# call the function
SortData(data,array_length,patient_sorted)

