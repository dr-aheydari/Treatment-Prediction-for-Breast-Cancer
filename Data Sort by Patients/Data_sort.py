#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:01:13 2019

@author: aliheydari
"""
from __future__ import print_function
import re
import numpy as np
import pandas as pd


data = open("NC_BRCA.maf", "r+")
patient_sorted = open("11_sorted.maf", "w+")
patient_ids = open( "PatientIDs.txt", "r+" )
patients = []

#with patient_ids as fh:
for line in patient_ids:
        line = line.rstrip("\n")
        patients.append(line)

    

#
#for i in range (0, 40):
#    print("Patient {} has an ID of : {}".format(i, patients[i]))
    
#print(len(patients))
#for i in range(0,1099):

counter = 0
#for i in patients:



    

#        print(i);

for i in range(0, len(patients)):
        
   for line in data:
           
        if re.match('(.*){}(.*)'.format(patients[i]), line):
        
            patient_sorted.write(line)

        
