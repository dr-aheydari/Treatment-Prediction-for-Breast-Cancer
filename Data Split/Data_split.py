#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:16:01 2019

@author: aliheydari
"""

"""

This script modifies our file and splits into (almost) equal size files for
the training data, test and the forbidden test data"

"""

import fileinput
import numpy as np
import pandas as pd




# defining a fucntion that removes the comments since maftools does not like it
def rmComments(inpFile, outFile):

    inpfile = open(inpFile, "r+")
    outfile = open(outFile, "w+")

    outfile.write(inpfile.readline())

    for line in inpfile:
        if not line.lstrip().startswith("#"):
            outfile.write(line)

    inpfile.close()
    outfile.close()
    
    
    
    
def DataSplit(inpFile):
    
    # get the count for the number of lines in the file
    count = len(open(inpFile).readlines(  ));
    print("Total line numbers: {}".format(count));
        
    inpfile = open(inpFile, "r+")
    
    with open("NC_BRCA.maf") as f:
        first_line = f.readline()
      
        
    lines_per_file = int(count/3);
    print("Count/3 : {}".format(lines_per_file))
#    lines_per_file = 300
    smallfile = None
    with open('NC_BRCA.maf') as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                small_filename = 'small_BRCA_{}.maf'.format(lineno + lines_per_file)
                print(small_filename);
                smallfile = open(small_filename, "w")
#                line_pre_adder(small_filename, first_line);   
                
             
            smallfile.write(line)
        
        if smallfile:
            smallfile.close()
            
            

    inpfile.close()
#    outfile1.close()
    
    
def line_pre_adder(filename, line_to_prepend):
    
#    with open(filename) as f:
#        first_line = f.readline()
    print("oh shit")
    f = fileinput.input(filename, inplace=1)
    for xline in f:
        if f.isfirstline():
            print (line_to_prepend.rstrip('\r\n') + '\n' + xline)
#            else
#            print xline,

    

# Now call the function with the name of the files
    # new.maf -> input
    # new2.maf -> output
    
#rmComments("BRCA.maf","NC_BRCA.maf");
#
#DataSplit("NC_BRCA.maf")
#
#with open("NC_BRCA.maf") as f:
#    first_line = f.readline()

#line_pre_adder("small_BRCA_44298.maf", first_line);
#line_pre_adder("small_BRCA_88596.maf", first_line);
#line_pre_adder("small_BRCA_132894.maf", first_line);


## now split the file evenly (ish)

# check the three files have the same first line
    
with open("NC_BRCA.maf") as f:
    first_line = f.readline()

with open("small_BRCA_44298.maf") as f:
    first_line1 = f.readline()
    
with open("small_BRCA_88596.maf") as f:
    first_line2 = f.readline()
    
with open("small_BRCA_88596.maf") as f:
    first_line3 = f.readline()

if first_line3 == first_line and first_line2 == first_line and first_line1 == first_line:
    print("All three splitted files now have the same header : )" );


# ALL Done
