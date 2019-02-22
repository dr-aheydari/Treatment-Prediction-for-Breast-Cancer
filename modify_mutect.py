#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:38:45 2019

@author: aliheydari
"""

##### This script modifies our file tgat we have for the mutations so that the 
# maftools can read it without a problem : ) ########


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
    

# Now call the function with the name of the files
    # new.maf -> input
    # new2.maf -> output
rmComments("new.maf","new2.maf");

