import os

os.system("pip install easydict");
import easydict
import numpy as np
import torch
# below n is the number of loss functions you will have
# we assume you will have 3 

def set_hyper() : 
    
    args = easydict.EasyDict({

            # input layer (rows of the matrix)
            "input_size": 19163 ,
            # num of hidden nodes of the hid. layer
            "hidden_size":100 ,
            "num_classes" : 2,
            "lr_min" : 0.0001,
            "batch_size": 100,  
            # learning rate (ideally between 0.01 - 0.0001)
            "learning_rate": 0.05,
            ## ^^ this is something neat to fuck around with ^^ ##
            
            # an epoch is "the number of times entire dataset is trained"
            "num_epochs" : 1500

         })
    

    return args


def print_hyp(args):
    
    print("Input size : {} ".format(args.input_size));
    print("Hidden size : {} ".format(args.hidden_size));
    print("number of classes (outputs) : {} ".format(args.num_classes))
    print("Learning Rate (step sizes for GD) : {} ".format(args.learning_rate));
    print("number of epochs : {}".format(args.num_epochs))





