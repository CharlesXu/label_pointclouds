#!/usr/bin/env python 
'''
A simple class for defining Points
'''
import numpy as np

class Point:
    
    label_dict = {'1004':0, '1100':1, '1103':2, '1200':3, '1400':4} 
        
    def __init__(self, x=0.0, y=0.0, z=0.0, label=0, feature=[]):
        self._x = x 
        self._y = y
        self._z = z
        self._label = label
        self._feature = np.array(feature)
    