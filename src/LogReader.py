#!/usr/bin/env python 
'''
A simple log reader for the provided files
format: x y z node_id node_label [features]
'''
from Point import Point

class LogReader:
    
    def __init__(self, path):
        self._path = path
        self._points = []
        
    def read(self):
        #clear previous points
        self._points = []
        
        with open(self._path) as f:
            for line in f:
                if line[0] == '#':
                    continue
                entries = line.split()
                if len(entries) < 1:
                    continue
                
                pt = Point(float(entries[0]), float(entries[1]), entries[2], Point.label_dict[entries[3]] ,[float(entry) for entry in entries[5:]])
                
                self._points.append(pt)
                
        return self._points