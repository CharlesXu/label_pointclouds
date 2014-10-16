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
            print '[LogReader] Opened file '+self._path
            for line in f:
                if line[0] == '#':
                    continue
                entries = line.split()
                if len(entries) < 1:
                    continue
                
                pt = Point(float(entries[0]), float(entries[1]), float(entries[2]), Point.label_dict[entries[4]] ,[float(entry) for entry in entries[5:]])
                
                self._points.append(pt)
        print '[LogReader] '+str(len(self._points))+' points loaded'
        return self._points