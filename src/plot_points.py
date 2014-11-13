#!/usr/bin/env python 
'''
A script to plot a bunch of points
'''
from mayavi import mlab
from LogReader import LogReader
import numpy as np

def plot_points(points):
    print '[plot_points] Plotting points!'
    xs = np.array([int(point._x) for point in points])
    ys = np.array([int(point._y) for point in points])
    zs = np.array([int(point._z) for point in points])
    labels = np.array([int(point._label) for point in points])
    mlab.points3d(xs, ys, zs, labels, scale_factor = 0.4, mode='cube')
    mlab.show()
    
def plot_predicted_labels(points, labels):
    print '[plot_points] Plotting points!'
    xs = np.array([int(point._x) for point in points])
    ys = np.array([int(point._y) for point in points])
    zs = np.array([int(point._z) for point in points])
    mlab.points3d(xs, ys, zs, labels, scale_factor = .4, mode='cube')
    mlab.show()

if __name__ == "__main__":
    
    #Load a log
    log_object = LogReader('../data/oakland_part3_am_rf.node_features')
    points = log_object.read()
    
    plot_points(points)