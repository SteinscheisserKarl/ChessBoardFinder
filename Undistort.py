#!/usr/bin/python

'''
Created on 2020-01-24

@author: Thomas Klaube
'''
import cv2
import numpy as np


# specific calibration result of my camera for videos/images of size 480x480! This is not usable
# for any other cam or any other resolution!
CameraMatrix = np.array([[1.49152680e+03, 0.00000000e+00, 3.20334458e+02],
 [0.00000000e+00, 1.49276657e+03, 2.29837124e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
DistCoefficients = np.array([-6.92919302e-01, -1.19690798e+00, 8.27444231e-03, -1.27203119e-03, 2.27170669e+01])

# Distance for Cam is ~221 cm from the camera lens to the table 

def UndistortImage(img):

    undistorted = cv2.undistort(img, CameraMatrix, DistCoefficients,newCameraMatrix=CameraMatrix)
    return undistorted
