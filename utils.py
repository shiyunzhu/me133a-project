#!/usr/bin/env python3
import numpy as np

def Rx(theta):
    return np.array([[ 1, 0            , 0            ],
                     [ 0, np.cos(theta),-np.sin(theta)],
                     [ 0, np.sin(theta), np.cos(theta)]])

def Ry(theta):
    return np.array([[ np.cos(theta), 0, np.sin(theta)],
                     [ 0            , 1, 0            ],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rz(theta):
    return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                     [ np.sin(theta), np.cos(theta) , 0 ],
                     [ 0            , 0             , 1 ]])

def get_error_p(x, p):
    return x - p

def get_error_r(R, R_d):
    return np.array(0.5 * (np.cross(R[:, 0], R_d[:, 0])
    + np.cross(R[:, 1], R_d[:, 1])
    + np.cross(R[:, 2], R_d[:, 2]))).reshape(3, 1)

def damped_pseudo_inverse(J, gamma):
    N = len(J[0]) # number of columns
    Jt = np.transpose(J)
    return np.linalg.inv(Jt @ J + gamma ** 2 * np.identity(N)) @ Jt
