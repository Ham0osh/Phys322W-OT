import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import scipy.interpolate as interpolate


def local_max(arr, N = 3):
    '''find local maximums of an array where local is defined as N points on either side'''
    local_maxs = []
    
    #loop through the array
    i = N
    indexes = []
    
    while i < len(arr) - 1 - N:
        
        iterate = 1
        #flag
        local_max = True
        
        for j in range(N):
            if arr[i] < arr[i + j]:
                local_max = False
                iterate = j
                break
                
            elif arr[i] < arr[i - j]:
                local_max = False
                break
            
        if local_max:
            local_maxs.append(arr[i])
            indexes.append(i)
            
        i += iterate
        
    return np.array(local_maxs), np.array(indexes)



def unique_maxs(y: np.array, N = 5, error_tol = 1e-3):
    '''finds the unique local maximums using local max and where unique is defined as different by the error_tol'''
    maxs, _ = np.sort(local_max(y, N = N))
    
    try:
        unique_maxs = [maxs[0]]
    except IndexError: 
        print('no maxs')
        return
        
    #remove repeats within a certain error tolerance
    for i in range(1,len(maxs)):
        if np.abs(maxs[i] - unique_maxs[-1]) > error_tol:
            unique_maxs.append(maxs[i])

    return unique_maxs
