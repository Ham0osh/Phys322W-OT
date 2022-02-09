import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import scipy.interpolate as interpolate


def local_max(arr, N = 2, strict = False):
    '''find local maximums of an array where local is defined as M points on either side, if strict is true
    then it will follow this process exactly if strict is false it will also count local maxes that are at least
    one space from the edge if they satisfy the requirement within the remaining array'''
    local_maxs = []
    M = int(N/2)
    
    #loop through the array
    if not strict:
        i = 1

    else:
        i = M

    indexes = []
    
    while i < len(arr) - 1:
        
        iterate = 1
        #flag
        local_max = True
        
        for j in range(M):
            try:
                #will make index error when your with M of the edges so except index error
                if arr[i] < arr[i + j]:
                    local_max = False
                    iterate = j
                    break

            except IndexError:
                if strict:
                    #reproduce old behaviour
                    local_max = False
                    break
                #continue on to looking in the opposite direction
                else:
                    pass
            
            try:
                if arr[i] < arr[i - j]:
                    local_max = False
                    break

            except IndexError:
                if strict:
                    local_max = False
                    break
                else:
                    pass

            
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


def noise(func, sigma = 0):
    '''adds gaussian noise with standard deviation sigma to any function'''
    def noisy_func(*args, **kwargs):
        modified = func(*args, **kwargs)
        modified += np.random.normal(scale = sigma, size = len(modified))
        return modified
    return noisy_func
