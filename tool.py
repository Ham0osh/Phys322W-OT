import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import scipy.constants as cnst
from matplotlib import cm
from matplotlib.collections import LineCollection
from dataclasses import dataclass
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
                
            try:
                if arr[i] < arr[i - j]:
                    local_max = False
                    break

            except IndexError:
                if strict:
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


def noise(func, sigma = 0):
    '''adds gaussian noise with standard deviation sigma to any function'''
    def noisy_func(*args, **kwargs):
        modified = func(*args, **kwargs)
        modified += np.random.normal(scale = sigma, size = len(modified))
        return modified
    return noisy_func

def circle(thetas,radius = 1):
    x = radius*np.cos(thetas)
    y = radius*np.sin(thetas)
    return x,y


def outliers(x_arr,y_arr, auto = True, max_movement = 1):
    '''takes a an array of x and y data and returns the index of any outliers, where
    outliers is defined as a shift in position greater than the max_movement per time step'''
    outlier_arr = []
    copy_x = list(x_arr)
    copy_y = list(y_arr)
    dist_arr = ((x_arr[1:] - x_arr[:-1])**2 + (y_arr[1:] - y_arr[:-1])**2)**(1/2)

    if auto:
        max_movement = 5*np.std(dist_arr)

    i = 1
    while i < len(copy_x):
        dx = (copy_x[i] - copy_x[i-1])
        dy = (copy_y[i] - copy_y[i-1])
        if np.sqrt(dx**2 + dy**2) > max_movement:
            outlier_arr.append(i + len(outlier_arr))
            copy_x.pop(i)
            copy_y.pop(i)
        else:
            i += 1

    return np.array(copy_x), np.array(copy_y), np.array(outlier_arr)


def gaussian(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mu)/sigma)**2)

@dataclass
class gaussain_method_analysis:
    '''class for storing the analysis from histogram projection function'''
    pOpt_x: np.array = np.array([])
    pCov_x: np.array = np.array([]) 
    pOpt_y: np.array = np.array([])
    pCov_y: np.array = np.array([])
    bins_x: np.array = np.array([])
    bins_y: np.array = np.array([])
    hist_x: np.array = np.array([]) 
    hist_y: np.array = np.array([])
    k_x: float = 0
    k_y: float = 0
    
def get_k_equipartition(x,y):
    '''
    Input: x and y data [um] (preferably decimated)
    Output: k_x, k_y in [pN/um]
    Assumes boltzman constant and T is approx 298 K = 25 C
    '''
    k_b = cnst.Boltzmann # m^2 kg s-2 K-1
    T = 298 # K
    var_x = np.var(x/(10**6)) #[m^2]
    var_y = np.var(y/(10**6)) #[m^2]
    k_x = k_b*T/(var_x) #[N/m]
    k_y = k_b*T/(var_y) #[N/m]
    return k_x*10**6, k_y*10**6 # in [pN/um]

def gaussian_analysis(x,y, nbins = 10, p0 = [0,0.05]):
    n, bins = np.histogram(y, bins = nbins)
    normalizing_factor = np.sum(n)*abs(bins[0] - bins[1])
    error = np.sqrt(n)
    error = np.where(error == 0, 1, error)

    norm_err = error/normalizing_factor
    n_y, bins = np.histogram(y, bins = nbins, density = True)
    bins_y = (bins[:-1] + bins[1:])/2

    dx = np.linspace(min(bins)*1.2, max(bins)*1.2, 100)
    pOpt_y, pCov_y = curve_fit(gaussian, bins_y, n_y, p0 = p0, sigma = norm_err, absolute_sigma=True)

    temp = 298
    sigma = pOpt_y[1]/10**6
    vari = sigma**2
    k_y = cnst.Boltzmann*temp/vari*10**6

    ### x analysis starts here
    n, bins = np.histogram(x, bins = nbins)
    normalizing_factor = np.sum(n)*abs(bins[0] - bins[1])
    error = np.sqrt(n)
    error = np.where(error == 0, 1, error)


    n_x, bins = np.histogram(x, bins = nbins, density=True)
    #get centre position of the bins
    bins_x = (bins[:-1] + bins[1:])/2

    norm_err = error/normalizing_factor

    dx = np.linspace(min(bins)*1.2, max(bins)*1.2, 100)
    pOpt_x, pCov_x = curve_fit(gaussian, bins_x, n_x, p0 = [0,0.05], sigma = norm_err, absolute_sigma=True) # units are in um
    

    sigma = pOpt_x[1]/10**6
    vari = sigma**2
    k_x = cnst.Boltzmann*temp/vari*10**6

    ret = gaussain_method_analysis(pCov_x = pCov_x, pOpt_x = pOpt_x, pCov_y = pCov_y, pOpt_y = pOpt_y, bins_x = bins_x, bins_y = bins_y,
        hist_x = n_x, hist_y = n_y, k_x = k_x, k_y = k_y)

    
    return ret


def make_histogram_projection(x,y, cmap = 'viridis', nbins = 10,printBool = True, plot = True, p0_y = [0,0.05],
    p0_x = [0,0.05]):
    '''takes x and y data and makes the histogram projection as well as analysis
    of fits using a gaussian function'''
    # start with a square Figure

    fig = plt.figure(figsize=(8, 8))

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)


    ax = fig.add_subplot(gs[1, 0])

    #no idea how this formatting works
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1) 

    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(np.linspace(0,1,len(x)))
    line = ax.add_collection(lc)
    

    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax_histy,label = "passage of time")
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    n, bins = np.histogram(y, bins = nbins)
    normalizing_factor = np.sum(n)*abs(bins[0] - bins[1])
    error = np.sqrt(n)
    #sometimes the bins have nothing in them and dividing by zero will give error so we have
    #to replace the zeroes with something I chose 1 for no particular reason
    error = np.where(error == 0, 1, error)


    n_y, bins, _ = ax_histy.hist(y, bins = nbins, density=True, orientation = 'horizontal')
    #get centre position of the bins
    bins_y = (bins[:-1] + bins[1:])/2



    norm_err = error/normalizing_factor
    ax_histy.errorbar(n_y, bins_y, xerr = norm_err, fmt = 'k.', capsize = 3, ms = 1)


    dx = np.linspace(min(bins)*1.2, max(bins)*1.2, 100)
    pOpt_y, pCov_y = curve_fit(gaussian, bins_y, n_y, p0 = p0_y, sigma = norm_err, absolute_sigma=True)
    ax_histy.plot(gaussian(dx, *pOpt_y), dx)

    temp = 298
    sigma = pOpt_y[1]/10**6
    vari = sigma**2
    k_y = cnst.Boltzmann*temp/vari*10**6
    if printBool:
        print(f"for y:\nmean = {pOpt_y[0]}\nsigma = {sigma*10**6}")
        print(f"variance = {vari*10**12}")
        print(f"k = {k_y} [pN/um]")


    ##### this is where the x analysis begins

    n, bins = np.histogram(x, bins = nbins)
    normalizing_factor = np.sum(n)*abs(bins[0] - bins[1])
    error = np.sqrt(n)
    error = np.where(error == 0, 1, error)


    n_x, bins, _ = ax_histx.hist(x, bins = nbins, density=True)
    #get centre position of the bins
    bins_x = (bins[:-1] + bins[1:])/2

    norm_err = error/normalizing_factor
    ax_histx.errorbar(bins_x, n_x, yerr = norm_err, fmt = 'k.', capsize = 3, ms = 1)

    dx = np.linspace(min(bins)*1.2, max(bins)*1.2, 100)
    pOpt_x, pCov_x = curve_fit(gaussian, bins_x, n_x, p0 = p0_x, sigma = norm_err, absolute_sigma=True) # units are in um
    ax_histx.plot(dx, gaussian(dx, *pOpt_x))

    sigma = pOpt_x[1]/10**6
    vari = sigma**2
    k_x = cnst.Boltzmann*temp/vari*10**6
    if printBool:
        print(f"for x:\nmean = {pOpt_x[0]}\nsigma = {sigma*10**6}")
        print(f"variance = {vari*10**12}")
        print(f"k = {k_x} [pN/um]")


    ret = gaussain_method_analysis(pCov_x = pCov_x, pOpt_x = pOpt_x, pCov_y = pCov_y, pOpt_y = pOpt_y, bins_x = bins_x, bins_y = bins_y,
        hist_x = n_x, hist_y = n_y, k_x = k_x, k_y = k_y)

    
    return ret

@dataclass
class PositionData:
    raw_x: np.array = np.array([])
    raw_y: np.array = np.array([])
    rad_px: np.array = np.array([])
    x_px: np.array = np.array([])
    y_px: np.array = np.array([])
    x_dat: np.array = np.array([])
    y_dat: np.array = np.array([])
    x: np.array = np.array([])
    y: np.array = np.array([])
    x_dec: np.array = np.array([])
    y_dec: np.array = np.array([])

def extract_data(*args,interval = 5, um_per_px = 1/23.68, **kwargs):
    kwargs['unpack'] = True
    rad_px, raw_x, raw_y = np.genfromtxt(*args, **kwargs)
    x_px, y_px, idx = outliers(raw_x, raw_y)
    if len(idx) != 0:
        print(f'{len(idx)} outliers found')
    rad_px = np.array([rad_px[i] for i in range(len(x_px)) if i not in idx])

    rad_dat = rad_px*um_per_px
    x_dat = x_px*um_per_px
    y_dat = y_px*um_per_px
    x_ave = np.mean(x_dat)
    y_ave = np.mean(y_dat)

    rad_arr = rad_dat
    x = x_dat-x_ave
    y = y_dat-y_ave
    x_dec = x[::interval]
    y_dec = y[::interval]
    
    ret = PositionData(raw_x = raw_x, raw_y = raw_y, rad_px = rad_px, x_px = x_px, y_px = y_px, x_dat = x_dat,
        y_dat = y_dat, x = x, y = y, x_dec = x_dec, y_dec = y_dec)
    return ret