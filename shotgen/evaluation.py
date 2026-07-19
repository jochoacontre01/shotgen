import numpy as np
from scipy.ndimage import gaussian_filter

def signal_to_noise_ratio(arr, ref_arr=None):
    """
    Calculate the Signal-to-Noise Ratio (SNR).
    
    If ref_arr is provided:
        Computes the power-based SNR in decibels (dB) between the noisy array `arr`
        and the clean reference array `ref_arr`.
        SNR = 10 * log10( sum(ref_arr**2) / sum((ref_arr - arr)**2) )
        
    If ref_arr is None:
        Estimates SNR of the single array `arr` by defining high-amplitude regions
        as signal and low-amplitude regions as background/noise.
        SNR = 10 * log10( mean(signal_region**2) / mean(noise_region**2) )
        where signal region is |arr| >= 70th percentile, and noise region is |arr| <= 30th percentile.
        
    Parameters
    ----------
    arr : ndarray
        The array to evaluate.
    ref_arr : ndarray, optional
        The clean reference array. Default is None.
        
    Returns
    -------
    float
        The SNR value in decibels (dB).
    """
    arr = np.asanyarray(arr, dtype=np.float64)
    if ref_arr is not None:
        ref_arr = np.asanyarray(ref_arr, dtype=np.float64)
        if arr.shape != ref_arr.shape:
            raise ValueError(f"Shape mismatch: arr shape {arr.shape} != ref_arr shape {ref_arr.shape}")
        
        signal_power = np.mean(ref_arr**2)
        noise_power = np.mean((ref_arr - arr)**2)
        if noise_power == 0:
            return float('inf')
        if signal_power == 0:
            return -float('inf')
        return 10.0 * np.log10(signal_power / noise_power)
    else:
        abs_arr = np.abs(arr)
        p70 = np.percentile(abs_arr, 70)
        p30 = np.percentile(abs_arr, 30)
        
        signal_region = arr[abs_arr >= p70]
        noise_region = arr[abs_arr <= p30]
        
        signal_power = np.mean(signal_region**2) if signal_region.size > 0 else 0.0
        noise_power = np.mean(noise_region**2) if noise_region.size > 0 else 0.0
        
        if noise_power == 0:
            return float('inf')
        if signal_power == 0:
            return -float('inf')
        return 10.0 * np.log10(signal_power / noise_power)

def contrast_to_noise_ratio(arr, signal_mask=None, background_mask=None, ref_arr=None):
    """
    Calculate the Contrast-to-Noise Ratio (CNR).
    
    If ref_arr is provided, we compute the CNR comparing `arr` to `ref_arr` using standard deviation:
        CNR = std(ref_arr) / std(ref_arr - arr)
    Otherwise, we compute the CNR of `arr` itself.
        CNR = |mean(signal) - mean(background)| / std(background)
        
    If masks are not provided:
        Signal region is estimated as elements with absolute value >= 70th percentile.
        Background/noise region is estimated as elements with absolute value <= 30th percentile.
        
    Parameters
    ----------
    arr : ndarray
        The array to evaluate.
    signal_mask : ndarray of bool, optional
        Boolean mask of the signal region. Default is None.
    background_mask : ndarray of bool, optional
        Boolean mask of the background region. Default is None.
    ref_arr : ndarray, optional
        The reference array to compare against. Default is None.
        
    Returns
    -------
    float
        The CNR value.
    """
    arr = np.asanyarray(arr, dtype=np.float64)
    
    if ref_arr is not None:
        ref_arr = np.asanyarray(ref_arr, dtype=np.float64)
        if arr.shape != ref_arr.shape:
            raise ValueError(f"Shape mismatch: arr shape {arr.shape} != ref_arr shape {ref_arr.shape}")
        
        contrast = np.std(ref_arr)
        noise = np.std(ref_arr - arr)
        if noise == 0:
            return float('inf')
        return contrast / noise

    if signal_mask is None:
        abs_arr = np.abs(arr)
        p70 = np.percentile(abs_arr, 70)
        signal_mask = abs_arr >= p70
        
    if background_mask is None:
        abs_arr = np.abs(arr)
        p30 = np.percentile(abs_arr, 30)
        background_mask = abs_arr <= p30
        
    signal_data = arr[signal_mask]
    background_data = arr[background_mask]
    
    if signal_data.size == 0 or background_data.size == 0:
        return 0.0
        
    mu_s = np.mean(signal_data)
    mu_b = np.mean(background_data)
    sigma_b = np.std(background_data)
    
    if sigma_b == 0:
        sigma_s = np.std(signal_data)
        denom = np.sqrt(sigma_s**2 + sigma_b**2)
        if denom == 0:
            return 0.0
        return np.abs(mu_s - mu_b) / denom
        
    return np.abs(mu_s - mu_b) / sigma_b

def structural_similarity_index(arr, ref_arr, sigma=1.5, data_range=None, K1=0.01, K2=0.03):
    """
    Calculate the Structural Similarity Index (SSIM) between two arrays.
    
    Parameters
    ----------
    arr : ndarray
        The test array.
    ref_arr : ndarray
        The reference array.
    sigma : float, optional
        Standard deviation of the Gaussian filter. Default is 1.5.
    data_range : float, optional
        Dynamic range of the input images. Default is max(ref_arr) - min(ref_arr).
    K1 : float, optional
        Constant for luminance stability. Default is 0.01.
    K2 : float, optional
        Constant for contrast stability. Default is 0.03.
        
    Returns
    -------
    float
        The mean SSIM index value.
    """
    arr = np.asanyarray(arr, dtype=np.float64)
    ref_arr = np.asanyarray(ref_arr, dtype=np.float64)
    
    if arr.shape != ref_arr.shape:
        raise ValueError(f"Shape mismatch: arr shape {arr.shape} != ref_arr shape {ref_arr.shape}")
        
    if data_range is None:
        data_range = np.max(ref_arr) - np.min(ref_arr)
        if data_range == 0:
            data_range = 1.0
            
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
    mu1 = gaussian_filter(arr, sigma)
    mu2 = gaussian_filter(ref_arr, sigma)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = gaussian_filter(arr ** 2, sigma) - mu1_sq
    sigma2_sq = gaussian_filter(ref_arr ** 2, sigma) - mu2_sq
    sigma12 = gaussian_filter(arr * ref_arr, sigma) - mu1_mu2
    
    # Ensure variances are positive (avoiding negative values due to numerical precision)
    sigma1_sq = np.maximum(sigma1_sq, 0.0)
    sigma2_sq = np.maximum(sigma2_sq, 0.0)
    
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    return np.mean(ssim_map)
