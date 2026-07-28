import numpy as np
from scipy.ndimage import gaussian_filter


def signal_to_noise_ratio(clean_signal, noisy_signal):
    """
    Calculate the Signal-to-Noise Ratio (SNR) in decibels (dB).

    Parameters
    ----------
    clean_signal : np.ndarray
        The reference signal without noise.
    noisy_signal : np.ndarray
        The signal containing noise.

    Returns
    -------
    float
        The SNR in dB.
    """
    noise = noisy_signal - clean_signal
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def contrast_to_noise_ratio(image, feature_mask, background_mask):
    """
    Calculate the Contrast-to-Noise Ratio (CNR).

    Parameters
    ----------
    image : np.ndarray
        The 2D or 3D image array.
    feature_mask : np.ndarray
        Boolean mask identifying the feature region of interest.
    background_mask : np.ndarray
        Boolean mask identifying the background region.

    Returns
    -------
    float
        The CNR value.
    """
    mu_feature = np.mean(image[feature_mask])
    mu_bg = np.mean(image[background_mask])
    std_bg = np.std(image[background_mask])
    if std_bg == 0:
        return float('inf')
    return np.abs(mu_feature - mu_bg) / std_bg


def structural_similarity_index(img1, img2, data_range=None, win_size=7, K1=0.01, K2=0.03):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.

    Parameters
    ----------
    img1 : np.ndarray
        First image array.
    img2 : np.ndarray
        Second image array.
    data_range : float, optional
        Dynamic range of pixel values. If None, computed from max-min of img1.

    Returns
    -------
    float
        Mean SSIM index across the image.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    if data_range is None:
        data_range = img1.max() - img1.min()

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    sigma = win_size / 6.0
    mu1 = gaussian_filter(img1.astype(float), sigma)
    mu2 = gaussian_filter(img2.astype(float), sigma)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1.astype(float) ** 2, sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2.astype(float) ** 2, sigma) - mu2_sq
    sigma12 = gaussian_filter(img1.astype(float) * img2.astype(float), sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


__all__ = [
    "signal_to_noise_ratio",
    "contrast_to_noise_ratio",
    "structural_similarity_index",
]
