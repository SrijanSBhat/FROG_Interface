import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.fft import fft, ifft, fftshift, ifftshift
from stqdm import stqdm

def gaussian(x, A, mu, sigma, offset):
    return A * np.exp(- (x - mu)**2 / (2 * sigma**2)) + offset

def crop_image(image, center, dimensions):
    x_center, y_center = center
    width, height = dimensions

    # Calculate cropping boundaries
    x_start = max(0, x_center - width // 2)
    x_end = min(image.shape[1], x_center + width // 2)
    y_start = max(0, y_center - height // 2)
    y_end = min(image.shape[0], y_center + height // 2)

    # Crop the image
    cropped_image = image[y_start:y_end, x_start:x_end]

    return cropped_image


def padded_trace(trace, image_size):
    n1 = trace.shape[0]
    n2 = trace.shape[1]
    new_rows = image_size
    new_cols = image_size
    padded = np.zeros((new_rows, new_cols))

    start_row = (new_rows - n1)//2
    start_col = (new_cols - n2)//2

    padded[start_row:start_row+n1, start_col:start_col+n2] = trace

    return padded

#Vanilla Functions: to do: import these functions as a package! For now this works
def grid(N,N_tau,range):
    time = np.linspace(-range,range,N)
    taus = np.linspace(-range * 0.08,range * 0.08,N_tau)
    dt = time[1] - time[0]
    freq = np.fft.fftshift(np.fft.fftfreq(time.shape[-1],d = np.mean(np.diff(time*1e-15))))*1e-12
    return time,taus,freq,dt

def A_lot_of_Bens_help(E, taus, time, freq):
    spectrum = []
    s = []
    for i, tau in enumerate(taus):
        roll_index = np.nanargmin(np.abs(time)) - np.nanargmin(np.abs(time + tau))
        rolled_pulse = np.roll(E,roll_index)
        Sig = E * rolled_pulse
        s.append(Sig)
        spec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Sig)))

        spectrum.append(spec)

    spectrum = np.array(spectrum)
    intensity = np.abs(spectrum)**2
    intensity = intensity/np.max(intensity)
    s = np.array(s)
    return s, spectrum, intensity

# def vanilla_frog(initial_pulse, exp_data, taus, time, freq, iterations=100, geometry = "SHG"):

#     pulse = initial_pulse.copy()
#     error_history = []
    
#     for i in range(iterations):
#         # Step 1: Generate signal field based on current E(t)
#         signal, spectrum, intensity = A_lot_of_Bens_help(pulse, taus, time, freq)
        
#         # Calculate FROG error according to Eq. (8.6)
#         # Find the scaling factor μ that minimizes the error
#         mu = np.sum(exp_data * intensity) / np.sum(intensity**2)
#         error = np.sqrt(np.mean((exp_data - mu * intensity)**2))
#         error_history.append(error)
        
#         # Step 2: Apply data constraint (Eq. 8.4)
#         # Replace magnitude with sqrt of experimental data, keep phase
#         amp = np.abs(spectrum)
#         phase = np.angle(spectrum)
#         epsilon = 1e-10  # Avoid division by zero
#         new_spectrum = np.sqrt(exp_data + epsilon) * np.exp(1j * phase)
        
#         # Step 3: Inverse Fourier transform to get E'sig(t, τ)
#         new_signal = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(new_spectrum, axes=1), axis=1), axes=1)
        
#         # Step 4: Generate new E(t) by integration over τ (Eq. 8.5)
#         pulse = np.sum(new_signal, axis=0)
        
#         # Normalize the pulse
#         pulse = pulse / np.max(np.abs(pulse))
    
#     # Final result
#     signal, spectrum, intensity = A_lot_of_Bens_help(pulse, taus, time, freq)
#     mu = np.sum(exp_data * intensity) / np.sum(intensity**2)
#     final_error = np.sqrt(np.mean((exp_data - mu * intensity)**2))
    
#     return pulse, error_history

def vanilla_frog(initial_pulse, exp_data, taus, time, freq, iterations=100, geometry="SHG"):
    pulse = initial_pulse.copy()
    error_history = []
    
    for i in stqdm(range(iterations), desc="Running Vanilla FROG"):  # Progress bar here
        # Step 1: Generate signal field based on current E(t)
        signal, spectrum, intensity = A_lot_of_Bens_help(pulse, taus, time, freq)
        
        # Step 2: Calculate FROG error (Eq. 8.6)
        mu = np.sum(exp_data * intensity) / np.sum(intensity**2)
        error = np.sqrt(np.mean((exp_data - mu * intensity)**2))
        error_history.append(error)
        
        # Step 3: Apply data constraint (Eq. 8.4)
        amp = np.abs(spectrum)
        phase = np.angle(spectrum)
        epsilon = 1e-10
        new_spectrum = np.sqrt(exp_data + epsilon) * np.exp(1j * phase)
        
        # Step 4: Inverse FFT to get new signal field
        new_signal = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(new_spectrum, axes=1), axis=1), axes=1)
        
        # Step 5: Update pulse estimate by integrating over τ (Eq. 8.5)
        pulse = np.sum(new_signal, axis=0)
        
        # Normalize pulse
        pulse = pulse / np.max(np.abs(pulse))
    
    # Final calculation
    signal, spectrum, intensity = A_lot_of_Bens_help(pulse, taus, time, freq)
    mu = np.sum(exp_data * intensity) / np.sum(intensity**2)
    final_error = np.sqrt(np.mean((exp_data - mu * intensity)**2))
    
    return pulse, error_history

def calculate_fwhm(x, y):
    """Calculate the FWHM of a pulse.
    
    Parameters:
    x : array-like
        The x-axis data (time or frequency)
    y : array-like
        The intensity profile
        
    Returns:
    float: The FWHM value
    """
    # Normalize the data
    y_norm = y / np.max(y)
    
    # Find the indices where the intensity is closest to 0.5 (half maximum)
    half_max_indices = np.where(np.diff(np.signbit(y_norm - 0.5)))[0]
    
    # If we have at least two crossings
    if len(half_max_indices) >= 2:
        # Use linear interpolation to find more precise positions
        x1, x2 = x[half_max_indices[0]], x[half_max_indices[-1]]
        fwhm = abs(x2 - x1)
        return fwhm
    else:
        print("Could not find two crossings at half maximum.")
        return None
