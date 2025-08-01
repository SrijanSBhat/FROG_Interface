import streamlit as st
import numpy as np
import scipy
from numba import njit
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit
from stqdm import stqdm
import os

st.set_page_config(layout="wide", page_title="FROG Interface")
_, col1, col2, col3  = st.columns([0.2, 0.4, 0.4, 0.2])
uploaded_file = st.sidebar.file_uploader("Upload an image")
Transpose_button = st.sidebar.checkbox("Transpose Data", False)
choice = st.sidebar.selectbox("Method", ['Vanilla', 'PCGP', 'ML'])
measured_spectrum = st.sidebar.checkbox('Measured Spectrum', False)
if measured_spectrum:
    spectrum_file = st.sidebar.file_uploader("Upload your Measured spectrum")
bg = st.sidebar.checkbox('Backround file', False)
if bg:
    bg_file = st.sidebar.file_uploader("Upload your Background File")

bg_manual = st.sidebar.checkbox('Background Value', False)
if bg_manual:
    bg_value = st.sidebar.number_input('Enter the Background Value', value = 0)



if choice == 'Vanilla':
    guess = st.sidebar.checkbox("Pulse Input", False)
    # if measured_spectrum:
    #     fwhm = measured
    if guess:
        fwhm = st.sidebar.number_input(f'Full Width at Half Maximumℹ', value=150, help = "Expected Full width Half Maximum of the pulse.")
    N = st.sidebar.number_input('Number of Time Points', value=1000, help = "No of time points of the pulse to be predicted")
    N_Tau = N
    image_size = N
    ranges = st.sidebar.number_input('Range', value=5679, help = 'Add info here')
    iters = st.sidebar.number_input('Number of Iterations', min_value=30, max_value=600, value=100, help = 'Number of iterations to run the Algorithm for')
    start_button = st.sidebar.button("Start")
    

#Defining common functions

if uploaded_file == None:
    st.write('Upload a FROG Trace')
else:
    filename, file_extension = os.path.splitext(uploaded_file.name)
    if file_extension == '.txt':
        data = np.loadtxt(uploaded_file)
    else:
        image = Image.open(uploaded_file).convert("L")  # grayscale
        data = np.array(image, dtype=np.float32)

    if not bg:
        if bg_manual:
            data = data - np.min(data)
            # Subtract the mode of the values
            data = data - bg_value
            data[data < 0] = 0
            data = data / np.max(data)
        else:
            data = data - np.min(data)
            # Subtract the mode of the values
            unique, counts = np.unique(data, return_counts=True)
            most_common = unique[np.argmax(counts)]
            data = data - most_common
            data[data < 0] = 0
            data = data / np.max(data)

    else: 
        bg = Image.open(bg_file).convert('L')
        bg_arr = np.array(bg, dtype=np.float32)
        data = data - bg
        data /= np.max(data)
        

if choice == 'Vanilla':
    from vanilla import *
    if start_button:
        time, taus, freq, dt = grid(N, N_Tau, ranges)
        if not guess: 
            pulse_guess = np.random.rand(N) + 1j * np.random.rand(N)
        else: 
            pulse_guess = np.exp(-2*np.log(2) * (time/fwhm)**2)
            

        data_padded = padded_trace(
            data,
            image_size,
        )

        retrieved_pulse, error_history = vanilla_frog(
            initial_pulse = pulse_guess,
            exp_data=data_padded.T,
            taus=taus,
            time=time,
            freq=freq,
            iterations=iters
        )

        # if time[600] < time[np.argmax(retrieved_pulse)] < time[-600]:
        #     retrieved_pulse, error_history = vanilla_frog(
        #         initial_pulse = pulse_guess,
        #         exp_data=data_padded,
        #         taus=taus,
        #         time=time,
        #         freq=freq,
        #         iterations=iters
        #     )

        signal, spectrum, intensity = A_lot_of_Bens_help(retrieved_pulse, taus, time, freq)
        
        temporal_intensity = np.abs(retrieved_pulse)**2
        fwhm_calc = calculate_fwhm(time, temporal_intensity)
        max_pos = np.argmax(np.abs(retrieved_pulse))
        
        A0 = np.max(temporal_intensity)
        mu0 = time[np.argmax(temporal_intensity)]
        sigma0 = 340 / 2.3548
        offset0 = np.min(temporal_intensity)
        p0_temp = [A0, mu0, sigma0, offset0]
        
        popt_temp, pcov_temp = curve_fit(gaussian, time, temporal_intensity, p0=p0_temp)
        A_temp, mu_temp, sigma_temp, offset_temp = popt_temp
        
        FWHM_temp = 2 * np.sqrt(2 * np.log(2)) * sigma_temp
        
        dt_fs = time[1] - time[0]      # time step in fss
        dt = dt_fs * 1e-15       # convert time step to seconds

        N = len(time)
        freq = np.fft.fftshift(np.fft.fftfreq(N, d=dt))  # in Hz
        spectrum = np.fft.fftshift(np.fft.fft(retrieved_pulse))
        spectral_intensity = np.abs(spectrum)**2
        spectral_intensity /= np.max(spectral_intensity)  #Normalise

        # Convert frequency axis to wavelength.
        c = 3e8                 # Speed of light (m/s)
        lambda0 = 1030e-9       # Central wavelength (1030 nm in m)
        f0 = c / lambda0

        # Convert frequency to wavelength (in nm) using:
        # wavelength = c / (f0 + freq) then convert m -> nm.
        wavelength = 1e9 * c / (f0 + freq)

        # Reverse arrays so that wavelength increases with index
        wavelength = wavelength[::-1]
        spectral_intensity = spectral_intensity[::-1]

        # Initial guess for the spectral fit.
        A0_spec = np.max(spectral_intensity)
        mu0_spec = wavelength[np.argmax(spectral_intensity)]
        sigma0_spec = 10.0 / 2.3548  
        offset0_spec = np.min(spectral_intensity)
        p0_spec = [A0_spec, mu0_spec, sigma0_spec, offset0_spec]

        # Fit the Gaussian to the spectral intensity data.
        popt_spec, pcov_spec = curve_fit(gaussian, wavelength, spectral_intensity, p0=p0_spec)
        A_spec, mu_spec, sigma_spec, offset_spec = popt_spec

        # Calculate the spectral FWHM in nm.
        FWHM_spec = 2 * np.sqrt(2 * np.log(2)) * sigma_spec


        phase = np.angle(retrieved_pulse)

        with col1:
            st.markdown("### Experimental FROG Trace")
            figi, axi = plt.subplots()
            axi.set_title("Experimental FROG Trace")
            mesh = axi.pcolormesh(taus, freq, data_padded, shading='auto', cmap='plasma')
            figi.colorbar(mesh, ax=axi, label="Intensity")
            plt.tight_layout()
            st.pyplot(figi)
            st.markdown("### Retrieved Pulse")
            fig, ax = plt.subplots()
            ax.set_title("Retrieved E(t)", fontsize=14)
            ax.plot(time, np.abs(retrieved_pulse), label='Intensity')
            ax.plot(time, phase, label='Phase')
            ax.set_xlabel('Time (fs)', fontsize=14)
            ax.set_ylabel('Amplitude / Radians', fontsize=14)
            ax.set_xlim(time[max_pos] - 3*fwhm_calc, time[max_pos] + 3*fwhm_calc)

            ax.legend()
            st.pyplot(fig)

        with col2:
            st.markdown("### Retrieved FROG Trace")
            fig2, ax2 = plt.subplots()
            ax2.set_title(f"Retrieved FROG trace, error = {error_history[-1]:.6f}")
            mesh = ax2.pcolormesh(taus, freq, intensity.T, shading='auto', cmap='magma')
            fig2.colorbar(mesh, ax=ax2, label="Intensity")
            plt.tight_layout()
            st.pyplot(fig2)

            st.markdown("### Spectral Intensity")
            fig3, ax3 = plt.subplots()
            ax3.plot(wavelength, spectral_intensity, 'r-', label='Spectral Intensity')
            ax3.set_title('Spectral Intensity', fontsize=14)
            ax3.set_xlabel('Wavelength (nm)', fontsize=14)
            ax3.set_ylabel('Intensity', fontsize=14)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            if measured_spectrum:
                spectrum_exp = np.loadtxt(spectrum_file)
                ax3.plot(spectrum_exp, 'b-', label='Measured Spectrum')

            plt.tight_layout()
            st.pyplot(fig3)

elif choice == 'PCGP':
    from pcgp import *
    calibration = st.sidebar.number_input('Calibration Value', value = 5.734, format="%.3f")
    iters = st.sidebar.number_input('Number of Iterations', value=200, help = 'Number of iterations to run the Algorithm for')
    svd = st.sidebar.selectbox("SVD Method", ['power', 'full'])
    mode = st.sidebar.selectbox("Geometry", ['shg', 'blind'])
    start_button = st.sidebar.button("Start")

    if start_button:
        N = data.shape[0] * calibration
        N = int(np.round(N/2) * 2)
        data_padded = padded_trace(
            data,
            N,
        )

        if Transpose_button:
            data_padded = data_padded.T

        Mexp = np.sqrt(data_padded)

        res = PCGP(Mexp, iterations=iters, mode='shg', pulse=None, gatepulse=None, svd=svd)

        exp_trace, ret_trace, pulse, error = extract_data(Mexp, res['min_sp'], res['min_gp'], mode)

        with col1:
            st.markdown("### Experimental FROG Trace")
            fig1, ax1 = plt.subplots()
            ax1.imshow(exp_trace, cmap='plasma')
            ax1.set_title('Experimental Trace')
            plt.tight_layout()
            st.pyplot(fig1)

            st.markdown("### Retrieved Pulse")
            fig2, ax2 = plt.subplots()
            ax2.set_title("Retrieved E(t)", fontsize=14)
            ax2.plot(np.abs(pulse),c='r', dashes=(3, 1), label='Signal')
            ax2b = ax2.twinx()
            ax2b.plot(np.unwrap(np.angle(pulse)), c='b', label = 'Phase')
            ax2.set_xlabel('Time (fs)', fontsize=14)
            ax2.set_ylabel('Amplitude', fontsize=14)
            ax2b.set_ylabel('Radians', fontsize = 14)

            ax2.legend()
            st.pyplot(fig2)

        with col2:
            st.markdown("### Retrieved FROG Trace")
            fig3, ax3 = plt.subplots()
            ax3.set_title(f'FROG error = {error:E}')
            ax3.imshow(ret_trace, cmap='magma')
            plt.tight_layout()
            st.pyplot(fig3)
            if measured_spectrum:
                spectrum_exp = np.loadtxt(spectrum_file)
                ax3.plot(spectrum_exp, 'b-', label='Measured Spectrum')



elif choice == 'ML':
    st.write('Work in Progress')
