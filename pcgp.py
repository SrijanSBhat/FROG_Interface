import numpy as np
from stqdm import stqdm

def randompulse(n):
    tvec = np.arange(n) - int(n / 2)
    p = np.exp(-tvec ** 2 / (n / 16) ** 2) * np.exp(2.0j * np.pi * np.random.rand(n))
    #p = np.random.rand(n) * np.exp(2.0j * np.pi * np.random.rand(n))
    return p

def FROG(field1, field2, mode='shg'): 
    nn = len(field1)
    n2 = int(nn / 2)
    if mode == 'shg':
        ap = np.outer(field1, field2) + np.outer(field2, field1)  # Axis: time-time
    elif mode == 'blind':
        ap = np.outer(field1, field2)
    m1 = np.zeros(np.shape(ap), dtype=np.complex128)
    m2 = np.zeros(np.shape(ap), dtype=np.complex128)
    for i in range(n2 - 1, -n2, -1):
        m1[i + n2, :] = np.roll(ap[i + n2, :], -i)
    m1 = np.transpose(m1)
    for i in range(nn):
        m2[i, :] = np.roll(np.fft.fft(np.roll(m1[i, :], +n2)), -n2)  # time-freq
    m2 = np.transpose(m2)  # freq - time
    m2 = m2 / np.max(np.max(np.abs(m2)))
    return m2

def pcgp_iteration(mexp, pulse, gatepulse, mode='shg', svd='full'):

    if pulse is None:
        nn = np.shape(mexp)[0]
        pulse = randompulse(nn)
    if gatepulse is None:
        nn = np.shape(mexp)[0]
        gatepulse = randompulse(nn)
    nn = len(pulse)
    n2 = int(nn / 2)
    m2 = FROG(pulse, gatepulse, mode=mode)
    i1 = np.abs(mexp) ** 2 / np.sum(np.sum(np.abs(mexp) ** 2))
    i2 = np.abs(m2) ** 2 / np.sum(np.sum(np.abs(m2) ** 2))
    ferr = np.sqrt(1 / nn ** 2 * np.sum(np.sum(np.abs(i1 - i2) ** 2)))

    m3 = np.abs(mexp) * np.exp(1.0j * np.angle(m2))
    m3 = np.transpose(m3)  # zeit - freq
    m4 = np.zeros(np.shape(m2), dtype=np.complex128)
    m5 = np.zeros(np.shape(m2), dtype=np.complex128)

    for i in range(nn):
        m4[i, :] = np.roll(np.fft.ifft(np.roll(m3[i, :], -n2)), n2)
    for i in range(n2 - 1, -n2, -1):
        m5[i + n2, :] = np.roll(m4[:, i + n2], i)  # time-time
    if svd=='full':
        # full SVD
        u, w, v = np.linalg.svd(m5)
        pulse = u[:, 0]
        gatepulse = v[0, :]
    else:
        #  power method
        pulse = np.dot(np.dot(m5, np.transpose(m5)), pulse)
        gatepulse = np.dot(np.dot(np.transpose(m5), m5), gatepulse)
        pulse = pulse / np.sqrt( np.sum( np.abs(pulse)**2))
        gatepulse = gatepulse / np.sqrt(np.sum(np.abs(gatepulse) ** 2))
    return pulse, gatepulse, ferr


def PCGP(mexp, iterations=10, mode='shg', pulse=None, gatepulse=None, svd='full'):
    errors = []
    sps = []
    gps = []

    for jj in stqdm(range(iterations), desc="Running PCGP Iterations"):
        pulse, gatepulse, frogerr = pcgp_iteration(mexp, pulse, gatepulse, mode=mode, svd=svd)
        errors.append(frogerr)
        sps.append(pulse)
        gps.append(gatepulse)

    rd = {'errors': errors, 'sp': sps, 'gp': gps, 'exp': mexp}
    minerr = np.min(rd['errors'])
    indx = np.nonzero(minerr == rd['errors'])[0][0]
    rd['minerror'] = minerr
    rd['min_sp'] = rd['sp'][indx]
    rd['min_gp'] = rd['gp'][indx]
    rd['mode'] = mode
    return rd

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


def remove_relative_phase_offset(field1, field2, border=6,
                          intermediate_steps=50,
                           magnitudes=np.linspace(0, 4, 5)):
    """Remove the arbitrary phase differences between two fields (linear+offset).

    The relative phases for two fields in the context of Frog traces can differ by some
    arbitrary offsets: the first one is the constant offset, the second one is the slope
    of the phase in the spectral domain (which corresponds to the temporal position).
    This algorithm removes these two for the second input field with respect to the first one.
    It varies the spectral phase (offset + linear component) and tries to find the
    best overlap.

    Arguments:

        field1 : reference field

        field2 : field whose phase shall be adjusted;

    Optional arguments:

        border : Numerical value for the phases to be varied in between

        intermediate_steps : number of intermediate steps for each magnitude

        magnitudes: np.array or list of powers of 10. Phase offsets and slopes
                    as border * 10 **(-magnitudes) are tried and refines successively.

    Returns:

        field3 : field2 with adjusted phase.

    """
    nn = len(field1)
    n2 = int(nn / 2.0)
    xv = np.arange(nn) - n2
    testSP1 = np.roll(np.fft.fft(np.roll(field1, n2)), n2)
    testSP2 = np.roll(np.fft.fft(np.roll(field2, n2)), n2)
    NN = intermediate_steps
    MM = np.zeros((NN, NN))
    PosNullX = 0
    PosNullY = 0
    for magnitude in magnitudes:
        jjv = np.linspace(-border * 10 ** -magnitude, border * 10 ** -magnitude, NN)\
              + PosNullX
        kkv = np.linspace(-border * 10 ** -magnitude, border * 10 ** -magnitude, NN)\
              + PosNullY
        for j, jj in enumerate(jjv):
            for k, kk in enumerate(kkv):
                MM[j, k] = np.sum(
                    (np.imag(testSP1) - np.imag(testSP2 \
                                * np.exp(1.0j * (xv * jj + kk)))) ** 2
                    * (np.real(testSP1) - np.real(testSP2 \
                                * np.exp(1.0j * (xv * jj + kk)))) ** 2
                )
        minindex = np.nonzero(MM == np.min(np.min(MM)))
        PosNullX = jjv[minindex[0][0]]
        PosNullY = kkv[minindex[1][0]]

    field3 = np.roll(
        np.fft.ifft(
            np.roll(
                testSP2 * np.exp(1.0j * (PosNullX * xv + PosNullY))
                , n2)
        ), n2)
    return field3


def remove_shg_freq_ambi(pulse):
    """Test and remove the shg frequency ambiguity (spectral shift by n/2).

    During reconstruction of SHG Frog, a common ambiguity occuring is a field shifted by
    half of the window in the spectral domain. As it gives the same (often correct) Frog
    trace as the unshifted field, a physically valid field can be constructed by shifting
    the spectrum back to the center of the frequency domain.

    Arguments:

         pulse : pulse field

    Returns:

        pulse2 : shifted pulse field

    """
    l = len(pulse)
    l2 = int(l / 2)
    tests = np.exp(- ((np.arange(l) - l2) / (l / 4)) ** 6)
    pulsspeksh = np.fft.ifft(np.fft.fftshift(np.fft.fft(pulse)))
    normal = np.sum(np.abs(np.multiply(tests, np.fft.fftshift(np.fft.fft(pulse)))))
    shifted = np.sum(np.abs(np.multiply(tests, np.fft.fftshift(np.fft.fft(pulsspeksh)))))
    if normal > shifted:
        return pulse
    else:
        return pulsspeksh



def remove_blind_freq_ambi(pulse, gpulse, inter_steps=200, check_shg_ambi=True):
    """Experimental: remove blind frog frequency ambiguity."""
    errors = []
    for i in range(inter_steps):
        ll = len(pulse)
        xv = np.arange(ll) - int(ll / 2)
        dom = np.pi / inter_steps * 2 * i
        sps = np.fft.fftshift(np.abs(np.fft.fft(pulse * np.exp(1.0j * dom * xv))))
        spg = np.fft.fftshift(np.abs(np.fft.fft(gpulse * np.exp(-1.0j * dom * xv))))
        errors.append(np.sum(np.abs(sps - spg) ** 2))
    minerr = np.min(errors)
    ii = np.nonzero(minerr == errors)[0][0]
    dom = np.pi / inter_steps * 2 * ii
    pulse = pulse * np.exp(1.0j * dom * xv)
    gpulse = gpulse * np.exp(-1.0j * dom * xv)
    if check_shg_ambi:
        pulse = remove_shg_freq_ambi(pulse)
        gpulse = remove_shg_freq_ambi(gpulse)
    return pulse, gpulse



def specshift_to_ref(referencepulse, pulse, inter_steps=200):
    """Shift spectrum of some pulse for maximum overlap with reference pulse spectrum.
    
    Input:
    
        referencepulse : reference
        
        pulse : pulse whose spectrum shall be shifted
        
    Optional Arguments: 
    
        inter_steps : number of intermediate steps to try 
    
    """
    errors = []
    for i in range(inter_steps):
        ll = len(referencepulse)
        xv = np.arange(ll) - int(ll / 2)
        dom = np.pi / inter_steps * 2 * i
        sps = np.fft.fftshift(np.abs(np.fft.fft(referencepulse)))
        spg = np.fft.fftshift(np.abs(np.fft.fft(pulse * np.exp(-1.0j * dom * xv))))
        errors.append(np.sum(np.abs(sps - spg) ** 2))
    minerr = np.min(errors)
    ii = np.nonzero(minerr == errors)[0][0]
    dom = np.pi / inter_steps * 2 * ii
    pulse = pulse * np.exp(-1.0j * dom * xv)
    return pulse

def calculate_fwhm(t, intensity):
    intensity = intensity / np.max(intensity)
    half_max = 0.5
    crossings = np.where(np.diff(np.sign(intensity - half_max)))[0]

    if len(crossings) < 2:
        return 0.0  # Not enough crossings

    t_fwhm = []
    for idx in crossings:
        # Linear interpolation between points around the crossing
        t1, t2 = t[idx], t[idx + 1]
        y1, y2 = intensity[idx], intensity[idx + 1]
        slope = (y2 - y1) / (t2 - t1)
        if slope == 0:
            continue
        t_zero = t1 + (half_max - y1) / slope
        t_fwhm.append(t_zero)

    if len(t_fwhm) >= 2:
        return np.abs(t_fwhm[-1] - t_fwhm[0])
    else:
        return 0.0
    
def extract_data(mexp, pulse, gpulse, mode='shg',
                fixshg_ambi=True,
                fixblind_ambi=True):
    nn = len(pulse)
    mres = FROG(pulse, gpulse, mode=mode)
    i1 = np.abs(mexp) ** 2 / np.sum(np.sum(np.abs(mexp) ** 2))
    i2 = np.abs(mres) ** 2 / np.sum(np.sum(np.abs(mres) ** 2))
    ferr = np.sqrt(1 / nn ** 2 * np.sum(np.sum(np.abs(i1 - i2) ** 2)))
    if mode=='shg' and fixshg_ambi:
        pulse = remove_shg_freq_ambi(pulse)
        gpulse = remove_shg_freq_ambi(gpulse)
    if mode=='blind' and fixblind_ambi:
        pulse, gpulse = remove_blind_freq_ambi(pulse, gpulse )

    return i1, i2, pulse, ferr