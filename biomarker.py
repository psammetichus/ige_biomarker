import numpy as np
import scipy.signal as scisig

def sComputeMeasures():
    pass

def fBandPass(data,low,high,dt):
    """bandpass the data with cutoff freqs low and high; dt is 1/samprate"""
    freqs = np.array([low,high]) * (dt*2) #half-cycles/sample
    b,a = scisig.butter(2, freqs, btype="bandpass")
    return scisig.filtfilt(b,a,data,1)

def fAlphaPeak(data, o1chan, o2chan, dt): 
    """computes peak alpha freq by scanning O1 and O2 and looking 
    for freqs in the 8-13 cps range"""
    specO1 = fSpectrum(data[o1chan])
    specO2 = fSpectrum(data[o2chan])
    pass

def fSpectrum(eegdata, dt):
    """computes freq spectrum"""
    Fs = 1./dt
    N = np.size(eegdata)
    dF = Fs/N
    f = np.arange(-Fs/2, Fs/2, dF)
    signal = eegdata-mean(eegdata)
    spectron = np.absolute(np.fft.fftshift(np.rfft(signal)))
    return spectron,f[f>0]

def fDespur(M):
    connmatrix = np.ones(M.shape[0])
    pass

def biomarker(eegdata, dt):
    """computes alpha, meanDeg, and local coupling
    based on data and time interval dt in sec (sampling rate)
    eegdata is assumed to be of shape (nCH, nsamples)"""
    nCH = eegdata.shape[0]
    eeglen = eegdata.shape[1]
    #average reference
    mydata = mydata - np.average(eegdata, 0)
    # select 20 s segment
    Nt = min(np.floor(20./dt), eeglen)

    #bandpass data, 2nd order butterworth
    #in initial paper, they do an FFT-based filter
    mydata2 = fBandPass(mydata,1,48,dt)
    lowalphadata = fBandPass(mydata,6,9,dt)
    mydata = mydata/np.average(np.std(mydata2,1))

    #compute std/PLF
    mySTDband = np.std(lowalphadata)
    
    #instantaneous phase and spectral power
    iphase = np.angle(scisig.hilbert(lowalphadata))
    spower = np.var(lowalphadata)

    #PLF and phase lag
    sr = 256
    a = np.fromfunction(lambda n,m: np.mean(np.exp(1j*(iphase[n,25:-256]
      - iphase[m,256:-256]))), (n,m))
    plf[n,m] = np.absolute(a)
    lag[n,m] = np.angle(a)
    
    meanDeg = np.mean(np.sum(plf))
    alpha = fAlphaPeak(mydata,dt)
    plfSparse = fDespur(plf[lag>0])
    
