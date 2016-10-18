import numpy as np
import scipy.signal as scisig

def sComputeMeasures():
    pass


def sClassifyL00(n):
    alpha = np.zeros(n)
    meanDeg = np.zeros(n)
    local = np.zeros(n, 19, 20)

def fAlphaPeak():
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
    plf = np.zeros((nCH,nCH))
    lag = np.zeros((nCH,nCH))
    for m in range(nCH):
        for n in range(nCH):
            a = np.mean(np.exp(1j*(iphase[n,256:-256] - iphase[m,256:-256])))
            plf[n,m] = np.absolute(a)
            lag[n,m] = np.angle(a)
    
    meanDeg = np.mean(np.sum(plf))
    alpha = pass
    
