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
    mydata = mydata/np.average(np.std(mydata2,1))

    #compute std/PLF
    mySTDband = np.std(fBandPass(mydata,6,9,dt),1)
    
