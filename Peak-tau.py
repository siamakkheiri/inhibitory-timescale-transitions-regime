import numpy as np
import scipy
# from scipy.signal import find_peaks
from scipy import signal
from scipy.ndimage import gaussian_filter1d
# from scipy.signal import butter, lfilter, savgol_filter, sosfiltfilt, correlate
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import matplotlib.gridspec as gridspec
import nest
# import nest.voltage_trace
# import nest.raster_plot
from scipy.signal import find_peaks

nest.ResetKernel()
nest.set_verbosity("M_WARNING")
nest.SetKernelStatus({"overwrite_files": True,'print_time': True,'local_num_threads':1})
nest.rng_seed = 22587425
np.random.seed(31360215)
dt = 0.1
nest.resolution = dt
tmax= 10000
start = 200

order = 300
N_E1, N_I1 = int(0.8*order),int(0.2*order)
N1 = N_E1 + N_I1 #300

Vrest = -70.0
Vtt = -51.0
tau_d_inh_exc = np.arange(1.0, 10.1, 0.2)
tau_d_inh_inh = np.arange(1.0, 10.1, 0.2)
# f_decay_exc = np.zeros(len(t_decay_i))
# f_decay_inh = np.zeros(len(t_decay_i))
# p_decay_exc = np.zeros(len(t_decay_i))
# p_decay_inh = np.zeros(len(t_decay_i))
bin = 1
t = tmax-start
fs = int((t/(2*bin))-1 )
frequency = np.zeros( fs )


# nperseg = int(t//8)
# overlap = nperseg//2
# psd_len = int(nperseg/2 + 1)

# PSD_exc_w = np.zeros([len(t_decay_i),psd_len])
# PSD_inh_w = np.zeros([len(t_decay_i),psd_len])

sigma = 1.0   # sigma in samples (here 1 sample = 1 ms because B=1.0)
# g_win = (1/np.sqrt(2*np.pi*(sigma**2)))*scipy.signal.windows.gaussian(11, std=sigma)
tau_d_exc_exc = 5.0
# tau_d_inh_inh = 7.0 
tau_d_exc_inh = 5.0 
external_rate = 4000
delay = 2
probability = 0.4


def preprocess_psd(PSD, freq):
    """
    Sort frequency axis in ascending order and reorder PSD accordingly.
    Works for 1D or multi-dimensional PSD (frequency axis assumed last).
    """
    freq = np.asarray(freq)
    PSD = np.asarray(PSD)

    idx = np.argsort(freq)
    freq_sorted = freq[idx]

    if PSD.ndim == 1:
        PSD_sorted = PSD[idx]
    else:
        PSD_sorted = PSD[..., idx]

    return PSD_sorted, freq_sorted


def smooth_psd_gaussian(PSD, freq, sigma_hz):
    """
    Gaussian smoothing along frequency axis.
    """
    df = np.mean(np.diff(freq))

    if df <= 0:
        raise ValueError("Frequency axis must be strictly increasing.")

    sigma_bins = sigma_hz / df

    if sigma_bins < 1e-6:
        return PSD

    return gaussian_filter1d(PSD, sigma=sigma_bins, axis=-1)


def dominant_peak_in_band(PSD, freq, low, high, min_prominence=0.0):
    """
    Return dominant peak frequency and power within band.
    If no peak found, returns simple maximum.
    """

    mask = (freq >= low) & (freq <= high)
    if not np.any(mask):
        return 0.0, 0.0

    psd_band = PSD[..., mask]
    freq_band = freq[mask]

    if psd_band.ndim > 1:
        raise ValueError("This function expects 1D PSD for peak detection.")

    peaks, props = find_peaks(psd_band, prominence=min_prominence)

    if peaks.size > 0:
        idx = np.argmax(props["prominences"])
        peak_freq = freq_band[peaks[idx]]
        peak_power = psd_band[peaks[idx]]
    else:
        # fallback to simple max
        idx = np.argmax(psd_band)
        peak_freq = freq_band[idx]
        peak_power = psd_band[idx]

    return peak_freq, peak_power


def extract_dominant_frequency(PSD, freq,
                                band_low, band_high,
                                sigma_hz=2,
                                min_prominence=0.0):
    """
    Full pipeline:
    - sort frequency
    - smooth PSD
    - extract dominant peak in band
    """

    PSD, freq = preprocess_psd(PSD, freq)

    PSD_smooth = smooth_psd_gaussian(PSD, freq, sigma_hz)

    peak_freq, peak_power = dominant_peak_in_band(
        PSD_smooth,
        freq,
        band_low,
        band_high,
        min_prominence=min_prominence
    )

    return peak_freq, peak_power, PSD_smooth

# def max_peak_frequency(PSD, freq, low, high, k=1.0, prominence=0.01):

#     # --- Band-pass mask ---
#     mask = (freq >= low) & (freq <= high)
#     psd_band = PSD[mask]
#     freq_band = freq[mask]

#     if psd_band.size == 0:
#         return 0.0, 0.0

#     # --- Threshold ---
#     threshold = np.mean(psd_band) + k * np.std(psd_band)

#     # --- Find peaks above threshold ---
#     peaks, properties = find_peaks(psd_band,
#                                    height=threshold,
#                                    prominence=prominence)

#     if peaks.size == 0:
#         return 0.0, 0.0

#     # --- Select the largest peak (by power) ---
#     peak_heights = properties["peak_heights"]
#     max_idx = np.argmax(peak_heights)

#     peak_power = peak_heights[max_idx]
#     peak_freq = freq_band[peaks[max_idx]]

#     return peak_freq, peak_power

Peak_Freq_exc = np.zeros([len(tau_d_inh_exc),len(tau_d_inh_inh)])
Peak_Freq_inh = np.zeros([len(tau_d_inh_exc),len(tau_d_inh_inh)])
Peak_PSD_exc = np.zeros([len(tau_d_inh_exc),len(tau_d_inh_inh)])
Peak_PSD_inh = np.zeros([len(tau_d_inh_exc),len(tau_d_inh_inh)])


for te in range(len(tau_d_inh_exc)):
    for ti in range(len(tau_d_inh_inh)):
        nest.ResetKernel()
        ndict_pop1_e = {"C_m":200.0,"g_L":10.0,'t_ref':1.0, 'E_ex':0.0, 'E_in':-70.0,
                "tau_rise_ex":0.5 ,'tau_decay_ex':tau_d_exc_exc ,
                "tau_rise_in":0.5 ,'tau_decay_in':tau_d_inh_exc[te],
                "V_reset":-59.0,"V_th":-52.0,  
                "I_e":0.0,
                "V_m": [Vrest+(Vtt-Vrest)*np.random.rand() for x in range(N_E1)] 
                }

        ndict_pop1_i = {"C_m":100.0, "g_L":10.0,'t_ref':1.0, 'E_ex':0.0, 'E_in':-70.0,
                "tau_rise_ex":0.5 ,'tau_decay_ex':tau_d_exc_inh ,
                "tau_rise_in":0.5 ,'tau_decay_in':tau_d_inh_inh[ti] ,
                "V_reset":-59.0 ,"V_th":-52.0,
                "I_e":0.0,
                "V_m": [Vrest+(Vtt-Vrest)*np.random.rand() for x in range(N_I1)] 
                }

        # define population : ############################################

        pop_e1 = nest.Create("iaf_cond_beta", N_E1, params = ndict_pop1_e)
        pop_i1 = nest.Create("iaf_cond_beta", N_I1, params = ndict_pop1_i)
        pop1 = pop_e1 + pop_i1

        # records ################################################################################################
        

        multi1 = nest.Create("multimeter",{"record_from":["V_m","g_ex","g_in"]})
        nest.SetStatus(multi1, {'interval': 0.1,"start":start})
        # nest.SetStatus(multi1, {"record_to" : "ascii", "label" : "multi1"})

        spikes1 = nest.Create("spike_recorder", {"start":start})

        nest.Connect(multi1, pop1)
        nest.Connect(pop1, spikes1)

        # generate external input #############################################
        rate_ex_s = external_rate
        rate_in_s = external_rate
        noise_e_s = nest.Create("poisson_generator",params={"rate" : rate_ex_s})
        noise_i_s = nest.Create("poisson_generator",params={"rate": rate_in_s})

        nest.Connect(noise_e_s , pop_e1, syn_spec={"weight": 0.2, "delay": 0.1})
        nest.Connect(noise_i_s , pop_i1 , syn_spec={"weight": 0.25, "delay": 0.1})

        # #the synaptic connection pop1 ##################################################################
        P = probability
        P_ee, P_ie, P_ei, P_ii = P, P, P, P


        CEE  = int(P_ee*N_E1)   
        CIE  = int(P_ie*N_E1)    
        CEI  = int(P_ei*N_I1)      
        CII  = int(P_ii*N_I1)

        gee1 = 4.8/CEE  #0.05   5.8
        gie1 = 57.6/CIE    #0.6   57
        gei1 = -21.6/CEI  #0.9  21.6
        gii1 = -84/CII     #3.5   84

        #  print(CEE,CIE,CEI,CII)
        nest.Connect(pop_e1,pop_e1,{'rule': 'fixed_indegree','indegree': CEE, 'allow_autapses': False,
                    "allow_multapses":False}, syn_spec = {"weight":gee1,"delay":delay})

        nest.Connect(pop_e1,pop_i1,{'rule': 'fixed_indegree','indegree': CIE, 'allow_autapses': False,
                    "allow_multapses":False}, syn_spec = {"weight":gie1,"delay":delay})

        nest.Connect(pop_i1,pop_e1,{'rule': 'fixed_indegree','indegree': CEI, 'allow_autapses': False,
                    "allow_multapses":False}, syn_spec = {"weight":gei1,"delay":delay})

        nest.Connect(pop_i1,pop_i1,{'rule': 'fixed_indegree','indegree': CII, 'allow_autapses': False,
                    "allow_multapses":False}, syn_spec = {"weight":gii1,"delay":delay})
        #############################################################################
        nest.Simulate(tmax)
        #############################################################################
        # plt.rcParams["figure.figsize"] = (8,4)
        # nest.raster_plot.from_device(spikes1, hist=True, hist_binwidth=2.0)
        # plt.title('EIF MODEL'+ '; ge='+str(gee) + ' ; gi='+str(gei)   )
        # plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

        A1 = spikes1.get()
        senderspop1 = A1["events"]['senders']
        timespop1 = A1["events"]["times"]
        idxe1 = np.where(senderspop1<=N_E1)
        idxi1 = np.where(senderspop1>N_E1)

        ######################################## Histogram #########################################################
        
        bins = np.arange(start, tmax+bin, bin)

        te1, b = np.histogram(timespop1[idxe1], bins)

        ti1, b = np.histogram(timespop1[idxi1], bins)

        ################### normalize #################
        # te1 = 1e3*te1/(N_E1*B)
        # ti1 = 1e3*ti1/(N_I1*B)
        # ################################################
        # te1 = np.convolve(te1, g_win, mode='same')
        # ti1 = np.convolve(ti1, g_win, mode='same')
        te1 = gaussian_filter1d(te1, sigma=sigma, mode='reflect')*(1e3/(N_E1*bin))
        ti1 = gaussian_filter1d(ti1, sigma=sigma, mode='reflect')*(1e3/(N_I1*bin))
        #######################################################################################################
        n = len(te1)
        dt0 = bin*1e-3
        L = np.arange(1,np.floor(n/2), dtype='int' )

        fhat_exc = np.fft.fft(te1)
        PSD_e = 2*(np.abs(fhat_exc)**2)*(dt0/n)

        freq = np.fft.fftfreq(fhat_exc.size,dt0 )      #(1000/(dt0*n))*np.arange(n)
        frequency = freq[L]
        ####################################################################################
        
        fhat_inh = np.fft.fft(ti1)
        PSD_i = 2*(np.abs(fhat_inh)**2)*(dt0/n) 
        #########################################################
        
        # Peak_Freq_exc[te,ti], Peak_PSD_exc[te,ti],a =  max_peak_frequency(
        #     PSD_e,
        #     freq,
        #     band_low=0,
        #     band_high=500
        #     )

        # Peak_Freq_inh[te,ti], Peak_PSD_inh[te,ti],b =  max_peak_frequency(
        #     PSD_i,
        #     freq,
        #     band_low=0,
        #     band_high=500,
        #     )
        #########################################################
        Peak_Freq_exc[te,ti], Peak_PSD_exc[te,ti],a =  extract_dominant_frequency(
            PSD_e,
            freq,
            band_low=0,
            band_high=200,
            sigma_hz=2,
            min_prominence=0.01
            )

        Peak_Freq_inh[te,ti], Peak_PSD_inh[te,ti],b =  extract_dominant_frequency(
            PSD_i,
            freq,
            band_low=0,
            band_high=200,
            sigma_hz=2,
            min_prominence=0.01
            )
        ######################################################################################


    

np.savetxt('VS-codes/spectrogram/tau/freqn.txt',frequency)

np.savetxt('VS-codes/spectrogram/tau/freq_excNet1.txt',Peak_Freq_exc)
np.savetxt('VS-codes/spectrogram/tau/freq_inhNet1.txt',Peak_Freq_inh)


np.savetxt('VS-codes/spectrogram/tau/PSD_excNet1.txt',Peak_PSD_exc)
np.savetxt('VS-codes/spectrogram/tau/PSD_inhNet1.txt',Peak_PSD_inh)



