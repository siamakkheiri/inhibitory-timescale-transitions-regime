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
from scipy.signal import welch

nest.ResetKernel()
nest.set_verbosity("M_WARNING")
nest.SetKernelStatus({"overwrite_files": True,'print_time': True,'local_num_threads':2})
nest.rng_seed = 22587425
np.random.seed(31360215)
dt = 0.1
nest.resolution = dt
tmax= 3000
start = 200

order = 1000
N_E1, N_I1 = int(0.8*order),int(0.2*order)
N1 = N_E1 + N_I1 #300

Vrest = -70.0
Vtt = -51.0
tau_d_inh = np.arange(1.0, 10.1, 0.1)
# tau_d_inh_exc = np.arange(1.0, 10.1, 0.1)
# tau_d_inh_inh = 2.0     #np.arange(1.0, 10.1, 0.1)
# f_decay_exc = np.zeros(len(t_decay_i))
# f_decay_inh = np.zeros(len(t_decay_i))
# p_decay_exc = np.zeros(len(t_decay_i))
# p_decay_inh = np.zeros(len(t_decay_i))
bin = 1.0
t = tmax-start
fs = int((t/(2*bin))-1 )
frequency = np.zeros( fs )
PSD_exc = np.zeros([len(tau_d_inh),fs])
PSD_inh = np.zeros([len(tau_d_inh),fs])

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
probability = 0.2

############################## pairwise correlation ###############################################      
bin_size = 1.0 #ms
t_min = start
t_max = tmax

def compute_binned_spike_trains(spike_times, neuron_ids, bin_size, t_min, t_max):
     neuron_ids = np.array(neuron_ids)
     spike_times = np.array(spike_times)
     bins = np.arange(t_min, t_max+ bin_size, bin_size)
     unique_ids = np.unique(neuron_ids)

     binned_trains = []
     for nid in unique_ids :
          times = spike_times[neuron_ids == nid]
          binned,_ = np.histogram(times, bins)
          binned_trains.append(binned)
     return np.array(binned_trains), bins[:-1]


def compute_pairwise_correlation(binned_trains):
    if binned_trains.shape[0] < 2: 
        return np.nan
    
    # np.corrcoef expects variables in rows, observations in columns
    # This replaces the nested for-loops and pearsonr calls
    corr_matrix = np.corrcoef(binned_trains)
    
    # Handle cases where std is 0 (which results in NaNs)
    corr_matrix = np.nan_to_num(corr_matrix)
    
    return corr_matrix

corr_exc = np.zeros(len(tau_d_inh))
corr_inh = np.zeros(len(tau_d_inh))

# mean_fr_exc = np.zeros(len(tau_d_inh))
# mean_fr_inh = np.zeros(len(tau_d_inh))

for td in range(len(tau_d_inh)):
     nest.ResetKernel()
     nest.set_verbosity("M_WARNING")
     nest.SetKernelStatus({"overwrite_files": True,'print_time': True,'local_num_threads':1})
     nest.rng_seed = 22587425
     np.random.seed(31360215)
     nest.resolution = dt
     ndict_pop1_e = {"C_m":200.0,"g_L":10.0,'t_ref':1.0, 'E_ex':0.0, 'E_in':-70.0,
               "tau_rise_ex":0.5 ,'tau_decay_ex':tau_d_exc_exc ,
               "tau_rise_in":0.5 ,'tau_decay_in':tau_d_inh[td],
               "V_reset":-59.0,"V_th":-52.0,  
               "I_e":0.0,
               "V_m": [Vrest+(Vtt-Vrest)*np.random.rand() for x in range(N_E1)] 
               }

     ndict_pop1_i = {"C_m":100.0, "g_L":10.0,'t_ref':1.0, 'E_ex':0.0, 'E_in':-70.0,
               "tau_rise_ex":0.5 ,'tau_decay_ex':tau_d_exc_inh ,
               "tau_rise_in":0.5 ,'tau_decay_in':tau_d_inh[td] ,
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
     ##############################################################################################
     binned_trains_exc, bin_centers= compute_binned_spike_trains(timespop1[idxe1], senderspop1[idxe1], bin_size, t_min, t_max)
     corr_matrix_exc = compute_pairwise_correlation(binned_trains_exc)
     corr_exc[td] = np.mean (corr_matrix_exc[np.triu_indices_from(corr_matrix_exc, k=1)])
     

     binned_trains_inh, bin_centers = compute_binned_spike_trains(timespop1[idxi1], senderspop1[idxi1], bin_size, t_min, t_max)
     corr_matrix_inh = compute_pairwise_correlation(binned_trains_inh)
     corr_inh[td]  = np.mean (corr_matrix_inh[np.triu_indices_from(corr_matrix_inh, k=1)])
     
     ######################################## Histogram #########################################################
     
     bins = np.arange(start, tmax+bin, bin)

     te1, b = np.histogram(timespop1[idxe1], bins)

     ti1, b = np.histogram(timespop1[idxi1], bins)

     ################### normalize #################
     # te1 = 1e3*te1/(N_E1*bin)
     # ti1 = 1e3*ti1/(N_I1*bin)
     # ################################################
     # te1 = np.convolve(te1, g_win, mode='same')
     # ti1 = np.convolve(ti1, g_win, mode='same')
     te1 = gaussian_filter1d(te1, sigma=sigma, mode='reflect')*(1e3/(N_E1*bin))
     ti1 = gaussian_filter1d(ti1, sigma=sigma, mode='reflect')*(1e3/(N_I1*bin))
     #######################################################################################################
     ####################################################################################

     n = len(te1)
     dt0 = bin*1e-3
     L = np.arange(1,np.floor(n/2), dtype='int' )
     fhat_exc = np.fft.fft(te1)

     PSD_e = 2*(np.abs(fhat_exc)**2)*(dt0/n)
     PSD_exc[td,:] = PSD_e[L]
     freq = np.fft.fftfreq(fhat_exc.size,dt0 )      #(1000/(dt0*n))*np.arange(n)
     # print(len(freq))
     frequency = freq[L]
     # print(len(frequency))

     ####################################################################################

     fhat_inh = np.fft.fft(ti1)
     PSD_i = 2*(np.abs(fhat_inh)**2)*(dt0/n) 
     PSD_inh[td,:] = PSD_i[L]
     ################################## WELCH #######################
     # fs_w = 1.0 / dt0  # Sampling frequency in Hz
     # freq1, PSD_exc_w[td,:] = welch(te1, fs=fs_w, window='hann', nperseg=nperseg, noverlap=overlap, scaling='density')

     # freq2, PSD_inh_w[td,:] = welch(ti1, fs=fs_w, window='hann', nperseg=nperseg, noverlap=overlap, scaling='density')

     ###################################### Mean Firing Rate   ################################################
     # mean_fr_exc[td] = np.mean(te1)
     # mean_fr_inh[td] = np.mean(ti1)


    
# np.savetxt('VS-codes/spectrogram/t_decay1.txt',t_decay_i)
# np.savetxt('VS-codes/P_connection/freqn.txt',frequency)
np.savetxt('spectrogram/freqr.txt',frequency)
np.savetxt('spectrogram/n1000PSD_exc1.txt',PSD_exc)
np.savetxt('spectrogram/n1000PSD_inh1.txt',PSD_inh)

np.savetxt('spectrogram/n1000corr_exc1.txt',corr_exc)
np.savetxt('spectrogram/n1000corr_inh1.txt',corr_inh)

