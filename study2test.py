######   study strong population   ######################

import numpy as np
from scipy import  signal
from scipy.stats import pearsonr
# from scipy import signal
from scipy.signal import  find_peaks,correlate, savgol_filter, sosfiltfilt, hilbert,butter, lfilter ,filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import  welch, csd
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import matplotlib.gridspec as gridspec
import nest
# import nest.voltage_trace
import nest.raster_plot
# import pickle
import os
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

plt.rcdefaults() # reset the plot configurations to default
# plt.rcParams['text.usetex']= True
# plt.rcParams['font.family']= 'serif'
# plt.rcParams['font.serif']= ["Tahoma"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans"],
})

nest.ResetKernel()
nest.set_verbosity("M_WARNING")
nest.SetKernelStatus({"overwrite_files": True,'print_time': True,'local_num_threads':1})
nest.rng_seed = 22587425
np.random.seed(31360215)

dt = 0.1
nest.resolution = dt
tmax= 2000

order = 1000         # number of neuron
N_E, N_I = int(0.8*order),int(0.2*order)
N = N_E + N_I       

Vrest = -70.0
V_initial = -51.0
V_reset = -59.0
V_thr = -52.0
t_ref = 1.0

rate = 4000 # external rate

delay_exc = 2.0
delay_inh = 2.0

# synapses time constant
tau_rise = 0.5
tau_exc_exc = 5.0
tau_exc_inh = 5.0
tau_inh_exc = 4.0
tau_inh_inh = 4.0


ndict_pop_exc= {"C_m":200.0,"g_L":10.0,'t_ref':t_ref , 'E_ex':0.0, 'E_in':-70.0,
          "tau_rise_ex":tau_rise ,'tau_decay_ex': tau_exc_exc,
          "tau_rise_in":tau_rise ,'tau_decay_in': tau_inh_exc, #Td_exc, 
           "V_reset":V_reset,"V_th":V_thr,  
           "E_L":-70.0, 
           # "I_e":0.0,
            "V_m": [Vrest+(V_initial-Vrest)*np.random.rand() for x in range(N_E)] 
            }

ndict_pop_inh = {"C_m":100.0, "g_L":10.0,'t_ref':t_ref , 'E_ex':0.0, 'E_in':-70.0,
          "tau_rise_ex":tau_rise ,'tau_decay_ex': tau_exc_inh,
          "tau_rise_in":tau_rise ,'tau_decay_in': tau_inh_inh,
          "V_reset":V_reset ,"V_th":V_thr,
          "E_L":-70.0,
          #  "I_e":0.0,
           "V_m": [Vrest+(V_initial-Vrest)*np.random.rand() for x in range(N_I)] 
           }

# define two population : ############################################

pop_exc = nest.Create("iaf_cond_beta", N_E, params = ndict_pop_exc)
pop_inh = nest.Create("iaf_cond_beta", N_I, params = ndict_pop_inh)
pop = pop_exc + pop_inh

### records ################################################################################################
start = 200

multi1 = nest.Create("multimeter",{"record_from":["V_m","g_ex","g_in"]})
nest.SetStatus(multi1, {'interval': 0.1,"start":start})
# nest.SetStatus(multi1, {"record_to" : "ascii", "label" : "multi1"})

spikes_recorder1 = nest.Create("spike_recorder", {"start":start}) 

nest.Connect(multi1, pop)
nest.Connect(pop, spikes_recorder1 )

### generate external input #############################################

rate_ex_s = rate
rate_in_s = rate
noise_e_s = nest.Create("poisson_generator",params={"rate" : rate_ex_s}) #,"stop":400
noise_i_s = nest.Create("poisson_generator",params={"rate": rate_in_s})

nest.Connect(noise_e_s , pop_exc, syn_spec={"weight": 0.2, "delay": 0.1})
nest.Connect(noise_i_s , pop_inh , syn_spec={"weight": 0.25, "delay": 0.1})

### the synaptic connection pop ##################################################################
P = 0.4
P_ee, P_ie, P_ei, P_ii = P, P, P, P

CEE  = int(P_ee*N_E)   
CIE  = int(P_ie*N_E)    
CEI  = int(P_ei*N_I)      
CII  = int(P_ii*N_I)

# gee = 0.05   #5.8
# gie = 0.6  
# gei = -0.9 
# gii = -3.5   


gee = 4.8/CEE  #0.05   5.8
gie = 57.6/CIE    #0.6   57
gei = -21.6/CEI  #0.9  21.6
gii = -84/CII     #3.5   84

print("gee=",gee, ", gie=",gie, ", gei=",gei, ", gii=", gii)
t_peak = (tau_inh_inh*tau_rise/(tau_inh_inh-tau_rise))*np.log(tau_inh_inh/tau_rise)

f = 1/(np.exp(-t_peak/tau_inh_inh)-np.exp(-t_peak/tau_rise))  # normilaze factor of S(t)
# print(f)

# gii = gii/( f*(tau_rise-tau_inh_inh)/(tau_inh_inh*tau_rise) )
# print(t_peak,gii)

# print(CEE,CIE,CEI,CII)
nest.Connect(pop_exc,pop_exc,{'rule': 'fixed_indegree','indegree': CEE, 'allow_autapses': False,
               "allow_multapses":False}, syn_spec = {"weight":gee,"delay":delay_exc})

nest.Connect(pop_exc,pop_inh,{'rule': 'fixed_indegree','indegree': CIE, 'allow_autapses': False,
               "allow_multapses":False}, syn_spec = {"weight":gie,"delay":delay_exc})

nest.Connect(pop_inh,pop_exc,{'rule': 'fixed_indegree','indegree': CEI, 'allow_autapses': False,
               "allow_multapses":False}, syn_spec = {"weight":gei,"delay":delay_inh})

nest.Connect(pop_inh,pop_inh,{'rule': 'fixed_indegree','indegree': CII, 'allow_autapses': False,
               "allow_multapses":False}, syn_spec = {"weight":gii,"delay":delay_inh})

#############################################################################
nest.Simulate(tmax)
#############################################################################
# plt.rcParams["figure.figsize"] = (8,5)
# #plt.figure(figsize=(8,4))
# nest.raster_plot.from_device(spikes_recorder1 , hist=True, hist_binwidth=2.0)
# plt.title('Raster #300')
# plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

A1 = spikes_recorder1.get()
senderspop1 = A1["events"]['senders']
timespop1 = A1["events"]["times"]
idxe1 = np.where(senderspop1<=N_E)
idxi1 = np.where(senderspop1>N_E)

# idxe_part = np.where( (senderspop1<=N_E))
# idxi_part = np.where((senderspop1>N_E) & (senderspop1<=(N)) )
# print(idxi_part)

# exc_neurons = np.arange(1, N_E+1)
# inh_neurons = np.arange(N_E+1, N+1)

# idxe1 = np.isin(senderspop1, exc_neurons)
# idxi1 = np.isin(senderspop1, inh_neurons)

########## SPIKE TRAIN SYNCHRONY (STS) INDEX ############################
def compute_STS(spike_times, t_start_ms, t_end_ms, bin_ms=1.0):
    spikes = spike_times[
        (spike_times >= t_start_ms) & (spike_times < t_end_ms)
    ]

    n_bins = int((t_end_ms - t_start_ms) / bin_ms)
    bins = np.linspace(t_start_ms, t_end_ms, n_bins + 1)

    A, _ = np.histogram(spikes, bins=bins)

    mean_A = np.mean(A)
    # print(mean_A)
    return np.mean(A**2) / mean_A**2
# print(compute_STS(timespop1[idxe1],start,tmax))
# print(compute_STS(timespop1[idxi1],start,tmax))

# print("Exc spikes found:", np.sum(np.isin(senderspop1, exc_neurons)))
# print("Inh spikes found:", np.sum(np.isin(senderspop1, inh_neurons)))



################# compute ISI, CV #################################################################################
# # تنظیم پارامترهای تحلیل و رسم
# # -------------------------------
# min_spikes_for_isi = 10

# # bins خطی و لگاریتمی (جدا قابل تغییرند)
# bin_isi = 5 #ms
# bins_linear_exc = np.arange(0, 200+ bin_isi, bin_isi)
# bins_linear_inh = np.arange(0, 200+ bin_isi,  bin_isi)
# bins_log_exc = np.logspace(np.log10(1.0), np.log10(500), 50)
# bins_log_inh = np.logspace(np.log10(1.0), np.log10(500), 50)

# # -------------------------------
# # تابع کمکی برای محاسبه ISI هر جمعیت
# # -------------------------------
# def compute_isi(times, senders, min_spikes):
#     per_neuron_isis = {}
#     pooled_isis = []
#     unique_senders = np.unique(senders)
    
#     for gid in unique_senders:
#         idx = np.where(senders == gid)[0]
#         if idx.size < 2:
#             continue
#         spike_times_gid = np.sort(times[idx])
#         isis = np.diff(spike_times_gid)
#         if len(isis) >= (min_spikes - 1):
#             per_neuron_isis[gid] = isis
#             pooled_isis.extend(isis.tolist())
    
#     pooled_isis = np.array(pooled_isis)
#     return per_neuron_isis, pooled_isis

# # -------------------------------
# # محاسبه ISI برای نورون‌های تحریکی و مهاری
# # -------------------------------
# per_neuron_exc, pooled_exc = compute_isi(timespop1[idxe1], senderspop1[idxe1], min_spikes_for_isi)
# per_neuron_inh, pooled_inh = compute_isi(timespop1[idxi1], senderspop1[idxi1], min_spikes_for_isi)

# # print(f"Excitatory neurons with >= {min_spikes_for_isi} spikes: {len(per_neuron_exc)}")
# # print(f"Inhibitory neurons with >= {min_spikes_for_isi} spikes: {len(per_neuron_inh)}")

# # -------------------------------
# # آمار توصیفی برای هر جمعیت
# # -------------------------------
# def isi_stats(per_neuron_isis, pooled_isis, label):
#     neuron_mean_isi = []
#     neuron_cv = []
#     for isis in per_neuron_isis.values():
#         mean_ = np.mean(isis)
#         std_ = np.std(isis)
#         neuron_mean_isi.append(mean_)
#         neuron_cv.append(std_ / mean_ if mean_ > 0 else np.nan)
#     neuron_mean_isi = np.array(neuron_mean_isi)
#     neuron_cv = np.array(neuron_cv)
    
#     # print(f"\n--- {label} population ---")
#     # print(f"Pooled ISI mean: {np.mean(pooled_isis):.3f} ms")
#     # print(f"Pooled ISI median: {np.median(pooled_isis):.3f} ms")
#     # print(f"Per-neuron mean ISI: {np.nanmean(neuron_mean_isi):.3f} ms")
#     # print(f"Mean CV across neurons: {np.nanmean(neuron_cv):.3f}")
    
#     return neuron_mean_isi, neuron_cv

# mean_exc, cv_exc = isi_stats(per_neuron_exc, pooled_exc, "Excitatory")
# mean_inh, cv_inh = isi_stats(per_neuron_inh, pooled_inh, "Inhibitory")

# #######################b FOR ARTICLES #########################
# # ############ exc neuron ISI  #############################
# plt.figure(figsize=(12,5.5))
# # plt.rcParams['font.size'] = 15
# # 1. هیستوگرام ISI (خطی)
# plt.subplot(1,2,1)
# plt.hist(pooled_exc, bins=bins_linear_exc, density=True, alpha=0.8, color='skyblue', edgecolor='k')
# # plt.hist(pooled_inh, bins=bins_linear_inh, density=True, alpha=0.6, label='Inhibitory', color='blue')
# plt.xlabel("ISI (ms)",fontsize=40, labelpad=8.0)
# plt.ylabel("PDF",fontsize=40, labelpad=8.0)
# # plt.title("Pooled ISI (linear bins)")
# plt.tick_params(labelsize=25, axis="x",  pad = 8.0 )  #, pad = 9.0
# plt.tick_params(labelsize=25, axis="y",  pad = 8.0 )  #, pad = 9.0
# plt.gca().xaxis.set_major_locator(MultipleLocator(50))
# plt.gca().xaxis.set_major_formatter('{x:.0f}')
# plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
# plt.tick_params(which='major', length=14, width=1.8, color='k')
# plt.tick_params(which='minor', length=6, width=1.0, color='k')


# ############ inh neuron ISI  #############################

# plt.subplot(1,2,2)
# plt.hist(pooled_inh, bins=bins_linear_inh, density=True, alpha=0.7,  color='tomato', edgecolor='k')
# plt.xlabel("ISI (ms)",fontsize=40, labelpad=12.0)
# # plt.ylabel("PDF",fontsize=40, labelpad=12.0)
# # plt.title("Pooled ISI (linear bins)",fontsize=14)
# # plt.xlim(0,200)
# plt.tick_params(labelsize=25, axis="x",  pad = 8.0 )  #, pad = 9.0
# plt.tick_params(labelsize=25, axis="y",  pad = 8.0 )  #, pad = 9.0
# plt.gca().xaxis.set_major_locator(MultipleLocator(50))
# plt.gca().xaxis.set_major_formatter('{x:.0f}')
# plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
# plt.tick_params(which='major', length=14, width=1.8, color='k')
# plt.tick_params(which='minor', length=6, width=1.0, color='k')
# plt.tight_layout(pad=5.0)

# for ax in plt.gcf().get_axes():  # gcf = get current figure
#     for spine in ax.spines.values():
#         spine.set_linewidth(1.0)

# # folder_name = r'VS-codes/assess/new'
# # file_name = 'ISI, tau_exc='+str(tau_exc_exc) +', tau_inh='+str(tau_inh_inh) +'.png' 
# # if not os.path.exists(folder_name):  # چک کردن وجود پوشه
# #     os.makedirs(folder_name)
# # file_path = os.path.join(folder_name, file_name)  
# # plt.savefig(file_path, dpi=150, bbox_inches='tight') 

# #####################################################################################################
# plt.figure(figsize=(16,9))
# # 1. هیستوگرام ISI (خطی)
# plt.subplot(2,3,1)
# plt.hist(pooled_exc, bins=bins_linear_exc, density=True, alpha=0.8, color='skyblue', edgecolor='k')
# # plt.hist(pooled_inh, bins=bins_linear_inh, density=True, alpha=0.6, label='Inhibitory', color='blue')
# plt.xlabel("ISI (ms)",fontsize=30, labelpad=8.0)
# plt.ylabel("PDF",fontsize=30, labelpad=8.0)
# # plt.title("Pooled ISI (linear bins)")
# plt.tick_params(labelsize=22, axis="x",  pad = 8.0 )  #, pad = 9.0
# plt.tick_params(labelsize=22, axis="y",  pad = 8.0 )  #, pad = 9.0
# plt.gca().xaxis.set_major_locator(MultipleLocator(50))
# plt.gca().xaxis.set_major_formatter('{x:.0f}')
# plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
# plt.tick_params(which='major', length=14, width=1.8, color='k')
# plt.tick_params(which='minor', length=6, width=1.0, color='k')


# # 2. هیستوگرام ISI (لگاریتمی)
# plt.subplot(2,3,2)
# plt.hist(pooled_exc, bins=bins_log_exc, density=True, alpha=0.8, color='skyblue', edgecolor='k')
# # plt.hist(pooled_inh, bins=bins_log_inh, density=True, alpha=0.6, label='Inhibitory', color='blue')
# plt.xscale('log')
# plt.xlabel("ISI (ms)", fontsize=30, labelpad=8.0)
# plt.ylabel("PDF", fontsize=30, labelpad=8.0)
# # plt.title("Pooled ISI (log bins)")
# plt.tick_params(labelsize=22, axis="x",  pad = 8.0 )  #, pad = 9.0
# plt.tick_params(labelsize=22, axis="y",  pad = 8.0 )  #, pad = 9.0
# plt.tick_params(which='major', length=14, width=1.8, color='k')
# plt.tick_params(which='minor', length=6, width=1.0, color='k')


# # 4. CV  در هر نورون 
# plt.subplot(2,3,3)
# plt.hist(cv_exc[~np.isnan(cv_exc)], bins=30, alpha=0.7, color='skyblue', edgecolor='k')
# # plt.hist(cv_inh[~np.isnan(cv_inh)], bins=30, alpha=0.7, label='Inhibitory', color='blue')
# plt.xlabel("CV (ISI)", fontsize=30, labelpad=12.0)
# plt.ylabel("Count",fontsize=30, labelpad=12.0)
# # plt.title("Distribution of ISI variability (CV)")
# plt.tick_params(labelsize=22, axis="x",  pad = 8.0 )  #, pad = 9.0
# plt.tick_params(labelsize=22, axis="y",  pad = 8.0 )  #, pad = 9.0

# plt.gca().xaxis.set_minor_locator(MultipleLocator(0.05))
# plt.tick_params(which='major', length=14, width=1.8, color='k')
# plt.tick_params(which='minor', length=6, width=1.0, color='k')

# # plt.tight_layout()

# ############ inihibitory neuron ISI

# # plt.figure(figsize=(12,10))

# # 1. هیستوگرام ISI (خطی)
# plt.subplot(2,3,4)
# plt.hist(pooled_inh, bins=bins_linear_inh, density=True, alpha=0.7,  color='tomato', edgecolor='k')
# plt.xlabel("ISI (ms)",fontsize=30, labelpad=12.0)
# plt.ylabel("PDF",fontsize=30, labelpad=12.0)
# # plt.title("Pooled ISI (linear bins)",fontsize=14)
# # plt.xlim(0,200)
# plt.tick_params(labelsize=22, axis="x",  pad = 8.0 )  #, pad = 9.0
# plt.tick_params(labelsize=22, axis="y",  pad = 8.0 )  #, pad = 9.0
# plt.gca().xaxis.set_major_locator(MultipleLocator(50))
# plt.gca().xaxis.set_major_formatter('{x:.0f}')
# plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
# plt.tick_params(which='major', length=14, width=1.8, color='k')
# plt.tick_params(which='minor', length=6, width=1.0, color='k')

# # 2. هیستوگرام ISI (لگاریتمی)
# plt.subplot(2,3,5)
# plt.hist(pooled_inh, bins=bins_log_inh, density=True, alpha=0.7, color='tomato', edgecolor='k')
# plt.xscale('log')
# plt.xlabel("ISI (ms)",fontsize=30, labelpad=8.0)
# plt.ylabel("PDF",fontsize=30, labelpad=8.0)
# # plt.title("Pooled ISI (log bins)")
# plt.tick_params(labelsize=22, axis="x",  pad = 12.0 )  #, pad = 9.0
# plt.tick_params(labelsize=22, axis="y",  pad = 12.0 )  #, pad = 9.0

# plt.tick_params(which='major', length=14, width=1.8, color='k')
# plt.tick_params(which='minor', length=6, width=1.0, color='k')


# # 4. CV  در هر نورون 
# plt.subplot(2,3,6)
# plt.hist(cv_inh[~np.isnan(cv_inh)], bins=30, alpha=0.7, color='tomato' , edgecolor='k')
# plt.xlabel("CV (ISI)",fontsize=30, labelpad=12.0)
# plt.ylabel("Count",fontsize=30, labelpad=12.0)
# # plt.title("Distribution of ISI variability (CV)")
# plt.tick_params(labelsize=22, axis="x",  pad = 8.0 )  #, pad = 9.0
# plt.tick_params(labelsize=22, axis="y",  pad = 8.0 )  #, pad = 9.0
# # plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
# # plt.gca().xaxis.set_major_formatter('{x:.0f}')
# plt.gca().xaxis.set_minor_locator(MultipleLocator(0.05))
# plt.tick_params(which='major', length=14, width=1.8, color='k')
# plt.tick_params(which='minor', length=6, width=1.0, color='k')

# plt.tight_layout(pad=4.0)

# for ax in plt.gcf().get_axes():  # gcf = get current figure
#     for spine in ax.spines.values():
#         spine.set_linewidth(1.0)
# # folder_name = r'VS-codes/assess/new'
# # file_name = 'ISI, tau_exc='+str(tau_exc_exc) +', tau_inh='+str(tau_inh_inh) +'new.png' 
# # if not os.path.exists(folder_name):  # چک کردن وجود پوشه
# #     os.makedirs(folder_name)
# # file_path = os.path.join(folder_name, file_name)  
# # plt.savefig(file_path, dpi=150, bbox_inches='tight') 

###############################################################################################################################

mm1 = multi1.get()
s1 = mm1["events"]['senders']
t1 = mm1["events"]["times"][1::N]

Vm = np.zeros([N,len(t1)])
g_e = np.zeros([N,len(t1)])
g_i = np.zeros([N,len(t1)])

# I_syn_e = np.zeros([N,len(t1)])
# I_syn_i = np.zeros([N,len(t1)])

for i in range(0,N):
     Vm[i,:] = mm1["events"]['V_m'][i::N]
     g_e[i,:] = mm1["events"]['g_ex'][i::N]
     g_i[i,:] = mm1["events"]['g_in'][i::N]
    #  I_syn_e[i,:] = -g_e[i,:]*(Vm[i,:]-0)
    #  I_syn_i[i,:] = -g_i[i,:]*(Vm[i,:]-(-70))

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


# def compute_pairwise_correlation(binned_trains):
#     n = binned_trains.shape[0]
#     corr_matrix = np.zeros((n, n))
#     if n < 2: return np.nan 
#     for i in range(n):
#         for j in range(i, n):
#             if np.std(binned_trains[i]) > 0 and np.std(binned_trains[j]) > 0:
#                 r, _ = pearsonr(binned_trains[i], binned_trains[j])
#                 corr_matrix[i, j] = r
#                 corr_matrix[j, i] = r

#     return corr_matrix

def compute_pairwise_correlation(binned_trains):
    if binned_trains.shape[0] < 2: 
        return np.nan
    
    # np.corrcoef expects variables in rows, observations in columns
    # This replaces the nested for-loops and pearsonr calls
    corr_matrix = np.corrcoef(binned_trains)
    
    # Handle cases where std is 0 (which results in NaNs)
    corr_matrix = np.nan_to_num(corr_matrix)
    
    return corr_matrix


binned_trains_exc, bin_centers= compute_binned_spike_trains(timespop1[idxe1], senderspop1[idxe1], bin_size, t_min, t_max)
corr_matrix_exc = compute_pairwise_correlation(binned_trains_exc)
mean_corr_exc = np.mean (corr_matrix_exc[np.triu_indices_from(corr_matrix_exc, k=1)])
print('pairwise correlation exc', mean_corr_exc)

# فقط مقادیر بالای قطر اصلی که NaN نیستند را میانگین بگیرید
upper_tri = corr_matrix_exc[np.triu_indices_from(corr_matrix_exc, k=1)]
avg_corr_exc = np.mean(upper_tri[~np.isnan(upper_tri)])
print('pairwise correlation exc with Nan', avg_corr_exc)

binned_trains_inh, bin_centers = compute_binned_spike_trains(timespop1[idxi1], senderspop1[idxi1], bin_size, t_min, t_max)
corr_matrix_inh = compute_pairwise_correlation(binned_trains_inh)
mean_corr_inh = np.mean (corr_matrix_inh[np.triu_indices_from(corr_matrix_inh, k=1)])
print('pairwise correlation_inh', mean_corr_inh)

# plt.figure(20,figsize=(8, 8))
# plt.imshow(corr_matrix_exc, cmap='hot', interpolation='nearest')
# plt.colorbar(label='coefficient pearson exc')

# plt.figure(21,figsize=(8, 8))
# plt.imshow(corr_matrix_inh, cmap='hot', interpolation='nearest')
# plt.colorbar(label='coefficient pearson inh')
#########################################################################################################################


# fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,8)) #figsize=(8,5)
# #ax2 = ax1.twinx()
# # ax1.plot(t1,Vm[280,:],color='k')
# ax1.plot(t1,I_syn_e[10,:],color='tab:blue',lw=1.5 , label='g_Exc')
# ax1.plot(t1,I_syn_i[10,:],color='tab:red',lw=1.5 , label='g_inh')
# ax1.plot(t1, (I_syn_e[10,:]+I_syn_i[10,:]),color='tab:red',lw=1.5 , label='total')
# ax1.axhline((np.mean(I_syn_e[10,:])+np.mean(I_syn_i[10,:])) ,linestyle="dashed" ,color="black", lw=2.0)
# ax1.set_ylabel("$Mean\;Voltage$", size=16)
# ax1.set_xlabel('$t(ms)$', size=16)

# # ax1.plot(t1,V_mean_exc,color='orange',lw=2.2 , label='Voltage_Exc')
# # ax2.plot(t1,-g_i[280,:],color='r') #,alpha=0.6
# # ax2.plot(t1,-g_i[1,:],color='red',lw=2.2 ,label='I_syn_inh') #,alpha=0.6
# # ax2.plot(t1,g_e[280,:],color='b')
# # ax2.plot(t1,g_e[280,:],color='b',lw=2.2 , label='I_syn_exc')
# # a2 = ax2.twinx()comput frequen
# ax2.plot(t1,I_syn_e[280,:], color='tab:green', lw=1.5, label='#280_g_exc')
# ax2.plot(t1, I_syn_i[280,:], color='tab:red', lw=1.5, label='#280_g_inh')
# ax2.axhline((np.mean(I_syn_e[280,:])+np.mean(I_syn_i[280,:])) ,linestyle="dashed" ,color="black", lw=2.0)
# # ax2.plot(t1,-LFP_IN,'--', color='tab:red', lw=1.5, alpha=0.6)
# # ax2.plot(t1,(LFP_EX+LFP_IN), color='tab:blue', lw=1.5, label='I_syn_total')
# ax2.set_ylabel("$g$", size=16)
# ax2.set_xlabel('$t(ms)$', size=16)
# # ax2.tick_params(axis='y', labelcolor='c', direction='in')
# # a2.plot(t1,I_syn_i[280,:],color='y',lw=1.0 , label='I_syn_inh_IIIIIIIIIIII')
# # ax2.plot(t1,-g_e[0,:]*(Vm[0,:])-g_i[0,:]*(Vm[0,:]+70),lw=2.2 ,color='green',label='I_Syn_t')
# # ax2.plot(t1, I_syn_e[0,:]+I_syn_i[0,:], lw= 1.5,color='orange',label='I_Syn_t_IIIII')
# ax1.set_xlim(1000,1200)
# ax2.set_xlim(1000,1200)
# ax1.xaxis.set_tick_params(labelsize=14)
# ax1.yaxis.set_tick_params(labelsize=14)
# ax2.xaxis.set_tick_params(labelsize=14)
# ax2.yaxis.set_tick_params(labelsize=14)
# ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')
# ax1.grid()
# ax2.grid()

# folder_name = 'VS-codes/plots_compare'
# file_name = 'V_LFP_'+str(Td_inh) +'.png'  # file name
# if not os.path.exists(folder_name):  # چک کردن وجود پوشه
#     os.makedirs(folder_name)
# file_path = os.path.join(folder_name, file_name)  # تنظیم مسیر کامل فایل
# plt.savefig(file_path, dpi=200, bbox_inches='tight')  
# # ax3.legend(loc='upper right')

# fig, (ax8,ax9) = plt.subplots(2,1, figsize=(10,8)) #figsize=(8,5)
# num = 200
# # nume =40
# ax8.plot(t1,Vm[num,:],color='green',lw=1.5 , label='Voltage_exc')
# # ax8.plot(t1,Vm[nume,:],color='k',lw=1.5 , label='Voltage_Inh')
# a3 = ax8.twinx()
# a3.plot(timespop1[np.where(senderspop1==num+1)],senderspop1[np.where(senderspop1==num+1)],'.',color='r',markersize=7.0)
# # a3.plot(timespop1[np.where(senderspop1==nume+1)],senderspop1[np.where(senderspop1==nume+1)],'.',color='b',markersize=7.0)
# ax8.set_title("number of neuron"+str(num))
# ax8.set_ylabel("$Voltage$", size=16)
# ax8.set_xlabel('$t(ms)$', size=16)
# # ax8.plot(t1,Vm[83,:],color='r',lw=1.5 , label='Voltage_exc')
# # ax8.plot(t1,V_mean_inh,color='orange',lw=2.2 , label='mean_Voltage_Inh')
# # ax2.plot(t1,-g_i[280,:],color='r') #,alpha=0.6
# ax9.plot(t1, I_syn_e[num,:], color='tab:blue', lw=1.5 ,label='I_syn_exc')
# ax9.plot(t1, I_syn_i[num,:], color='red', lw=1.5 ,label='I_syn_inh') #,alpha=0.6
# # ax9.plot(t1, -g_i[num,:],'--' ,color='tab:red' ,lw=1.5 ,alpha=0.6 ,label='I_syn_inh') #
# ax9.plot(t1, (I_syn_e[num,:]+I_syn_i[num,:]), lw=1.5, color='fuchsia' ,label='I_Syn_t')
# ax9.set_ylabel("$I\;syn$", size=16)
# ax9.set_xlabel('$t(ms)$', size=16)
# ax8.set_xlim(1000,1100)
# ax8.set_ylim(-65,-50)

# ax9.set_xlim(1000,1100)
# # ax3.set_xlim(700,1100)
# ax8.xaxis.set_tick_params(labelsize=14)
# ax8.yaxis.set_tick_params(labelsize=14)
# ax9.xaxis.set_tick_params(labelsize=14)
# ax9.yaxis.set_tick_params(labelsize=14)
# ax8.legend(loc='upper right')
# ax9.legend(loc='upper right')
# ax8.grid()
# ax9.grid()
# ax8.set_xticks(np.arange(1000,1100,20))
# ax9.set_xticks(np.arange(1000,1100,20))
# folder_name = r'VS-codes/testdynamic'
# file_name = 'neuron exc,Td_exc='+str(Td_exc) +', Td_inh='+str(Td_inh) +'.png' 
# if not os.path.exists(folder_name):  # چک کردن وجود پوشه
#     os.makedirs(folder_name)
# file_path = os.path.join(folder_name, file_name)  
# plt.savefig(file_path, dpi=100, bbox_inches='tight')

# fig, (ax8,ax9) = plt.subplots(2,1, figsize=(10,8)) #figsize=(8,5)
# num = 83

# ax8.plot(t1,Vm[num,:],color='green',lw=1.5 , label='Voltage_exc')
# a3 = ax8.twinx()
# a3.plot(timespop1[np.where(senderspop1==num+1)],senderspop1[np.where(senderspop1==num+1)],'.',color='r',markersize=7.0)

# ax8.set_title("number of neuron"+str(num))
# ax8.set_ylabel("$Voltage$", size=16)
# ax8.set_xlabel('$t(ms)$', size=16)
# # ax8.plot(t1,Vm[83,:],color='r',lw=1.5 , label='Voltage_exc')
# # ax8.plot(t1,V_mean_inh,color='orange',lw=2.2 , label='mean_Voltage_Inh')
# # ax2.plot(t1,-g_i[280,:],color='r') #,alpha=0.6
# ax9.plot(t1, I_syn_e[num,:], color='tab:blue', lw=1.5 ,label='I_syn_exc')
# # ax9.plot(t1, I_syn_i[num,:], color='red', lw=1.5 ,label='I_syn_inh') #,alpha=0.6
# ax9.plot(t1, -I_syn_i[num,:],'--' ,color='tab:red' ,lw=1.5 ,alpha=0.6 ,label='I_syn_inh') #
# ax9.plot(t1, (I_syn_e[num,:]+I_syn_i[num,:]), lw=1.5, color='fuchsia' ,label='I_Syn_t')
# ax9.set_ylabel("$I\;syn$", size=16)
# ax9.set_xlabel('$t(ms)$', size=16)
# ax8.set_xlim(300,800)
# ax9.set_xlim(300,800)
# # ax3.set_xlim(700,1100)
# ax8.xaxis.set_tick_params(labelsize=14)
# ax8.yaxis.set_tick_params(labelsize=14)
# ax9.xaxis.set_tick_params(labelsize=14)
# ax9.yaxis.set_tick_params(labelsize=14)
# ax8.legend(loc='upper right')
# ax9.legend(loc='upper right')
# ax8.grid()
# ax9.grid()
# ax8.set_xticks(np.arange(300,800,50))
# ax9.set_xticks(np.arange(300,800,50))
# folder_name = r'VS-codes/testdynamic'
# file_name = 'neuron exc,Td_exc='+str(Td_exc) +', Td_inh='+str(Td_inh) +'.png' 
# if not os.path.exists(folder_name):  # چک کردن وجود پوشه
#     os.makedirs(folder_name)
# file_path = os.path.join(folder_name, file_name)  
# plt.savefig(file_path, dpi=100, bbox_inches='tight') 

# fig, (ax88,ax99) = plt.subplots(2,1, figsize=(10,8)) #figsize=(8,5)
# num1 = 285
# ax88.plot(t1,Vm[num1,:],color='green',lw=1.5 , label='Voltage_Inh')
# a4 = ax88.twinx()
# a4.plot(timespop1[np.where(senderspop1==(num1+1))],senderspop1[np.where(senderspop1==(num1+1))],'.',color='r',markersize=7.0)
# ax88.set_title("number of inh neuron"+str(num1))
# ax88.set_ylabel("$Voltage$", size=16)
# ax88.set_xlabel('$t(ms)$', size=16)
# # ax8.plot(t1,Vm[83,:],color='r',lw=1.5 , label='Voltage_exc')
# # ax8.plot(t1,V_mean_inh,color='orange',lw=2.2 , label='mean_Voltage_Inh')
# # ax2.plot(t1,-g_i[280,:],color='r') #,alpha=0.6
# ax99.plot(t1, I_syn_e[num1,:], color='tab:blue', lw=1.5 ,label='I_syn_exc')
# ax99.plot(t1, I_syn_i[num1,:], color='red', lw=1.5 ,label='I_syn_inh') #,alpha=0.6
# ax99.plot(t1, -I_syn_i[num1,:], '--', color='tab:red', alpha=0.6, lw=1.5 ,label='I_syn_inh') #
# ax99.plot(t1, (I_syn_e[num1,:]+I_syn_i[num1,:]), lw=1.5, color='fuchsia',label='I_Syn_t')
# ax99.set_ylabel("$I\;syn$", size=16)
# ax99.set_xlabel('$t(ms)$', size=16)
# ax88.set_xlim(1000,1100)
# ax99.set_xlim(1000,1100)
# # ax3.set_xlim(700,1100)
# ax88.xaxis.set_tick_params(labelsize=14)
# ax88.yaxis.set_tick_params(labelsize=14)
# ax99.xaxis.set_tick_params(labelsize=14)
# ax99.yaxis.set_tick_params(labelsize=14)
# ax88.legend(loc='upper right')
# ax99.legend(loc='upper right')
# ax88.grid()
# ax99.grid()
# ax88.set_xticks(np.arange(1000,1100,20))
# ax99.set_xticks(np.arange(1000,1100,20))

# folder_name = r'VS-codes/testdynamic'
# file_name = 'neuron inh,Td_exc='+str(Td_exc) +', Td_inh='+str(Td_inh) +'.png' 
# if not os.path.exists(folder_name):  # چک کردن وجود پوشه
#     os.makedirs(folder_name)
# file_path = os.path.join(folder_name, file_name)  
# plt.savefig(file_path, dpi=100, bbox_inches='tight')


###########################################################

# getot1 = np.mean(g_e,axis=0 )
# gitot1 = np.mean(g_i,axis=0 )

# ge_mean = np.mean(getot1)
# ge_mean = float("{:.2f}".format(ge_mean)) 
# gi_mean = np.mean(gitot1)
# gi_mean = float("{:.2f}".format(gi_mean)) 

####################################################################################################
B = 1.0
bins = np.arange(start, tmax+B, B)

te, b = np.histogram(timespop1[idxe1], bins)
tew = 1e3*te/(N_E*B)

ti, b = np.histogram(timespop1[idxi1], bins)
tiw = 1e3*ti/(N_I*B)

# ttot , b = np.histogram(timespop1, bins)
# ttot = 1e3*ttot/(N*B)
#######################################################################################################
sigma = 1.0

te1 = gaussian_filter1d(te, sigma=sigma, mode='reflect')*1e3/(N_E*B)
ti1 = gaussian_filter1d(ti, sigma=sigma, mode='reflect')*1e3/(N_I*B)

# g_win = (1/np.sqrt(2*np.pi*(sigma**2)))*scipy.signal.windows.gaussian(11, std=sigma)
# g_win1 = scipy.signal.windows.gaussian(10, std=sigma)
# g_win1 /= g_win1.sum()

# plt.figure(12)
# plt.plot(g_win,'.')
# plt.plot(g_win1,'.-',color='red')

# te1 = np.convolve(te1, g_win, mode='same')
# ti1 = np.convolve(ti1, g_win, mode='same')

# firing_rate_mean_exc = np.mean(te1)
# firing_rate_mean_inh = np.mean(ti1)
########################################  Fano factor  ###########################################
Fano_exc = np.var(te1)/np.mean(te1)
Fano_inh = np.var(ti1)/np.mean(ti1)
print("Fano exc = ",Fano_exc)
print("Fano inh = ",Fano_inh)
##############################################  PHASE  ######################################
fs = 1000.0 / B   # sampling frequency in Hz

# 1. PSD to find dominant peak in e.g. 30-200 Hz
f_exc, Pxx_exc = welch(tew, fs=fs, nperseg=min(2048, len(te)))
fmin, fmax = 20, 200
mask = (f_exc>=fmin) & (f_exc<=fmax)
if not mask.any():
    raise RuntimeError("Frequency range empty; check fs and signal length")
peak_freq_exc = f_exc[mask][np.argmax(Pxx_exc[mask])]
# print("Peak freq (E):", peak_freq_exc, "Hz")

# fig = plt.figure(figsize=(10 , 8))
# plt.semilogy(f_exc, Pxx_exc, color="k")
# plt.axvspan(30, 80, color="orange", alpha=0.3, label="Gamma")
# plt.axvspan(100, 180, color="blue", alpha=0.3, label="HFO")
# plt.xlim(0,200)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Power spectral density Exc")
# plt.legend()

# folder_name = r'VS-codes/testdynamic'
# file_name = 'neuron exc,Td_exc='+str(Td_exc) +', Td_inh='+str(Td_inh) +'.png' 
# if not os.path.exists(folder_name):  # چک کردن وجود پوشه
#     os.makedirs(folder_name)
# file_path = os.path.join(folder_name, file_name)  
# plt.savefig(file_path, dpi=100, bbox_inches='tight') 

f_inh, Pxx_inh = welch(tiw, fs=fs, nperseg=min(2048, len(te)))
fmin, fmax = 20, 200
mask = (f_inh>=fmin) & (f_inh<=fmax)
if not mask.any():
    raise RuntimeError("Frequency range empty; check fs and signal length")
peak_freq_inh = f_inh[mask][np.argmax(Pxx_inh[mask])]
# print("Peak freq (I):", peak_freq_inh, "Hz")


# fig = plt.figure(figsize=(10 , 8))
# plt.semilogy(f_inh, Pxx_inh, color="k")
# plt.axvspan(30, 80, color="orange", alpha=0.3, label="Gamma")
# plt.axvspan(100, 180, color="blue", alpha=0.3, label="HFO")
# plt.xlim(0,200)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Power spectral density Inh")
# plt.legend()

# # folder_name = r'VS-codes/assess'
# # file_name = 'welch_inh, Td_exc='+str(Td_exc) +', Td_inh='+str(Td_inh) +'.png' 
# # if not os.path.exists(folder_name):  # چک کردن وجود پوشه
# #     os.makedirs(folder_name)
# # file_path = os.path.join(folder_name, file_name)  
# # plt.savefig(file_path, dpi=120, bbox_inches='tight') 
#######################################################################################
def bandpass(x, low, high, fs, order=5):
    nyq = 0.5 * fs
    sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    y = sosfiltfilt(sos,x)
    return y

# use peak_freq from method (1)
bw = 10.0   # ± bandwidth in Hz, adjust
rE = tew - np.mean(tew)
rI = tiw - np.mean(tiw)
##############=======================   Def Find Peak Frequency with Welch method ====================================================
def Peak_band_freq (fmin, fmax, firng_rate, fs=fs):
    frequency, psd = welch(firng_rate, fs=fs, nperseg=min(2048, len(te)))
    mask = (frequency>=fmin) & (frequency<=fmax)
    if not mask.any():
        raise RuntimeError("Frequency range empty; check fs and signal length")
    peak_freq = frequency[mask][np.argmax(psd[mask])]
    return peak_freq

Peak_exc_gb = Peak_band_freq (30, 80, tew )
Peak_exc_hb = Peak_band_freq (100, 200, tew)
# print("Peak exc gamma band",Peak_exc_gb)
# print("Peak exc HFO band",Peak_exc_hb)
Peak_inh_gb = Peak_band_freq (30, 80, tiw )
Peak_inh_hb = Peak_band_freq (100, 200, tiw)
# print("Peak inh gamma band",Peak_inh_gb)
# print("Peak inh HFO band",Peak_inh_hb)

##======================================================
rE_g_f = bandpass(rE, Peak_exc_gb - bw, Peak_exc_gb + bw, fs)
rE_h_f = bandpass(rE, Peak_exc_hb - bw, Peak_exc_hb + bw, fs)

rI_g_f = bandpass(rI, Peak_inh_gb - bw, Peak_inh_gb + bw, fs)
rI_h_f = bandpass(rI, Peak_inh_hb - bw, Peak_inh_hb + bw, fs)

phiE_gb = np.angle(hilbert(rE_g_f))
phiE_hb = np.angle(hilbert(rE_h_f))

phiI_gb = np.angle(hilbert(rI_g_f))
phiI_hb = np.angle(hilbert(rI_h_f))

phase_diff_gb = np.angle(np.exp(1j*(phiE_gb - phiI_gb)))
phase_diff_hb = np.angle(np.exp(1j*(phiE_hb - phiI_hb)))
#================================================================================
rE_f = bandpass(rE, peak_freq_exc - bw, peak_freq_exc + bw, fs)
rI_f = bandpass(rI, peak_freq_inh - bw, peak_freq_inh + bw, fs)

phiE = np.angle(hilbert(rE_f))
phiI = np.angle(hilbert(rI_f))

# circular mean of phase difference
complex_mean = np.mean(np.exp(1j*(phiE - phiI)))
mean_diff = np.angle(complex_mean)  # rad
lag_ms = (mean_diff / (2*np.pi*peak_freq_exc)) * 1000.0
# print("Mean phase diff (E - I):", mean_diff, "rad  -> lag (ms):", lag_ms)

te_gamma = bandpass(tew, 30, 80, fs)
ti_gamma = bandpass(tiw, 30, 80, fs)
te_high  = bandpass(tew, 100, 200, fs)
ti_high  = bandpass(tiw, 100, 200, fs)

# Hilbert transform → phases
phase_te_gamma = np.angle(hilbert(te_gamma))
phase_ti_gamma = np.angle(hilbert(ti_gamma))
phase_te_high  = np.angle(hilbert(te_high))
phase_ti_high  = np.angle(hilbert(ti_high))

# envelope_te_gamma = np.abs(hilbert(te_gamma))
# envelope_ti_gamma = np.abs(hilbert(ti_gamma))
# envelope_te_high  = np.abs(hilbert(te_high))
# envelope_ti_high  = np.abs(hilbert(ti_high))


# plt.figure(figsize=(10, 5))
# plt.plot(envelope_te_gamma, label='Excitatory envelope', color='blue')
# plt.plot(te_gamma, label='Excitatory', color='k')
# plt.plot(envelope_ti_gamma, label='Inhibitory envelope', color='orange')
# plt.plot(ti_gamma, label='Inhibitory', color='r')
# plt.legend()
# plt.xlim(100,300)
# plt.xlabel('Time (ms)')
# plt.ylabel('Envelope amplitude')
# plt.title('Gamma band envelopes of excitatory and inhibitory populations')

# plt.figure(figsize=(10, 5))
# plt.plot(envelope_te_high, label='Excitatory envelope', color='blue')
# plt.plot(te_high, label='Excitatory', color='k')
# # plt.plot(bandpass(envelope_te_high, 30, 80, fs), label='XX', color='g')
# plt.plot(envelope_ti_high, label='Inhibitory envelope', color='orange')
# plt.plot(ti_high, label='EInhibitory', color='r')
# plt.legend()
# plt.xlim(100,300)
# plt.xlabel('Time (ms)')
# plt.ylabel('Envelope amplitude')
# plt.title('HFO band envelopes of excitatory and inhibitory populations')

# gamma_phase = np.angle(hilbert(te_gamma))   # فاز گاما
# hfo_amp   = np.abs(hilbert(ti_high))       # دامنه HFO

########################################
phase_diff_gamma = np.angle(np.exp(1j*(phase_te_gamma - phase_ti_gamma)))

phase_diff_high  = np.angle(np.exp(1j*(phase_te_high - phase_ti_high)))

mean_diff_gamma = np.angle(np.mean(np.exp(1j*phase_diff_gamma)))
mean_diff_high  = np.angle(np.mean(np.exp(1j*phase_diff_high)))

# print("Mean E-I phase diff (Gamma):", mean_diff_gamma, "rad")
# print("Mean E-I phase diff (High):", mean_diff_high, "rad")
########################################################################
#####################################################################
# plt.figure(figsize=(12,6))
# ax = plt.subplot(121, polar=True)
# n_binn = 40
# counts, binsz = np.histogram(phase_diff_gamma, bins=n_binn, range=(-np.pi, np.pi))
# ax.bar(binsz[:-1], counts, width=(2*np.pi/n_binn), bottom=0.0, color='skyblue', edgecolor='k', alpha=0.7)
# ax.set_title("Circular histogram of E-I phase difference (Gamma)")

# ax = plt.subplot(122, polar=True)
# n_binn = 40
# counts, binsz = np.histogram(phase_diff_high, bins=n_binn, range=(-np.pi, np.pi))
# ax.bar(binsz[:-1], counts, width=(2*np.pi/n_binn), bottom=0.0, color='skyblue', edgecolor='k', alpha=0.7)
# ax.set_title("Circular histogram of E-I phase difference (HFO)")
##################################################################################
# fig, axs = plt.subplots(1, 2, figsize=(12,4.0), subplot_kw={'projection':'polar'},dpi=120)
# plt.subplots_adjust(wspace=0.25)

# axs[0].hist(phase_diff_gamma, bins=50, density=True, bottom=0.0, color="mediumseagreen", alpha=0.8, edgecolor='k')
# axs[0].set_title("Gamma Band", va='bottom', y=1.25, fontsize=45 )

# axs[1].hist(phase_diff_high, bins=50, density=True, bottom=0.0, color="mediumseagreen", alpha=0.8, edgecolor='k')
# axs[1].set_title("HFO Band", va='bottom', y=1.25, fontsize=45)

# # axs[0].set_yticks(np.arange(0, 3.6, 1.0))
# # axs[1].set_yticks(np.arange(0, 2.1, 0.5))
# for ax in axs:
#     # تنظیم سایز اعداد روی محور شعاعی
#     ax.tick_params(axis='y', labelsize=26, labelcolor='k' )  # بزرگ کردن اعداد شعاع
#     ax.tick_params(axis='x', labelsize=26, pad=15.0)  # بزرگ کردن اعداد زاویه
#     ax.set_rlabel_position(135)
######################### one polar ################

# fig, ax = plt.subplots(figsize=(5, 6),subplot_kw={'projection': 'polar'}, dpi=120)
# plt.subplots_adjust(wspace=0.25)
# ax.hist(
#     phase_diff_high,
#     bins=50,
#     density=True,
#     bottom=0.0,
#     color="mediumseagreen",
#     alpha=0.8,
#     edgecolor='k'
# )
# ax.set_title("HFO Band", va='bottom', y=1.25, fontsize=45)
# ax.tick_params(axis='y', labelsize=26, labelcolor='k')
# ax.tick_params(axis='x', labelsize=26,  pad=15.0)
# ax.set_rlabel_position(135)
####################################################

# folder_name = r'VS-codes/assess/new'
# file_name = 'polar, tau_exc='+str(tau_exc_exc) +', tau_inh='+str(tau_inh_inh) +'new.png' 
# if not os.path.exists(folder_name):  # چک کردن وجود پوشه
#     os.makedirs(folder_name)
# file_path = os.path.join(folder_name, file_name)  
# plt.savefig(file_path, dpi=140, bbox_inches='tight') 

######################################## Delta Phi that Filtered signals around in specific band #######################################################

# t = np.arange(len(te)) / fs
# # plt.figure(figsize=(12,4))
# # plt.plot(t, rE_f, label='E filtered')
# # plt.plot(t, rI_f, label='I filtered', alpha=0.7)
# # plt.xlim(0.8,1)
# # plt.xlabel('Time (s)')
# # plt.ylabel('Amplitude')
# # plt.title(f'Filtered signals around {peak_freq:.1f} Hz')
# # plt.legend()


# # Phase difference
# phiE = np.angle(hilbert(rE_f))
# phiI = np.angle(hilbert(rI_f))
# phase_diff = np.angle(np.exp(1j*(phiE - phiI)))

# plt.figure(figsize=(10,4))
# plt.plot(t, phase_diff_gb, color='purple', linewidth=3.0 )
# plt.plot(t, phase_diff_hb, color='k', linewidth=3.0 )
# plt.axhline(np.angle(np.mean(np.exp(1j*phase_diff))), color='red', linestyle='--',dashes=(6,4),linewidth=2.0)   #, label='Mean phase diff'
# plt.xlabel('t (s)',fontsize=40)
# plt.ylabel('$\Delta \phi (rad)$',fontsize=40)
# # plt.xlim(-0.1,10.1)
# # plt.title('Instantaneous E-I Phase Difference',fontsize=20)
# plt.gca().xaxis.set_tick_params(labelsize=35, pad=8, width=1.8 )
# plt.gca().yaxis.set_tick_params(labelsize=33, pad=8, width=1.8 )

# plt.gca().xaxis.set_major_locator(MultipleLocator(2))
# plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
# plt.gca().yaxis.set_major_locator(MultipleLocator(2.0))
# plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
# plt.ylim(-3.14,+3.14)
# plt.tick_params(which='major', length=16, width=1.8, color='k')
# plt.tick_params(which='minor', length=8, width=1.0, color='k')
# for spine in plt.gca().spines.values():
#     spine.set_linewidth(1.0)


# # folder_name = r'VS-codes/assess/new'
# # file_name = 'Phase difference, tau_exc='+str(tau_exc_exc) +', tau_inh='+str(tau_inh_inh) +'new.png' 
# # if not os.path.exists(folder_name):  # چک کردن وجود پوشه
# #     os.makedirs(folder_name)
# # file_path = os.path.join(folder_name, file_name)  
# # plt.savefig(file_path, dpi=120, bbox_inches='tight') 
# ############=============================================================
# fig, axs = plt.subplots(2, 1)
# fig.subplots_adjust(hspace=0.8)
# axs[0].plot(t, phase_diff_hb, color='purple', linewidth=3.0 )
# axs[1].plot(t, phase_diff_gb, color='purple', linewidth=3.0 )

# axs[0].axhline(np.angle(np.mean(np.exp(1j*phase_diff_hb))), color='red', linestyle='--',dashes=(6,4),linewidth=2.0)
# axs[1].axhline(np.angle(np.mean(np.exp(1j*phase_diff_gb))), color='red', linestyle='--',dashes=(6,4),linewidth=2.0)
# # axs[0].set_xlabel('t (s)',fontsize=40)
# axs[0].set_ylabel('$\Delta \phi (rad)$',fontsize=40)
# axs[1].set_xlabel('t (s)',fontsize=40)
# axs[1].set_ylabel('$\Delta \phi (rad)$',fontsize=40)
# # plt.xlim(-0.1,10.1)
# # plt.title('Instantaneous E-I Phase Difference',fontsize=20)
# for i in range(2):
#     axs[i].xaxis.set_tick_params(labelsize=35, pad=8, width=1.8 )
#     axs[i].yaxis.set_tick_params(labelsize=33, pad=8, width=1.8 )

#     axs[i].xaxis.set_major_locator(MultipleLocator(2))
#     axs[i].xaxis.set_minor_locator(MultipleLocator(0.5))
    
#     axs[i].yaxis.set_major_locator(MultipleLocator(2.0))
#     axs[i].yaxis.set_minor_locator(MultipleLocator(0.5))
#     axs[i].set_ylim(-3.14,+3.14)

#     for spine in axs[i].spines.values():
#         spine.set_linewidth(1.0)

#     axs[i].tick_params(which='major', length=16, width=1.8, color='k')
#     axs[i].tick_params(which='minor', length=8, width=1.0, color='k')


# # # folder_name = r'VS-codes/assess/new'
# # # file_name = 'Phase difference_4.png' 
# # # if not os.path.exists(folder_name):  # چک کردن وجود پوشه
# # #     os.makedirs(folder_name)
# # # file_path = os.path.join(folder_name, file_name)  
# # # plt.savefig(file_path, dpi=120, bbox_inches='tight') 
# #=============================================================
# from matplotlib.gridspec import GridSpec
# fig = plt.figure(figsize=(15.0,9.5), dpi=120)
# gs = GridSpec(7, 3, 
#               width_ratios=[2.,0.2, 3.0],
#               height_ratios=[0.2, 3.0, 0.02, 0.01, 0.02, 3.0, 0.2],
#              hspace=1.0,
#              #wspace=0.25
#               )  # نسبت عرض‌ها

# ax1 = fig.add_subplot(gs[0:3,0], projection='polar')
# ax1p = fig.add_subplot(gs[4:,0], projection='polar')
# ax2 = fig.add_subplot(gs[1,2])
# ax2p = fig.add_subplot(gs[5,2])

# # محور اول: پلار
# #####################
# ax1.hist(phase_diff_gamma, bins=50, density=True, bottom=0.0, color="mediumseagreen", alpha=0.8, edgecolor='k')

# ax1p.hist(phase_diff_high, bins=50, density=True, bottom=0.0, color="mediumseagreen", alpha=0.8, edgecolor='k')


# ax1.text(-0.51, -0.15, "Gamma Band",
#          transform=ax1.transAxes,
#          fontsize=45,
#          ha='left',
#          va='center',rotation=90, rotation_mode='anchor', transform_rotates_text=True)

# ax1p.text(-0.51, 0.12, "HFO Band",
#           transform=ax1p.transAxes,
#           fontsize=45,
#           ha='left',
#           va='center',rotation=90, rotation_mode='anchor', transform_rotates_text=True)

# ax1.tick_params(axis='y', labelsize=26, labelcolor='k' )  # بزرگ کردن اعداد شعاع
# ax1.tick_params(axis='x', labelsize=26, pad=15.0)  # بزرگ کردن اعداد زاویه
# ax1.set_rlabel_position(135)
# ax1p.tick_params(axis='y', labelsize=26, labelcolor='k' )  # بزرگ کردن اعداد شعاع
# ax1p.tick_params(axis='x', labelsize=26, pad=15.0)  # بزرگ کردن اعداد زاویه
# ax1p.set_rlabel_position(135)
# #####################
# # ax1.hist(phase_diff_high, bins=50, density=True, bottom=0.0, color="mediumseagreen", alpha=0.8, edgecolor='k')
# # ax1.set_title("Gamma Band", va='bottom', y=1.25, fontsize=45)
# # ax1.tick_params(axis='y', labelsize=26, labelcolor='k' )  # بزرگ کردن اعداد شعاع
# # ax1.tick_params(axis='x', labelsize=26, pad=15.0)  # بزرگ کردن اعداد زاویه
# # ax1.set_rlabel_position(135)
# # محور دوم: معمولی

# ax2.plot(t, phase_diff_gb, color='purple', linewidth=3.0 )
# ax2.axhline(np.angle(np.mean(np.exp(1j*phase_diff_gb))), color='red', linestyle='--',dashes=(6,4),linewidth=2.0)   #, label='Mean phase diff'
# # ax2.set_xlabel('t (s)',fontsize=40)
# ax2.set_ylabel('$\Delta \phi (rad)$',fontsize=40)
# # plt.xlim(-0.1,10.1)
# # plt.title('Instantaneous E-I Phase Difference',fontsize=20)
# ax2.xaxis.set_tick_params(labelsize=35, pad=8, width=1.8 )
# ax2.yaxis.set_tick_params(labelsize=33, pad=8, width=1.8 )

# ax2.xaxis.set_major_locator(MultipleLocator(5))
# ax2.xaxis.set_minor_locator(MultipleLocator(2.5))
# ax2.yaxis.set_major_locator(MultipleLocator(2.0))
# ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
# ax2.set_ylim(-3.14,+3.14)
# ax2.tick_params(which='major', length=16, width=1.8, color='k')
# ax2.tick_params(which='minor', length=8, width=1.0, color='k')
# for spine in ax2.spines.values():
#     spine.set_linewidth(1.0)


# ax2p.plot(t, phase_diff_hb, color='purple', linewidth=3.0 )
# ax2p.axhline(np.angle(np.mean(np.exp(1j*phase_diff_hb))), color='red', linestyle='--',dashes=(6,4),linewidth=2.0)   #, label='Mean phase diff'
# ax2p.set_xlabel('t (s)',fontsize=40)
# ax2p.set_ylabel('$\Delta \phi (rad)$',fontsize=40)

# # plt.title('Instantaneous E-I Phase Difference',fontsize=20)
# ax2p.xaxis.set_tick_params(labelsize=35, pad=8, width=1.8 )
# ax2p.yaxis.set_tick_params(labelsize=33, pad=8, width=1.8 )

# ax2p.xaxis.set_major_locator(MultipleLocator(5))
# ax2p.xaxis.set_minor_locator(MultipleLocator(2.5))
# ax2p.yaxis.set_major_locator(MultipleLocator(2.0))
# ax2p.yaxis.set_minor_locator(MultipleLocator(0.5))
# ax2p.set_ylim(-3.14,+3.14)
# ax2p.tick_params(which='major', length=16, width=1.8, color='k')
# ax2p.tick_params(which='minor', length=8, width=1.0, color='k')
# for spine in ax2p.spines.values():
#     spine.set_linewidth(1.0)
# plt.subplots_adjust(wspace=0.25)
# folder_name = r'VS-codes/assess/new'
# file_name = 'ph4newf.png' 
# if not os.path.exists(folder_name):  # چک کردن وجود پوشه
#     os.makedirs(folder_name)
# file_path = os.path.join(folder_name, file_name)  
# plt.savefig(file_path, dpi=120, bbox_inches='tight') 
################################################## one polar and Delta phi ############################
# from matplotlib.gridspec import GridSpec
# fig = plt.figure(figsize=(16, 4.5), dpi=120)
# gs = GridSpec(3, 2, 
#               width_ratios=[3.0, 3.0],
#               height_ratios=[3,0.75, 0.75],
#             #  hspace= 0.5,
#              #wspace=0.25
#               )  # نسبت عرض‌ها

# ax1 = fig.add_subplot(gs[0:2,0], projection='polar')

# ax2 = fig.add_subplot(gs[0,1])

# # محور اول: پلار
# #####################
# ax1.hist(phase_diff, bins=50, density=True, bottom=0.0, color="mediumseagreen", alpha=0.8, edgecolor='k')
# # ax1.set_title("gamma Band", va='bottom', y=1.25, fontsize=45)
# ax1.tick_params(axis='y', labelsize=26, labelcolor='k' )  # بزرگ کردن اعداد شعاع
# ax1.tick_params(axis='x', labelsize=26, pad=15.0)  # بزرگ کردن اعداد زاویه
# ax1.set_rlabel_position(135)

# ax1.text(-0.5, -0.0, "HFO Band",
#          transform=ax1.transAxes,
#          fontsize=45,
#          ha='left',
#          va='center',rotation=90, rotation_mode='anchor', transform_rotates_text=True)
# ax2.plot(t, phase_diff, color='purple', linewidth=3.0 )
# ax2.axhline(np.angle(np.mean(np.exp(1j*phase_diff))), color='red', linestyle='--',dashes=(6,4),linewidth=2.0)   #, label='Mean phase diff'
# ax2.set_xlabel('t (s)',fontsize=40)
# ax2.set_ylabel('$\Delta \phi (rad)$',fontsize=40)
# # plt.xlim(-0.1,10.1)
# # plt.title('Instantaneous E-I Phase Difference',fontsize=20)
# ax2.xaxis.set_tick_params(labelsize=35, pad=8, width=1.8 )
# ax2.yaxis.set_tick_params(labelsize=33, pad=8, width=1.8 )

# ax2.xaxis.set_major_locator(MultipleLocator(5))
# ax2.xaxis.set_minor_locator(MultipleLocator(2.5))
# ax2.yaxis.set_major_locator(MultipleLocator(1.0))
# ax2.yaxis.set_minor_locator(MultipleLocator(0.2))
# plt.ylim(-1.5,+1.5)
# ax2.tick_params(which='major', length=16, width=1.8, color='k')
# ax2.tick_params(which='minor', length=8, width=1.0, color='k')
# for spine in ax2.spines.values():
#     spine.set_linewidth(1.0)
# plt.subplots_adjust(wspace=0.25)


# folder_name = r'VS-codes/assess/new'
# file_name = 'ph2new.png' 
# if not os.path.exists(folder_name):  # چک کردن وجود پوشه
#     os.makedirs(folder_name)
# file_path = os.path.join(folder_name, file_name)  
# plt.savefig(file_path, dpi=120, bbox_inches='tight') 

############################################################################################################
# # lags = signal.correlation_lags(te1.size, te1.size, mode="full")*B
# # # lags = np.arange(-len(te1)+1,len(te1))*B
# # # print(len(lags),len(te1))
# # e1 = bandpass(rE,129,132, fs)
# # i1 = bandpass(rI,129,132, fs)
# # cc = correlate(rE , rI , mode='full')
# # cc = cc / np.max(cc) 

# # lag_at_peak = lags[np.argmax(cc)]
# # time_shift = lag_at_peak 

# # print("Peak lag:", lag_at_peak, "ms")
# # print("time shift", time_shift)


# fig, (ax1, ax2) = plt.subplots(1, 2,
#                                subplot_kw=dict(projection='polar'),
#                                figsize=(12,6), dpi=120)

# # Gamma
# ax1.hist(phase_diff_gamma, bins=50, density=True)
# ax1.set_title("Gamma (30–80 Hz)")
# # ax1.set_rlabel_position(135)        # جابه‌جایی اعداد دامنه
# for label in ax1.get_yticklabels():
#     label.set_fontsize(10) 
# ax1.set_rticks(np.arange(0.5,2.6, 0.5))

# # High freq
# ax2.hist(phase_diff_high, bins=50, density=True)
# ax2.set_title("High freq (100–200 Hz)")
# for label in ax1.get_yticklabels():
#     label.set_fontsize(10)       
# ax2.set_rticks(np.arange(0.5,1.1,0.5))
# # folder_name = r'VS-codes/assess'
# # file_name = 'polar, Td_exc='+str(Td_exc) +', Td_inh='+str(Td_inh) +'.png' 
# # if not os.path.exists(folder_name):  # چک کردن وجود پوشه
# #     os.makedirs(folder_name)
# # file_path = os.path.join(folder_name, file_name)  
# # plt.savefig(file_path, dpi=120, bbox_inches='tight') 


# max_lag = 100  # ms
# mask = np.abs(lags) <= max_lag

# ax30.plot(lags[mask], cc[mask], lw=1.5, color="k")

# for spine in ax30.spines.values():
#     spine.set_linewidth(1.2)


# for spine in ['top', 'right']:
#      ax30.spines[spine].set_visible(False)


# ax30.xaxis.set_tick_params(labelsize=25, pad=8, width=2.5, length=16)
# ax30.yaxis.set_tick_params(labelsize=25, pad=8, width=2.5, length=16)
    
# ax5.yaxis.set_major_locator(MultipleLocator(5))
# ax5.yaxis.set_major_formatter('{x:.0f}')
# ax5.yaxis.set_minor_locator(MultipleLocator(2.5))

# ax30.xaxis.set_major_locator(MultipleLocator(10))
# ax30.xaxis.set_major_formatter('{x:.0f}')
# ax30.xaxis.set_minor_locator(MultipleLocator(2))

# ax30.tick_params(which='major', length=16, width=2, color='k')
# ax30.tick_params(which='minor', length=8, width=1.2, color='k')

# plt.axvline(0, color='r', linestyle='--')
# ax30.set_xlabel("Lag (ms)", fontsize=30, labelpad=14 )
# ax30.set_ylabel("Cross-correlation", fontsize=30, labelpad=14 )
# ax30.set_xlim(-max_lag,max_lag)
# ax30.set_title("E-I cross-correlation", fontsize=30)
# # folder_name = r'VS-codes/spectrogram/ING'
# # file_name = 'cross-correlation,tau_in=2.png' 
# # if not os.path.exists(folder_name):  # چک کردن وجود پوشه
# #     os.makedirs(folder_name)
# # file_path = os.path.join(folder_name, file_name)  
# # plt.savefig(file_path, dpi=120, bbox_inches='tight')
# # ########################################################################################################
fig = plt.figure(figsize=(12, 11))
gs = gridspec.GridSpec(2, 1, figure=fig,
                    #     width_ratios=[1.5, 0.25,1.],     # Column 0 is wider
                        height_ratios=[1.5, 1.5]        # Row 1 is taller
                       )

# Big plot spanning two columns
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

# fig, axes1 = plt.subplots(2,1, figsize=(10,10),dpi=100.0)
began = 1000
end = 1100
fig.subplots_adjust(hspace=0.2)
# setup(ax1)
# setup(ax2)
# setup(axes1[2])
ax1.plot(timespop1[idxe1],senderspop1[idxe1],'.',color='b',markersize=4.0)
ax1.plot(timespop1[idxi1],senderspop1[idxi1],'.',color='r',markersize=4.0)
ax1.axhline(280,linestyle="dashed" ,color="green")
ax1.set_xlim(began,end)
for spine in ['top', 'right']:
     ax1.spines[spine].set_visible(False)
# axes3 = ax1.twinx()
# ax2.plot(bins[:-1], te*1e3/(N_E*B) , color="blue" , alpha = 0.45, lw=2.5,label="sigma=0.")
# ax2.plot(bins[:-1], ti*1e3/(N_I*B) , color="red" , alpha = 0.45, lw=2.5,label="sigma=0.")

ax2.plot(bins[:-1], te1 , alpha = 0.8, lw=2.5, color="blue" )
# ax2.plot(bins[:-1], tew , alpha = 0.6, lw=2.5, color="orange" )
ax2.plot(bins[:-1], ti1 , alpha = 0.8, lw=2.5, color="red" )
# ax2.plot(bins[:-1], tiw , alpha = 0.6, lw=2.5, color="g" )

# ax2.fill_between(bins[:-1], ti1,  color='red', alpha=0.4)
# ax2.fill_between(bins[:-1], te1,  color='black', alpha=0.6)


ax2.set_yticks(np.arange(0, np.max(ti1)+2, 100))
ax2.set_ylim(0.,(np.max(tiw)+10))
ax2.set_ylabel('Rate(Hz)', size=30, labelpad=14)  # fontweight='bold'

# axes[0].set_xlabel('t(ms)',size=18)
ax1.set_ylabel('Neuron Index', size=30, labelpad=14 )
# ax1.xaxis.set_tick_params(labelsize=15, pad = 8)
# ax1.yaxis.set_tick_params(labelsize=15, pad = 8)
ax1.set_xlim(began, end)
# ax1.set_xticks(np.arange(began,end+5,50))

# ax1.set_yticks(np.arange(0,300+5,100))
ax1.tick_params(labelsize=25, axis="x", width=2, length=12, pad = 8.0 )  #, pad = 9.0
ax1.tick_params(labelsize=25, axis="y", width=2, length=12, pad = 8.0 )  #, pad = 9.0

ax1.yaxis.set_major_locator(MultipleLocator(100))
ax1.yaxis.set_major_formatter('{x:.0f}')
ax1.yaxis.set_minor_locator(MultipleLocator(50))

# ax1.set_ylim(-0.5,300.5)v@@@@@@@@@@@@@@@@@@@


ax1.xaxis.set_major_locator(MultipleLocator(50))
ax1.xaxis.set_major_formatter('{x:.0f}')
ax1.xaxis.set_minor_locator(MultipleLocator(10))
ax1.tick_params(which='major', length=16, width=2.5, color='k')
ax1.tick_params(which='minor', length=8, width=2, color='k')
# ax1.set_xticklabels([])


for spine in ['top', 'right']:
     ax2.spines[spine].set_visible(False)

# ax2.set_ylim(-0.5,400) @@@@@@@@@@
ax2.xaxis.set_major_locator(MultipleLocator(50))
ax2.xaxis.set_major_formatter('{x:.0f}')
ax2.xaxis.set_minor_locator(MultipleLocator(10))

ax2.yaxis.set_major_locator(MultipleLocator(100))
ax2.yaxis.set_major_formatter('{x:.0f}')
ax2.yaxis.set_minor_locator(MultipleLocator(50))

ax2.xaxis.set_tick_params(labelsize=25, pad=10, width=2, length=6)
ax2.yaxis.set_tick_params(labelsize=25, pad=10, width=2, length=6)
ax2.tick_params(which='major', length=16, width=2.5, color='k')
ax2.tick_params(which='minor', length=8, width=2, color='k')

ax2.set_xlabel('t(ms)', fontsize=30, labelpad=14 )
# ax2.set_ylabel("$\mathbf{Mean\;I_{syn}}$", fontsize=15)

# ax2.set_xticks(np.arange(began,end+5,50))
ax2.set_xlim(began-2,end+2)
# ax2.set_yticks(np.arange(-500,510,250))
# ax2.set_xticks(np.arange(began,end+5,50),fontsize=14)


# ax2.legend(loc='upper right')
for spine in ax1.spines.values():
    spine.set_linewidth(1.2)
for spine in ax2.spines.values():
    spine.set_linewidth(1.2)

# folder_name = r'VS-codes/spectrogram/ING'
# file_name = 'raster,PING,tau_in_in=2,tau_ex_in=8.png' 
# if not os.path.exists(folder_name):  # چک کردن وجود پوشه
#     os.makedirs(folder_name)
# file_path = os.path.join(folder_name, file_name)  
# plt.savefig(file_path, dpi=120, bbox_inches='tight')

# folder_name = r'VS-codes/assess/new'
# file_name = 'raster, tau_exc='+str(tau_exc_exc) +', tau_inh='+str(tau_inh_inh) +'.png' 
# if not os.path.exists(folder_name):  # چک کردن وجود پوشه
#     os.makedirs(folder_name)
# file_path = os.path.join(folder_name, file_name)  
# plt.savefig(file_path, dpi=150, bbox_inches='tight') 

# plt.legend(fontsize=12)

# folder_name = r'Desktop'

# file_name = 'raster_t_d='+str(Td_inh)+'.png'

# if not os.path.exists(folder_name): 
#     os.makedirs(folder_name)

# file_path = os.path.join(folder_name, file_name)  
# plt.savefig(file_path, dpi=200, bbox_inches='tight')  

# # setup(ax)
# ax.plot(timespop1[idxe1],senderspop1[idxe1],'.',color='k',markersize=3.0)
# ax.plot(timespop1[idxi1],senderspop1[idxi1],'.',color='r',markersize=3.0)
# ax.set_xlim(start,tmax)
# ax.set_xlabel('t(ms)',size=16)
# ax.set_ylabel('#neuron',size=16)
# plt.yticks(fontsize=12)
# plt.xticks(np.arange(start,tmax,20),fontsize=12)

##############  WINDOW  ###############

# fig, (ax3,ax4,ax5) = plt.subplots(3, 1, figsize=(8,8.5))
# # ax3.plot(bins[:-1], te1g, color="blue" , label= 'exc') #, drawstyle="steps-mid"
# ax3.plot(bins[:-1], te1, lw=1.5, color="b" , label= 'exc')
# # ax3.plot(bins[:-1], te1g, color="r" , label= 'exc')
# # ax3.plot(bins[:-1], te1gg, color="k" , label= 'exc')
# # ax3.plot(bins[:-1], instantaneous_amplitude, 'k--', label="Instantaneous Amplitude")
# ax3.hlines(firing_rate_mean_exc, xmin=start, xmax=tmax, colors='purple', linestyles='--', lw=1.5, label= 'firing_rate_mean_exc' )
# # ax3.plot(bins[:-1], x_env, color="g" , label= 'exc')
# # ax3.plot(bins[:-1], instantaneous_phase, label= "Instantaneous Phase")
# ax3.set_xlim(1300,1500)
# #plt.fill_between(bins[:-1], te*tmax/(B*N_E), y2=0, color ='blue',alpha=0.2) 
# ax3.legend()
# ax4.plot(bins[:-1], ti1, color="red", lw=2, label= 'inh' )#, drawstyle="steps-mid"
# # ax4.plot(bins[:-1], ti1g, color="green",label= 'inh' )#, drawstyle="steps-mid"
# ax4.hlines(firing_rate_mean_inh ,xmin=start, xmax=tmax,  colors='purple', linestyles='--', lw=1.5, label="firing_rate_mean_inh" )
# # plt.fill_between(bins[:-1],y1=ti*tmax/(B*N_I),y2=0,color ='red',alpha=0.2)
# ax4.set_xlim(1300,1500)
# ax4.legend()
# ax5.plot(bins[:-1],ttot1,color="g", lw=1.5, label = 'total' )
# ax5.set_xlim(1300,1500)
# plt.legend()

# folder_name = r'compare'
# # folder_name = 'D:\\path\\to\\your\\folder'
# file_name = 'firingrate_td_4.png'  # نام فایل

# if not os.path.exists(folder_name):  # چک کردن وجود پوشه
#     os.makedirs(folder_name)
# file_path = os.path.join(folder_name, file_name)  # تنظیم مسیر کامل فایل
# plt.savefig(file_path, dpi=200, bbox_inches='tight')  # ذخیره‌ی پلات


###############################################
# def autocorr_peak_strength(firing_rate):
#      ac = correlate(firing_rate,firing_rate , mode='full')
#      mid =len(ac)//2
#      ac = ac[mid+1:]  #skip zero lag
#      peak = np.max(ac)
#      return peak/np.std(ac)
# print("autocorr_peak_strength=",autocorr_peak_strength(te1),autocorr_peak_strength(ti1))


# def ac(signal):
#      ac = correlate( signal, signal, mode='full') /tmax
#      ac = np.max(ac)
#      nu = np.mean(signal)
#      # mid =len(ac)//2
#      ac_norm = ac/((nu)**2 )
#      return ac_norm
# lags = np.arange(-tmax+1,tmax)
# print(ac(te))
# plt.figure(14,figsize=(8, 4))
# plt.plot(lags, ac(te),lw=1.5)
# plt.xlim(-50,50)

########################################################
# محاسبه‌ی STFT
# fs = 800
# f, t_stft, Zxxe = signal.stft(te1, fs, nperseg=256)

# # نمایش طیف STFT
# plt.figure(14,figsize=(8, 4))
# plt.pcolormesh(t_stft, f, np.abs(Zxxe), shading='gouraud')
# plt.colorbar(label="Power")
# plt.ylim(10, 150)  # نمایش فرکانس‌های 10 تا 100 هرتز
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.title("Short-Time Fourier Transform (STFT) E")

# f, t_stft, Zxxi = signal.stft(ti1, fs, nperseg=256)

# plt.figure(15,figsize=(8, 4))
# plt.pcolormesh(t_stft, f, np.abs(Zxxi), shading='gouraud')
# plt.colorbar(label="Power")
# plt.ylim(10, 150)  # نمایش فرکانس‌های 10 تا 100 هرتز
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.title("Short-Time Fourier Transform (STFT) I")
############################################################

Nn = len(te1)
dt0 = B*1e-3
fhat_exc = np.fft.fft(te1)
# print(Nn,len(fhat1),fhat1.size)
PSD_exc = 2*(np.abs(fhat_exc)**2)*(dt0/Nn) 
PSD_exc = PSD_exc - np.mean(PSD_exc)
freq = np.fft.fftfreq(fhat_exc.size,dt0 )      #(1000/(dt0*n))*np.arange(n)
L = np.arange(1,np.floor(Nn/2), dtype='int' )
# np.savetxt('VS-codes/P_connection/f_test.txt',freq[L])


fhat_inh = np.fft.fft(ti1)
PSD_inh = 2*(np.abs(fhat_inh)**2)*(dt0/Nn) 

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


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

# freqx, power_e, PSD_exc_s = extract_dominant_frequency(
#     PSD_exc,
#     freq,
#     band_low=0,
#     band_high=200,
#     sigma_hz=2,
#     min_prominence=0.01
# )
# freqx, power_i, PSD_inh_s = extract_dominant_frequency(
#     PSD_exc,
#     freq,
#     band_low=0,
#     band_high=200,
#     sigma_hz=2,
#     min_prominence=0.01
# )

# print(extract_dominant_frequency(
#     PSD_exc,
#     freq,
#     band_low=0,
#     band_high=200,
#     sigma_hz=2,
#     min_prominence=0.01
# ))
# print(extract_dominant_frequency(
#     PSD_inh,
#     freq,
#     band_low=0,
#     band_high=200,
#     sigma_hz=2,
#     min_prominence=0.01
# ))


#     return freq_band[idx_max], psd_band[idx_max]
# psd_smoothed_exc = savgol_filter(PSD_exc, window_length=11, polyorder=3)  # تنظیم window_length بر اساس داده
# psd_smoothed_inh = savgol_filter(PSD_inh, window_length=11, polyorder=3)
# freq_signal_exc, power_signal_exc = max_peak_with_threshold(PSD_exc, freq, 0, 200)
# print('freq=', freq_signal_exc, "power=", power_signal_exc)
# print('freq=', freq_signal_exc, "power=", power_signal_exc)
# freq_signal_inh, power_signal_inh = max_peak_with_threshold(PSD_inh, freq, 0, 200)
# print('freq=', freq_signal_inh, "power=", power_signal_inh)
# print('freq=', freq_signal_inh, "power=", power_signal_inh)
########################################### gamma excitatory ########################################
# gamma_pass_exc = (30.0 <= np.abs(freq[L])) & (np.abs(freq[L]) <= 80.0)
# psd_gamma_exc = PSD_e1[L]*gamma_pass_exc
# f_gamma_exc = freq[L]*gamma_pass_exc
# mean_power_exc = np.mean(PSD_e1[L])
# std_power_exc = np.std(PSD_e1[L])
# threshold_exc = mean_power_exc + 2.5 * std_power_exc  
# # print('threshold=',threshold_exc)
# # power = max(psd) if max(psd) > threshold else 0
# if max(psd_gamma_exc) > threshold_exc :
#      power_gamma_exc = max(psd_gamma_exc)
#      index_freq_exc = psd_gamma_exc.argmax()
#      f_gamma_exc = freq[index_freq_exc+1]
# else:
#      f_gamma_exc = 0
#      power_gamma_exc = 0


def pass_filter_exc (low, high, k):
     freq_pass = (low <= np.abs(freq[L])) & (np.abs(freq[L]) <= high) 
     psd_band = PSD_exc[L]*freq_pass
     mean_power = np.mean(PSD_exc[L])
     std_power = np.std(PSD_exc[L])
     threshold = mean_power + k * std_power  
     if max(psd_band) > threshold :
          power_signal = max(psd_band)
          index_freq = psd_band.argmax()
          freq_signal = freq[index_freq+1]
     else:
          freq_signal = 0
          power_signal = 0
     return freq_signal, power_signal

def pass_filter_inh (low, high, k):
     freq_pass = (low <= np.abs(freq[L])) & (np.abs(freq[L]) <= high) 
     psd_band = PSD_inh[L]*freq_pass
     mean_power = np.mean(PSD_inh[L])
     std_power = np.std(PSD_inh[L])
     threshold = mean_power + k * std_power  
     if max(psd_band) > threshold :
          power_signal = max(psd_band)
          index_freq = psd_band.argmax()
          freq_signal = freq[index_freq+1]
     else:
          freq_signal = 0
          power_signal = 0
     return freq_signal, power_signal


# f_gamma_exc, power_gamma_exc = pass_filter_exc(30, 80, k=2.5)
# f_hfo_exc, power_hfo_exc = pass_filter_exc(100, 200, k=2.5)
# f_gamma_inh, power_gamma_inh = pass_filter_inh(30, 80, k=2.5)
# f_hfo_inh, power_hfo_inh = pass_filter_inh(100, 200, k=2.5)

# print('f_gamma_exc= ', f_gamma_exc, 'P_gamma_exc= ',power_gamma_exc)
# print('f_hfo_exc= ', f_hfo_exc, 'P_hfo_exc= ',power_hfo_exc)
# print('f_gamma_inh= ', f_gamma_inh, 'P_gamma_inh= ',power_gamma_inh)
# print('f_hfo_inh= ', f_hfo_inh, 'P_hfo_inh= ',power_hfo_inh)
#############################################
##########################################################
tmi = 0
tma = 200.0
fig,(ax5, ax6)= plt.subplots(2,1, figsize=(7.0,12.0))
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# fig.suptitle("$\\tau_{d\;inh}=$"+str(Td_inh),fontsize=30, color='k', bbox=props)

ax5.plot(freq[L], PSD_inh[L] ,lw=2.2, color = "red")
# ax5.plot(freq[L], PSD_inh_s[L] ,lw=2.2,alpha=0.6, color = "green")
# ax5.plot(f, psd ,lw=2.2, color = "green")
# ax5.axhline(threshold_exc, linestyle="dashed", linewidth=2, color='c')
# ax5.axhline(0.2*power_gamma_exc, linestyle="dashed", linewidth=2, color='pink')
# ax5.set_title("$excitatory network$",fontsize=30, y=1.1)  #, fontweight='bold'
ax5.set_xlim(freq[L[0]],freq[L[-1]])

ax5.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax5.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
ax5.yaxis.offsetText.set_fontsize(14) 
ax5.set_xlim(tmi,tma+5)
# ax5.set_xticks(np.arange(tmi,tma+1,100))
# ax5.tick_params(axis='both', which='major', labelsize=25)
# ax5.set_xlabel("$frequency(Hz)$",fontsize=30)

# plt.text(150, 575, '$\\tau_{d\;inh}=$'+str(Td_inh), fontsize = 30, 
#          bbox = dict(facecolor = 'green', alpha = 0.4))
ax5.set_ylabel("Power",fontsize=30 )
ax5.set_ylim(0,)
# ax5.grid()


ax6.plot(freq[L], PSD_exc[L], lw=2.2, color = "blue")
# ax6.plot(freq[L], PSD_exc_s[L] ,lw=2.2,alpha=0.6, color = "green")
ax6.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax6.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
ax6.yaxis.offsetText.set_fontsize(14) 
# ax6.axhline(threshold_inh, linestyle="dashed", linewidth=2, color='c')
# ax6.axhline(mean_power_i, linestyle="dashed", linewidth=2, color='k')
# ax6.axhline(0.2*power_g_i, linestyle="dashed", linewidth=2, color='pink')
# ax6.set_title("$inhibitory\:network$", fontsize=30, y=1.1)   # fontweight='bold'
# ax6.tick_params(axis='both', which='major', labelsize=30)
ax6.set_xlabel("Frequency(Hz)",fontsize=30 )
ax6.set_ylabel("Power",fontsize=30 )
ax6.set_xlim(tmi,tma+5)
ax6.set_ylim(0,)
# ax6.set_xticks(np.arange(tmi,tma+1,100))
# ax6.grid()
fig.tight_layout(pad=2.2)
# fig.suptitle('firing count rate', color='b', y=0.99, fontsize=14, fontax.fill_between(x, y, color='skyblue', alpha=0.6, label='Filled Area')weight='bold')
fig.subplots_adjust(wspace=0.25)

for spine in ax5.spines.values():
    spine.set_linewidth(1.0)
for spine in ax6.spines.values():
    spine.set_linewidth(1.0)

# for spine in ['top', 'right']:
#      ax5.spines[spine].set_visible(False)
# for spine in ['top', 'right']:
#      ax6.spines[spine].set_visible(False)

# ax5.xaxis.set_tick_params(labelsize=25, pad=8, width=2.5, length=16)
# ax5.yaxis.set_tick_params(labelsize=25, pad=8, width=2.5, length=16)
    
# # ax5.yaxis.set_major_locator(MultipleLocator(5))
# # ax5.yaxis.set_major_formatter('{x:.0f}')
# # ax5.yaxis.set_minor_locator(MultipleLocator(2.5))

# ax5.xaxis.set_major_locator(MultipleLocator(100))
# ax5.xaxis.set_major_formatter('{x:.0f}')
# ax5.xaxis.set_minor_locator(MultipleLocator(20))

# ax5.tick_params(which='minor', length=8, width=2, color='k')

# ax6.xaxis.set_tick_params(labelsize=25, pad=8, width=2.5, length=16)
# ax6.yaxis.set_tick_params(labelsize=25, pad=8, width=2.5, length=16)

# # ax6.yaxi     s.set_major_locator(MultipleLocator(200))
# # ax6.yaxis.set_major_formatter('{x:.0f}')
# # ax6.yaxis.set_minor_locator(MultipleLocator(100))

# ax6.xaxis.set_major_locator(MultipleLocator(100))
# ax6.xaxis.set_major_formatter('{x:.0f}')
# ax6.xaxis.set_minor_locator(MultipleLocator(20))

# ax6.tick_params(which='minor', length=8, width=2, color='k')

# fig.subplots_adjust(hspace=0.4)

# folder_name = r'VS-codes/assess/new'
# file_name = 'PSD, tau_exc='+str(tau_exc_exc) +', tau_inh='+str(tau_inh_inh) +'.png' 
# if not os.path.exists(folder_name):  # چک کردن وجود پوشه
#     os.makedirs(folder_name)
# file_path = os.path.join(folder_name, file_name)  
# plt.savefig(file_path, dpi=120, bbox_inches='tight') 


############################ WELCH METHOD ##########################################
# fs_w = 1.0 / dt0  # Sampling frequency in Hz

# freq, PSD_e1 = welch(te1, fs=fs_w, window='hann', nperseg=1250, noverlap=620, scaling='density')
# freq2, PSD_inh = welch(ti1, fs=fs_w, window='hann', nperseg=1250, noverlap=937, scaling='density')

# plt.figure(20,figsize=(8,6.25),dpi=120)
# plt.plot(freq, PSD_e1)
# print(len(PSD_e1))
######################################################

plt.show()
