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
from matplotlib.ticker import AutoMinorLocator, MultipleLocator,  ScalarFormatter, LogFormatter
plt.rcdefaults() # reset the plot configurations to default
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
tmax= 10000

N = 300    # number of neuron
N_E = int(0.8*N)
N_I = int(0.2*N)

# N = N_E + N_I

Vrest = -70.0
Vtt = -51.0
V_reset = -59.0
V_thr = -52.0
rate = 4000
tau_exc_exc = 5.0
tau_exc_inh = 5.0
tau_inh_exc = 2.0
tau_inh_inh = 2.0


ndict_pop_exc= {"C_m":200.0,"g_L":10.0,'t_ref':1.0, 'E_ex':0.0, 'E_in':-70.0,
          "tau_rise_ex":0.5 ,'tau_decay_ex': tau_exc_exc ,
          "tau_rise_in":0.5 ,'tau_decay_in': tau_inh_exc, #Td_exc, 
           "V_reset":V_reset,"V_th":V_thr,  
           # "E_L":-70.0, "I_e":0.0,
            "V_m": [Vrest+(Vtt-Vrest)*np.random.rand() for x in range(N_E)] 
            }

ndict_pop_inh = {"C_m":100.0, "g_L":10.0,'t_ref':1.0, 'E_ex':0.0, 'E_in':-70.0,
          "tau_rise_ex":0.5 ,'tau_decay_ex': tau_exc_inh,
          "tau_rise_in":0.5 ,'tau_decay_in': tau_inh_inh,
          "V_reset":V_reset ,"V_th":V_thr,
          # "E_L":-70.0, "I_e":0.0,
           "V_m": [Vrest+(Vtt-Vrest)*np.random.rand() for x in range(N_I)] 
           }

# define two population : ############################################

pop_exc = nest.Create("iaf_cond_beta", N_E, params = ndict_pop_exc)
pop_inh = nest.Create("iaf_cond_beta", N_I, params = ndict_pop_inh)
pop = pop_exc + pop_inh

# records ################################################################################################
start = 200

multi1 = nest.Create("multimeter",{"record_from":["V_m","g_ex","g_in"]})
nest.SetStatus(multi1, {'interval': 0.1,"start":start})
# nest.SetStatus(multi1, {"record_to" : "ascii", "label" : "multi1"})

spikes_recorder1 = nest.Create("spike_recorder", {"start":start}) 
# part_ndict_pop_exc = ndict_pop_exc[0:260]
# part_pop_inh = pop_inh[0:40]
# pp = part_ndict_pop_exc + part_pop_inh
nest.Connect(multi1, pop)
nest.Connect(pop, spikes_recorder1 )

# # generate external input #############################################

rate_ex_s = rate
rate_in_s = rate
noise_e_s = nest.Create("poisson_generator",params={"rate" : rate_ex_s}) #,"stop":400
noise_i_s = nest.Create("poisson_generator",params={"rate": rate_in_s})

nest.Connect(noise_e_s , pop_exc, syn_spec={"weight": 0.2, "delay": 0.1})
nest.Connect(noise_i_s , pop_inh , syn_spec={"weight": 0.25, "delay": 0.1})

# #the synaptic connection pop ##################################################################
P = 0.4
P_ee, P_ie, P_ei, P_ii =P, P, P, P

CEE  = int(P_ee*N_E)   
CIE  = int(P_ie*N_E)    
CEI  = int(P_ei*N_I)      
CII  = int(P_ii*N_I)

gee1 = 4.8/CEE  #0.05   5.8
gie1 = 57.6/CIE    #0.6   57
gei1 = -21.6/CEI  #0.9  21.6
gii1 = -84/CII     #3.5   84


print(CEE,CIE,CEI,CII)
nest.Connect(pop_exc,pop_exc,{'rule': 'fixed_indegree','indegree': CEE, 'allow_autapses': False,
               "allow_multapses":False}, syn_spec = {"weight":gee1,"delay":2.0})

nest.Connect(pop_exc,pop_inh,{'rule': 'fixed_indegree','indegree': CIE, 'allow_autapses': False,
               "allow_multapses":False}, syn_spec = {"weight":gie1,"delay":2.0})

nest.Connect(pop_inh,pop_exc,{'rule': 'fixed_indegree','indegree': CEI, 'allow_autapses': False,
               "allow_multapses":False}, syn_spec = {"weight":gei1,"delay":2.0})

nest.Connect(pop_inh,pop_inh,{'rule': 'fixed_indegree','indegree': CII, 'allow_autapses': False,
               "allow_multapses":False}, syn_spec = {"weight":gii1,"delay":2.0})

#############################################################################
nest.Simulate(tmax)
#############################################################################
# plt.rcParams["figure.figsize"] = (8,5)
# #plt.figure(figsize=(8,4))
# nest.raster_plot.from_device(spikes_recorder1 , hist=True, hist_binwidth=2.0)
# plt.title('Strong network #300')

# plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

A1 = spikes_recorder1.get()
senderspop1 = A1["events"]['senders']
timespop1 = A1["events"]["times"]
idxe1 = np.where(senderspop1<=N_E)
idxi1 = np.where(senderspop1>N_E)

idxe_part = np.where((160<=senderspop1) & (senderspop1<=N_E))
idxi_part = np.where((senderspop1>N_E) & (senderspop1<=260) )

# mm1 = multi1.get()
# s1 = mm1["events"]['senders']
# t1 = mm1["events"]["times"][1::N]

# Vm1 = np.zeros([N,len(t1)])
# g_e1 = np.zeros([N,len(t1)])
# g_i1 = np.zeros([N,len(t1)])

# I_syn_e = np.zeros([N,len(t1)])
# I_syn_i = np.zeros([N,len(t1)])

# for i in range(0,N):
#      Vm1[i,:] = mm1["events"]['V_m'][i::N]
#      g_e1[i,:] = mm1["events"]['g_ex'][i::N]
#      g_i1[i,:] = mm1["events"]['g_in'][i::N]
#      I_syn_e[i,:] = -g_e1[i,:]*(Vm1[i,:]-0)
#      I_syn_i[i,:] = -g_i1[i,:]*(Vm1[i,:]-(-70))



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

##########################################
# Remove DC (mean)
# te1 = te1 - np.mean(te1)
# t11 = ti1 - np.mean(ti1)
Nn = len(te1)
dt0 = B*1e-3
fhate1 = np.fft.fft(te1)
# PSD_e1 = (Nn) * (fhat1 * np.conj(fhat1)).real 
PSD_e1 = (np.abs(fhate1)**2)*(dt0/Nn) 
# PSD_e1 = PSD_e1/(N_E**2)
# PSD_e1 = PSD_e1 / N_E
# PSD_e1 = 10*np.log10(PSD_e1)
freq1 = np.fft.fftfreq(fhate1.size,dt0 )      #(1000/(dt0*n))*np.arange(n)
L1 = np.arange(1,np.floor(Nn/2), dtype='int' )

fhati1 = np.fft.fft(ti1)
PSD_i1 = (np.abs(fhati1)**2)*(dt0/Nn) 
freqs = freq1[L1]

####################################################################################
# ########################################################################################################
fig = plt.figure(figsize=(13, 6.5))
gs = gridspec.GridSpec(6, 4, figure=fig,
                        height_ratios=[2.2,0.2,1, 2.2,1.2, 2.5],
                        width_ratios=[30, 5, 0.8, 12 ],     # Column 0 is wider
                        wspace=0.5, 
                         hspace=0.5,
                        # height_ratios=[1, 1, 1, 1]        # Row 1 is taller
                       )
# right = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1,0:], wspace=0.3)
# ax2 = fig.add_subplot(right[0])
# ax3 = fig.add_subplot(right[1])

# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# fig.suptitle(
#           #    "$\\tau_{\mathrm{exc} \\to \mathrm{exc}}=$"+str(tau_exc_exc)+'ms,'+
#           #    "$\quad\\tau_{\mathrm{exc} \\to \mathrm{inh}}=$"+str(tau_exc_inh)+'ms'+
#              "$\\tau_{\mathrm{inh} \\to \mathrm{exc}}=$"+str(tau_inh_exc)+'ms'+
#              "$\quad\\tau_{\mathrm{inh} \\to \mathrm{inh}}=$"+str(tau_inh_inh)+'ms',
#              fontsize=20, color='k', bbox=props, 
#               x=0.43, y=0.98)

# Big plot spanning two columns
ax1 = fig.add_subplot(gs[0:4, 0:2])

ax2 = fig.add_subplot(gs[0:2, 3])
ax3 = fig.add_subplot(gs[3:5, 3])
ax4 = fig.add_subplot(gs[5, 0:2])
# fig, axes1 = plt.subplots(2,1, figsize=(10,10),dpi=100.0)
began = 1000
end = 1100

# ax1.plot(timespop1[idxe_part],senderspop1[idxe_part],'|',color='b',markersize=4.0, markeredgewidth=1.5 )
ax1.plot(timespop1[idxe1],senderspop1[idxe1],'|',color='b',markersize=4.0, markeredgewidth=1.5 )
# ax1.plot(timespop1[idxi_part],senderspop1[idxi_part],'|',color='r',markersize=4.0,  markeredgewidth=1.5)
ax1.plot(timespop1[idxi1],senderspop1[idxi1],'|',color='r',markersize=4.0,  markeredgewidth=1.5)
# ax1.set_xlabel('t(ms)', fontsize=15, labelpad=16 )
ax1.set_ylabel('Neuron', size=50, labelpad=12 )
# ax1.xaxis.set_tick_params(labelsize=15, pad = 8)
# ax1.yaxis.set_tick_params(labelsize=15, pad = 8)
ax1.set_xlim(began, end)
# ax1.set_xticks(np.arange(began,end+5,50))
ax1.tick_params(labelsize=35, axis="x", pad = 6.0 )  #, pad = 9.0
ax1.tick_params(labelsize=35, axis="y", pad = 6.0 )  #, pad = 9.0
# ax1.yaxis.set_major_locator(MultipleLocator(100))
# ax1.yaxis.set_major_formatter('{x:.0f}')
# ax1.yaxis.set_minor_locator(MultipleLocator(20))

ax1.xaxis.set_major_locator(MultipleLocator(50))
ax1.xaxis.set_major_formatter('{x:.0f}')
ax1.xaxis.set_minor_locator(MultipleLocator(10))

ax1.yaxis.set_major_locator(MultipleLocator(100))
ax1.yaxis.set_major_formatter('{x:.0f}')
ax1.yaxis.set_minor_locator(MultipleLocator(50))

# ax1.set_ylim(158,262)

# ax1.xaxis.set_tick_params(labelsize=15, pad=10, width=2, length=6)
# ax1.yaxis.set_tick_params(labelsize=15, pad=10, width=2, length=6)

ax1.tick_params(which='major', length=14, width=1.4, color='k')
ax1.tick_params(which='minor', length=8, width=1.0, color='k')

ax1.set_xlim(began-1,end+1)
ax1.set_xticklabels([])
for spine in ['top', 'right']:
     ax1.spines[spine].set_visible(False)

# ax4.plot(bins[:-1], ti1 , alpha = 0.8, lw=2.5, color="red" )
ax4.fill_between(bins[:-1], ti1, color="red")
# ax4.plot(bins[:-1], te1 , alpha = 0.8, lw=2.5, color="blue" )
ax4.fill_between(bins[:-1], te1, color="blue")

ax4.set_xlabel('t(ms)', fontsize=50, labelpad=12 )
ax4.set_ylabel('Rate(Hz)', size=50, labelpad=12 )
# ax1.set_xticks(np.arange(began,end+5,50))
# ax1.set_yticks(np.arange(0,300+5,100))
ax4.tick_params(labelsize=35, axis="x", width=2, length=10, pad = 6.0 )  #, pad = 9.0
ax4.tick_params(labelsize=35, axis="y", width=2, length=10, pad = 6.0 )  #, pad = 9.0
# ax1.yaxis.set_major_locator(MultipleLocator(100))
# ax1.yaxis.set_major_formatter('{x:.0f}')
# ax1.yaxis.set_minor_locator(MultipleLocator(20))

ax4.xaxis.set_major_locator(MultipleLocator(50))
ax4.xaxis.set_major_formatter('{x:.0f}')
ax4.xaxis.set_minor_locator(MultipleLocator(10))

# ax1.yaxis.set_major_locator(MultipleLocator(40))
# ax1.yaxis.set_major_formatter('{x:.0f}')
# ax1.yaxis.set_minor_locator(MultipleLocator(20))
# ax4.set_ylim(0,262)
# ax4.xaxis.set_tick_params(labelsize=15, pad=10, width=2, length=6)
# ax4.yaxis.set_tick_params(labelsize=15, pad=10, width=2, length=6)

ax4.tick_params(which='major', length=12, width=1.4, color='k')
ax4.tick_params(which='minor', length=5, width=1.0, color='k')
ax4.set_xlim(began-1,end+1)

for spine in ['top', 'right']:
     ax4.spines[spine].set_visible(False)
####################################################################
tmi = 0
tma = 200
# axes3 = ax1.twinx()
ax2.plot(freq1[L1], 2*PSD_i1[L1] ,lw=2.2, color = "r")
ax2.set_xlim(freq1[L1[0]],freq1[L1[-1]])


ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax2.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
ax2.yaxis.offsetText.set_fontsize(30) 

ax2.set_xlim(tmi,tma+5)
# ax2.set_xlabel("frequency(Hz)",fontsize=25, labelpad=14)
ax2.set_ylabel("PSD",fontsize=50, labelpad=6)
ax2.set_ylim(0,)

for spine in ['top', 'right']:
     ax2.spines[spine].set_visible(False)

ax2.xaxis.set_tick_params(labelsize=35)
ax2.yaxis.set_tick_params(labelsize=35)
# axe.yaxis.set_tick_params(labelsize=15, pad=8, width=2.5, length=6)
# ax2.yaxis.set_major_locator(MultipleLocator(100))
# ax2.yaxis.set_major_formatter('{x:.0f}')
# ax2.yaxis.set_minor_locator(MultipleLocator(50))

ax2.xaxis.set_major_locator(MultipleLocator(100))
ax2.xaxis.set_major_formatter('{x:.0f}')
ax2.xaxis.set_minor_locator(MultipleLocator(25))
ax2.tick_params(which='major', length=14, width=1.4, color='k')
ax2.tick_params(which='minor', length=8, width=1.0, color='k')
ax2.set_xticklabels([])
#########################################################
# ax3.plot(freq1[L1], PSD_e1[L1] ,lw=2.2, color = "b")
ax3.plot(freq1[L1], 2*PSD_e1[L1] ,lw=2.2, color = "b")
ax3.set_xlim(freq1[L1[0]],freq1[L1[-1]])

  
ax3.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax3.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
ax3.yaxis.offsetText.set_fontsize(30) 

ax3.set_xlim(tmi,tma+5)
ax3.set_xlabel("F(Hz)",fontsize=50, labelpad=6)
ax3.set_ylabel("PSD",fontsize=50, labelpad=6 )
ax3.set_ylim(0,)

for spine in ['top', 'right']:
     ax3.spines[spine].set_visible(False)

ax3.xaxis.set_tick_params(labelsize=30, pad=6)
ax3.yaxis.set_tick_params(labelsize=30, pad=6)
# axe.yaxis.set_tick_params(labelsize=15, pad=8, width=2.5, length=6)
# ax3.yaxis.set_major_locator(MultipleLocator(10))
# ax3.yaxis.set_major_formatter('{x:.0f}')
# ax3.yaxis.set_minor_locator(MultipleLocator(5))

ax3.xaxis.set_major_locator(MultipleLocator(100))
ax3.xaxis.set_major_formatter('{x:.0f}')
ax3.xaxis.set_minor_locator(MultipleLocator(25))
ax3.tick_params(which='major', length=14, width=1.4, color='k')
ax3.tick_params(which='minor', length=8, width=1.0, color='k')
# ax3.set_xticklabels([])
# ax2.grid()
# axes1[2].grid()
# ax2.legend(loc='upper right')
for spine in ax1.spines.values():
    spine.set_linewidth(1.0)
for spine in ax2.spines.values():
    spine.set_linewidth(1.0)
for spine in ax3.spines.values():
    spine.set_linewidth(1.0)
for spine in ax4.spines.values():
    spine.set_linewidth(1.)
# fig.tight_layout(pad=3.0)

folder_name = 'spectrogram/fin1'
# file_name = 'raster,tau_exc='+str(Td_exc)+',tau_inh='+str(Td_inh)+'n2.png' 
file_name = 'raster,tau_exc='+str(tau_exc_inh)+',tau_inh='+str(tau_inh_inh)+'ph.png'
file_path = os.path.join(folder_name, file_name)  
plt.savefig(file_path, dpi=350, bbox_inches='tight')



plt.show()
