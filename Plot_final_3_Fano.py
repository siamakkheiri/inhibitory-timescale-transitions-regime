import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

plt.rcdefaults() # reset the plot configurations to default
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans"],
})


tau_d_inh = np.arange(1.0, 10.1, 0.1)
# tau_d_inh_exc = np.arange(1.0, 8.1, 0.1)
# tau_d_inh_inh = np.loadtxt('VS-codes/spectrogram/tau_in_in.txt')
# freq1 = np.loadtxt('VS-codes/spectrogram/freq1.txt')
# frequency = np.loadtxt('VS-codes/P_connection/freqn.txt')
frequency = np.loadtxt('spectrogram/freqr.txt')
print(len(frequency))
# PSD_exc_w = np.loadtxt('VS-codes/spectrogram/n1PSD_exc-tau_d_exc5.txt')
# PSD_inh_w = np.loadtxt('VS-codes/spectrogram/n1PSD_inh-tau_d_exc5.txt')

PSD_exc = np.loadtxt('spectrogram/n1000PSD_exc1.txt')
PSD_inh = np.loadtxt('spectrogram/n1000PSD_inh1.txt')
print(np.shape(PSD_exc))
coherency_exc = np.loadtxt('spectrogram/n1000corr_exc1.txt')
coherency_inh = np.loadtxt('spectrogram/n1000corr_inh1.txt')

# coherency_exc = np.loadtxt('VS-codes/Fano/coherency_excPop1.txt')
# coherency_inh = np.loadtxt('VS-codes/Fano/coherency_inhPop1.txt')
# fano_exc = np.loadtxt('VS-codes/Fano/Fano_exc1.txt')
# fano_inh = np.loadtxt('VS-codes/Fano/Fano_inh1.txt')

# fano_exc_test = np.loadtxt('VS-codes/Fano/Fano_exc_test.txt')
# fano_inh_test = np.loadtxt('VS-codes/Fano/Fano_inh_test.txt')
###############ING#################################
# idx = np.where(np.isclose(tau_d_inh, 2.0))[0]
# # print(idx)
# print(tau_d_inh[idx])
# coh_exc_tau_i_e = coherency_exc[:, idx]
# coh_inh_tau_i_e = coherency_inh[:, idx]
# # print(coh_exc_tau_i_e)
# fano_exc_tau_i_e = fano_exc[:, idx]
# fano_inh_tau_i_e = fano_inh[:, idx]


# coh_exc_tau_i_i = coh_exc_tau_i_i.T
# coh_inh_tau_i_i = coh_inh_tau_i_i.T
#######################################################
######################################################
fig = plt.figure(figsize=(6.0,6.0),dpi=120)
gs =GridSpec(2,2,
             height_ratios=[4, 1.6],
             width_ratios=[1,0.05],
             hspace=0.05,
             #wspace=0.25
             )
ax_exc_spect = fig.add_subplot(gs[0,0])
pc_exc = fig.add_subplot(gs[1,0])

# plt.rcParams['font.size'] = 16
# plt.rcParams.update({'font.family':'serif'})
# plt.rcParams.update({'font.style':'italic'})
# Convert to dB scale
PSD_exc_smooth = gaussian_filter(PSD_exc, sigma =1.0)
# Optional:  Clip to 5th - 99th percentilefor better dynamic range
PSD_exc_dB = 10*np.log10(PSD_exc_smooth+1e-10)

# vmin = np.percentile(PSD_inh_dB, 60)
# vmax = np.percentile(PSD_inh_dB, 99.8)

# X,Y = np.meshgrid(frequency, tau_d_inh_exc )
X,Y = np.meshgrid(frequency, tau_d_inh )
im1 = ax_exc_spect.pcolormesh(Y, X, PSD_exc_dB,shading='nearest', cmap='jet')  #, vmax=52, , 'viridis' "magma"
ax_exc_spect.axvline(2.0 ,linestyle="dashed" ,color="k", lw=1.5)
ax_exc_spect.axvline(3.5 ,linestyle="dashed" ,color="k", lw=1.5)
ax_exc_spect.axvline(8.0 ,linestyle="dashed" ,color="k", lw=1.5)
ax_exc_spect.set_ylim(0,200)
# ax_exc_spect.set_xticks([1,4,7,10])
# ax_exc_spect.set_xticks(np.arange(1,10.1))
# ax_exc_spect.set_xlim(1,10)
ax_exc_spect.tick_params(labelsize=25, axis="x", pad = 6.0 )  #, pad = 9.0
ax_exc_spect.tick_params(labelsize=25, axis="y", pad = 6.0 )  #, pad = 9.0
# ax_exc_spect.xaxis.set_major_locator(MultipleLocator(1))
# ax_exc_spect.xaxis.set_major_formatter('{x:.0f}')
# ax_exc_spect.xaxis.set_minor_locator(MultipleLocator(1.0))

ax_exc_spect.yaxis.set_major_locator(MultipleLocator(50))
ax_exc_spect.yaxis.set_major_formatter('{x:.0f}')
ax_exc_spect.yaxis.set_minor_locator(MultipleLocator(10))

ax_exc_spect.tick_params(which='major', length=14, width=1.5, color='k')
ax_exc_spect.tick_params(which='minor', length=10, width=1.0, color='k')
ax_exc_spect.set_xticklabels([])
ax_exc_spect.set_xticks([])

cbmean_PSD_exc = fig.add_subplot(gs[0, 1])
cbar1 = fig.colorbar(im1, cax=cbmean_PSD_exc, )
# cbar1 = plt.colorbar(label='PSD (dB)')spectrogram
# cbar1.set_label('PSD (dB)', fontsize=30, labelpad=10)
cbar1.ax.tick_params(labelsize=25)
# cbar1.set_label('PSD (dB)', fontsize=30, labelpad=15)
# ax_exc_spect.set_xlabel('$\\tau_{\mathrm{inh}}$'+ ' (ms)',size=35)
ax_exc_spect.set_ylabel('Frequency (Hz)',size=35)
# ax_exc_spect.set_title('Excitatory Network', fontsize=30, pad=15)

###################################


###############################

# coh_exc_tau_i_e_filt = gaussian_filter1d(coh_exc_tau_i_e,sigma=1.0)
# pc_exc.plot(tau_d_inh, ee, lw=2.0,color='k')
# pc_exc.plot(tau_d_inh, coh_exc_tau_i_e, lw=2.0,color='green')
pc_exc.plot(tau_d_inh, coherency_exc, lw=2.0,color='r')
# pc_exc.plot(tau_d_inh, fano_exc_test, lw=2.0,color='k')
# pc_exc.plot(tau_d_inh, coh_exc_tau_i_e, lw=2.0,color='k')

pc_exc.set_xlabel('$\\tau_{\mathrm{inh \\to inh}}$'+ ' (ms)',size=35)
pc_exc.set_ylabel('pc',size=35, labelpad=10)
# pc_exc.xaxis.set_major_locator(MultipleLocator(1))
# pc_exc.xaxis.set_major_formatter('{x:.0f}')
pc_exc.xaxis.set_minor_locator(MultipleLocator(1.0))

pc_exc.yaxis.set_major_locator(MultipleLocator(40))
pc_exc.yaxis.set_minor_locator(MultipleLocator(20))
# pc_exc.set_yticks([0, 0.04,0.08])
# pc_exc.set_ylim(0,100)
# pc_exc.yaxis.tick_right()
pc_exc.tick_params(which='major', length=14, width=1.4, color='k')
pc_exc.tick_params(which='minor', length=10, width=1.0, color='k')
pc_exc.tick_params(labelsize=25, axis="x",  pad = 6.0 )  #, pad = 9.0
pc_exc.tick_params(labelsize=25, axis="y",  pad = 6.0 ) 
pc_exc.set_xticks([1,4,7,10]) 
# pc_exc.set_xticks(np.arange(1,10.1))
pc_exc.set_xlim(1.0,10.0)
# mean_fr_exc.set_title("Inhibitory Network",size=30, pad=20)
for spine in pc_exc.spines.values():
    spine.set_linewidth(1.0)
# for spine in ['top', 'right']:
#      pc_exc.spines[spine].set_visible(False)

folder_name = 'spectrogram/fin1'

file_name = 'test1_exc.png' 

file_path = os.path.join(folder_name, file_name) 
plt.savefig(file_path, dpi=150, bbox_inches='tight') 

###########################################################################

# ################################################ inhibitory ################################
fig = plt.figure(figsize=(6.0,6.0),dpi=120)
gs =GridSpec(2,2,
             height_ratios=[4, 1.6],
             width_ratios=[1,0.05],
             hspace=0.05,
             #wspace=0.25
             )
ax_inh_spect = fig.add_subplot(gs[0,0])
pc_inh = fig.add_subplot(gs[1,0])
# plt.rcParams['font.size'] = 16
# plt.rcParams.update({'font.family':'serif'})
# plt.rcParams.update({'font.style':'italic'})
# Convert to dB scale
PSD_inh_smooth = gaussian_filter(PSD_inh, sigma =1.0)
# Optional:  Clip to 5th - 99th percentilefor better dynamic range
PSD_inh_dB = 10*np.log10(PSD_inh_smooth+1e-10)

# vmin = np.percentile(PSD_inh_dB, 60)
# vmax = np.percentile(PSD_inh_dB, 99.8)

# X,Y = np.meshgrid(frequency, tau_d_inh_exc )
X,Y = np.meshgrid(frequency, tau_d_inh )
im2 = ax_inh_spect.pcolormesh(Y, X, PSD_inh_dB,  cmap='jet')  #vmax=65,, norm=LogNorm(), 'viridis' "magma"
ax_inh_spect.axvline(2.0 ,linestyle="dashed" ,color="k", lw=1.5)
ax_inh_spect.axvline(3.5 ,linestyle="dashed" ,color="k", lw=1.5)
ax_inh_spect.axvline(8.0 ,linestyle="dashed" ,color="k", lw=1.5)
ax_inh_spect.set_ylim(0,200)
# ax_inh_spect.set_xticks([1,4,7,10])
# ax_inh_spect.set_xticks(np.arange(1,10.1))
# ax_inh_spect.set_xlim(1,10)
ax_inh_spect.tick_params(labelsize=25, axis="x", pad = 6.0 )  #, pad = 9.0
ax_inh_spect.tick_params(labelsize=25, axis="y", pad = 6.0 )  #, pad = 9.0
# ax_inh_spect.xaxis.set_major_locator(MultipleLocator(1))

# ax_inh_spect.xaxis.set_minor_locator(MultipleLocator(1.0))

ax_inh_spect.yaxis.set_major_locator(MultipleLocator(50))
ax_inh_spect.yaxis.set_major_formatter('{x:.0f}')
ax_inh_spect.yaxis.set_minor_locator(MultipleLocator(10))

ax_inh_spect.tick_params(which='major', length=14, width=1.4, color='k')
ax_inh_spect.tick_params(which='minor', length=10, width=1.0, color='k')
# cbar1 = plt.colorbar(label='PSD (dB)')
# cbar1.set_label('PSD (dB)', fontsize=25, labelpad=25) 
# cbar1.ax.tick_params(labelsize=25)

cbmean_fr_inh = fig.add_subplot(gs[0, 1])
cbar2 = fig.colorbar(im2, cax=cbmean_fr_inh)
cbar2.set_label('PSD (dB)', fontsize=30, labelpad=10)
cbar2.ax.tick_params(labelsize=25)
# ax_inh_spect.set_xlabel('$\\tau_{\mathrm{inh}}$'+ ' (ms)',size=30)
# ax_inh_spect.set_ylabel('Frequency (Hz)',size=30)
# ax_inh_spect.set_title('Inhibitory Network', fontsize=30, pad=15)
ax_inh_spect.set_xticklabels([])
ax_inh_spect.set_xticks([])
###################################
###############################

# coh_inh_tau_i_e_filt = gaussian_filter1d(coh_inh_tau_i_e,sigma=1.0)
# pc_exc.plot(tau_d_inh, ee, lw=2.0,color='k')
# pc_inh.plot(tau_d_inh, coh_inh_tau_i_e, lw=2.0,color='green')
pc_inh.plot(tau_d_inh, coherency_inh, lw=2.0,color='green')
# pc_inh.plot(tau_d_inh, fano_inh_test, lw=2.0,color='k')
pc_inh.set_xlabel('$\\tau_{\mathrm{inh \\to inh}}$'+ ' (ms)',size=35)
pc_inh.set_ylabel('pc',size=30, labelpad=10)
pc_inh.set_xticks([1,4,7,10])
# pc_inh.set_xticks(np.arange(1,10))
# pc_inh.xaxis.set_major_locator(MultipleLocator(1))
pc_inh.xaxis.set_major_formatter('{x:.0f}')
pc_inh.xaxis.set_minor_locator(MultipleLocator(1.0))
pc_inh.yaxis.set_major_locator(MultipleLocator(150))
# pc_exc.yaxis.set_major_formatter('{x:.0f}')
pc_inh.yaxis.set_minor_locator(MultipleLocator(75))
# pc_inh.set_yticks([0, 0.4,0.8])
# pc_inh.set_ylim(0,390)                          
# pc_inh.yaxis.tick_right()
# pc_inh.yaxis.set_label_position("right")
pc_inh.tick_params(which='major', length=14, width=1.4, color='k')
pc_inh.tick_params(which='minor', length=10, width=1.0, color='k')
pc_inh.tick_params(labelsize=25, axis="x",  pad = 6.0 )  #, pad = 9.0
pc_inh.tick_params(labelsize=25, axis="y",  pad = 6.0 )  
pc_inh.set_xlim(1.0,10.1)
# mean_fr_exc.set_title("Inhibitory Network",size=30, pad=20)
for spine in pc_inh.spines.values():
    spine.set_linewidth(1.0)
# for spine in ['top', 'right']:
#      pc_exc.spines[spine].set_visible(False)

folder_name = 'spectrogram/fin1'

file_name = 'test1_inh.png' 


file_path = os.path.join(folder_name, file_name) 
plt.savefig(file_path, dpi=150, bbox_inches='tight') 

plt.show()