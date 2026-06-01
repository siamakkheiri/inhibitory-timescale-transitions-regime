import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import os
import numpy.ma as ma
plt.rcdefaults() # reset the plot configurations to default
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Serif", "Times New Roman", "DejaVu Serif"],
#     "mathtext.fontset": "cm",  # 'cm' stands for Computer Modern (LaTeX default)
#     # "axes.titleweight": "bold"
# })

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans"],
    # "mathtext.fontset": "cm",  # This is the "LaTeX font" magic line
    # "mathtext.fontset": "stixsans", # Math will now look like Sans-Serif
    "mathtext.fontset": "custom",
    # "mathtext.rm": "Liberation Sans",
    # "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "DejaVu Serif:bold",
})


frequency = np.loadtxt('VS-codes/spectrogram/tau/freqn.txt')
freq_exc = np.loadtxt('VS-codes/spectrogram/tau/freq_excNet1.txt')
freq_inh= np.loadtxt('VS-codes/spectrogram/tau/freq_inhNet1.txt')

PSD_exc = np.loadtxt('VS-codes/spectrogram/tau/PSD_excNet1.txt')
PSD_inh= np.loadtxt('VS-codes/spectrogram/tau/PSD_inhNet1.txt')



# tau_d_exc = np.arange(3.0, 10.1, 0.5)
# tau_d_inh = np.arange(1.0, 10.1, 0.5)
# tau_d_inh_exc = np.arange(4.0, 10.1, 2)
# tau_d_inh_inh = np.arange(1.0, 10.1, 2)
tau_d_inh_exc = np.arange(1.0, 10.1, 0.2)
tau_d_inh_inh = np.arange(1.0, 10.1, 0.2)

# print(np.shape(freq_HFO_inh))
########################################### EXCITATORY ###############################################

mask_freq_exc = ma.masked_where(freq_exc==0, freq_exc)
mask_PSD_exc = ma.masked_where(freq_exc==0, PSD_exc)
PSD_exc_dB = 10 * np.ma.log10(mask_PSD_exc)


################################################ INHIBITORY ###################################################
mask_freq_inh = ma.masked_where(freq_inh==0, freq_inh)
mask_PSD_inh = ma.masked_where(freq_inh==0, PSD_inh)
PSD_inh_dB = 10 * np.ma.log10(mask_PSD_inh)


###################################################### PLOT ##################################################################
def add_bold_label(ax, label):
    # \mathbf{} is the LaTeX command for Bold Roman (Serif)
    # We use double curly braces {{ }} because it's an f-string
    ax.text(-0.1, 1.25, rf"$\mathbf{{{label.upper()}}}$", 
            transform=ax.transAxes, 
            fontsize=28, 
            va='top', 
            ha='right')

fig, axes = plt.subplots(2, 2, figsize=(12, 10.))
add_bold_label(axes[0,0], label="A")
add_bold_label(axes[0,1], label="B")
add_bold_label(axes[1,0], label="C")
add_bold_label(axes[1,1], label="D")
# plt.rcParams.update({
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Liberation Sans"],
# })
X,Y = np.meshgrid( tau_d_inh_inh, tau_d_inh_exc )
# print("X.shape:", X.shape)  # (19, 15)
# print("Y.shape:", Y.shape)  # (19, 15)
# print("Y.shape:", freq_gamma_exc.shape)  # (19, 15)
boundaries = np.arange(0,171,1)
cmap = plt.jet
cmap = plt.cm.jet.copy()
cmap.set_under('white') 
norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
im1=axes[0,0].pcolormesh( X,Y, mask_freq_exc,  cmap=cmap, vmin=0, vmax=171, shading='auto')  #, norm=LogNorm(), 'viridis' "magma"

cbar=plt.colorbar(im1, ax=axes[0,0]) 
cbar.set_ticks(np.arange(0, 200, 50))
cbar.ax.yaxis.set_minor_locator(MultipleLocator(10))  # minor ticks every 0.5
cbar.ax.tick_params(labelsize=25,  width=1.5, length=10, direction='out', pad=3)
cbar.ax.tick_params(which='minor', width=1.0, length=6,  color='k')
cbar.ax.yaxis.set_major_formatter('{x:.0f}')
# cbar.ax.xaxis.set_label_position('top')
# cbar.ax.yaxis.set_label_position('left')
# cbar.ax.set_xlabel('Hz', rotation=0, labelpad=0)  # rotation=0 for horizontal label 
cbar.ax.set_title('F(Hz)', fontsize=25, pad=20) 
# axes[0,0].set_xlabel('$\\tau_{exc}$'+ ' (ms)',size=20)
# axes[0,0].set_xticklabels([])
axes[0,0].set_ylabel('$\\tau_{inh \\to exc}$'+ ' (ms)',size=30,labelpad=20)
# axes[0,0].set_title('Gamma Band', fontsize=20, pad=12)
###############################################################

cmap = plt.cm.jet.copy()
cmap.set_under('white') 

im2=axes[0,1].pcolormesh( X, Y, PSD_exc_dB,  cmap=cmap , shading='auto')  #, norm=LogNorm(), 'viridis' "magma"

cbar=plt.colorbar(im2, ax=axes[0,1]) 
cbar.update_ticks()
cbar.ax.set_title('PSD(dB)', fontsize=25, pad=20) 
cbar.ax.tick_params(labelsize=25,  width=1.5, length=10, direction='out', pad=3)
cbar.ax.yaxis.set_minor_locator(MultipleLocator(20))
cbar.ax.yaxis.set_major_formatter('{x:.0f}')
# axes[0,1].set_xlabel('$\\tau_{exc}$'+ ' (ms)',size=20)
# axes[0,1].set_ylabel('$\\tau_{inh}$'+ ' (ms)',size=20)
# axes[0,1].set_xticklabels([])
# axes[0,1].set_yticklabels([])
# axes[0,1].set_title('Gamma Band', fontsize=20, pad=12)
###########################################################################################################
# mask_freq_HFO_exc = np.ma.masked_where((freq_HFO_exc<90) | (freq_HFO_exc>200),freq_HFO_exc )
# mask_PSD_HFO_exc = ma.masked_where(PSD_HFO_exc<1e-6, PSD_HFO_exc)
# mask_PSD_HFO_exc = np.where(mask_freq_HFO_exc,mask_PSD_HFO_exc,0)
# PSD_HFO_exc_dB = 10 * np.ma.log10(PSD_HFO_exc+1e-6)
# mask_PSD_HFO_exc_dB = np.ma.masked_array(PSD_HFO_exc_dB, mask=mask_freq_HFO_exc.mask)

# ساختن نرمالایزر
boundaries = np.arange(0,171,1)
cmap = plt.cm.jet
cmap = plt.cm.jet.copy()

cmap.set_under('white') 

im3=axes[1,0].pcolormesh( X,Y, freq_inh,  cmap=cmap,  vmin=0, vmax=171, shading='auto')  #, norm=LogNorm(), 'viridis' "magma"
# plt.axvline(1.5 ,linestyle="dashed" ,color="white", lw=2.0)
# plt.axvline(5.0 ,linestyle="dashed" ,color="white", lw=2.0)
# plt.axvline(9.0 ,linestyle="dashed" ,color="white", lw=2.0)
# plt.ylim(0,200)
# axes[1,0].set_xticks(np.arange(3,10.1,1))
# axes[1,0].set_yticks(np.arange(1,10.1,1))
# plt.xlim(1,8)

cbar=plt.colorbar(im3, ax=axes[1,0]) 
cbar.set_ticks(np.arange(0, 200,50))
cbar.ax.yaxis.set_minor_locator(MultipleLocator(10))  # minor ticks every 0.5
cbar.ax.tick_params(labelsize=25,  width=1.5, length=10, direction='out', pad=3)
cbar.ax.tick_params(which='minor', width=1.0, length=6,  color='k')
cbar.ax.yaxis.set_major_formatter('{x:.0f}')
cbar.ax.set_title('F(Hz)', fontsize=25, pad=20) 
axes[1,0].set_xlabel('$\\tau_{inh \\to inh}$'+ ' (ms)',size=30, labelpad=10)
axes[1,0].set_ylabel('$\\tau_{inh \\to exc}$'+ ' (ms)',size=30, labelpad=20)
# axes[1,0].set_title('HFO Band', fontsize=20, pad=12)
###############################################################3

cmap = plt.cm.jet.copy()
cmap.set_under('white') 

# norm=colors.LogNorm(vmin=1e-6,vmax=PSD_HFO_exc.max())
im4=axes[1,1].pcolormesh(X,Y,  PSD_inh_dB,  cmap=cmap,shading='auto' )  #, norm=LogNorm(), 'viridis' "magma"
# plt.axvline(1.5 ,linestyle="dashed" ,color="white", lw=2.0)
# plt.axvline(5.0 ,linestyle="dashed" ,color="white", lw=2.0)
# plt.axvline(9.0 ,linestyle="dashed" ,color="white", lw=2.0)
# plt.ylim(0,200)
# axes[1,1].set_xticks(np.arange(3,10.1,1))
# axes[1,1].set_yticks(np.arange(1,10.1,1))

# axes[1,1].xaxis.set_major_locator(MultipleLocator(1))
# axes[1,1].xaxis.set_major_formatter('{x:.0f}')
# axes[1,1].xaxis.set_minor_locator(MultipleLocator(0.5))
# axes[1,1].yaxis.set_major_locator(MultipleLocator(1))
# axes[1,1].yaxis.set_major_formatter('{x:.0f}')
# axes[1,1].yaxis.set_minor_locator(MultipleLocator(0.5))

cbar=plt.colorbar(im4, ax=axes[1,1]) 
cbar.ax.tick_params(labelsize=25, width=1.5, length=10, direction='out', pad=3)
# cbar.locator = MultipleLocator(10)
# cbar.set_ticks([10,25,40])
cbar.update_ticks()
cbar.ax.set_title('PSD(dB)', fontsize=25, pad=20) 
axes[1,1].set_xlabel('$\\tau_{inh \\to inh}$'+ ' (ms)',size=30, labelpad=10)
# axes[0,1].set_ylabel('$\\tau_{inh}$'+ ' (ms)',size=20)
# axes[1,1].set_yticklabels([])
# axes[1,1].set_title('HFO Band Power', fontsize=20)
# plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.6, wspace=0.3)


for a in range(2):
    for b in range(2):
        axes[a,b].tick_params(labelsize=25, axis="x", width=1.5, length=12, pad = 8.0 ) 
        axes[a,b].tick_params(labelsize=25, axis="y", width=1.5, length=12, pad = 8.0 )
        axes[a,b].set_xlim(1,10)
        axes[a,b].set_xticks(np.arange(1,11,2))
        axes[a,b].set_xticks(np.arange(2,12,2),minor=True)

        axes[a,b].set_ylim(1,10)
        axes[a,b].set_yticks(np.arange(1,11,2))
        axes[a,b].set_yticks(np.arange(2,12,2),minor=True)
        axes[a,b].tick_params(which='major', length=10, width=1.5, color='k')
        axes[a,b].tick_params(which='minor', length=5, width=1.0, color='k')


folder_name = 'VS-codes/spectrogram/tau'
file_name = 'peak3D_new.png' 
file_path = os.path.join(folder_name, file_name) 
plt.savefig(file_path, dpi=150, bbox_inches='tight') 



plt.show()