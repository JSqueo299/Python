#Script to change legend spacing and location, text style and size, figure size, and margin size so you do not need to stretch the figure at all and distort the image.
# ======================================================== #
from __future__ import print_function, division
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle as pkl
import cantera as ct

mpl.rc('lines', linewidth=1.0, markersize=1.0)
########### SET THE FONT HERE -- FOR AIAA, SIZE=10 USUALLY ###########
mpl.rc('font', size=10, family='Times New Roman')
mpl.rc('legend', borderpad=0.25,
              handletextpad=0.05,
              borderaxespad=0.25,
              labelspacing=0.1,
              handlelength=1.5,
              columnspacing=0.25,
              numpoints=1,
              scatterpoints=1,
              fontsize=7,
              frameon=False)

def hide_axes(fig, even=False):
    plt.tight_layout()
    if even:
        for idx, ax in enumerate(fig.get_axes()):
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            if idx % 2 == 0:
                ax.xaxis.set_ticks_position('bottom')
            else:
                ax.xaxis.set_ticks_position('top')
    else:
        for ax in fig.get_axes():
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

im_width = 3.0 # 5.0
im_height = 2.0 # 3.0

fig = plt.figure()
fig.set_size_inches(im_width, im_height)

##### LOAD YOUR DATA HERE ###########
# Generating phi values       
phi_5 = [0.5, 0.7368, 0.977, 1.210, 1.447, 1.684, 1.921, 2.157, 2.394, 2.631, 2.868, 3.578, 5.0]
flamespeed_mean_5 = [0.550, 1.476, 2.189, 2.637, 2.870, 2.938, 2.893, 2.782, 2.638, 2.478, 2.309, 1.821, 1.015]
flamespeed_std_5 = [0.2709, 0.4998, 0.5477, 0.538, 0.500, 0.4473, 0.389, 0.3335, 0.286, 0.247, 0.210, 0.130, 0.0561]


# Calculating relative uncertainty
a = np.array(flamespeed_mean_5, dtype=np.float)
b = np.array(flamespeed_std_5, dtype=np.float)
relative_uncert = b/a
relative_uncert = list(relative_uncert)


# Plotting  SL                                                                                                                                           
#plt.errorbar(phi_5, flamespeed_mean_5, flamespeed_std_5, linestyle='None', markersize=1.5, marker='o', capsize=5, elinewidth=0.5, markeredgewidth=0.5)
plt.xlabel('$\phi$')
#plt.ylabel('$s_L$, m/s')

# Plotting Relative uncertainty
plt.scatter(phi_5, relative_uncert)
plt.ylabel(r'$\sigma$/$\bar s_L$')
                                                                                   

# This next line removes the top and right spines from the figure
hide_axes(fig)

##### CHANGE THESE PADDING VALUES SO EVERYTHING IS VISIBLE BUT THERE IS MINIMAL EXTRA SPACE #####
plt.subplots_adjust(left=0.18, right=0.95, top=0.96, bottom=0.22)
#plt.legend(loc='upper right')
# Always save at a high dpi so image quality is good
plt.savefig('SL_5_relative_final.png', dpi=600)

