import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

fraction = 1  # .05

norm = mpl.colors.Normalize(vmin=0, vmax=0.6)
cbar = ax.figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap='autumn'),
            ax=ax, pad=.05, fraction=fraction)
cbar.set_ticks([0,0.2,0.4,0.6])
cbar.set_ticklabels([0,0.2,0.4,0.6])
ax.axis('off')
# plt.show()

# fig, ax = plt.subplots(figsize=(2, 10))
# # fig.subplots_adjust(bottom=0.5)

# cmap = plt.cm.get_cmap("winter")
# norm = mpl.colors.Normalize(vmin=0, vmax=24)

# fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
#              cax=ax, orientation='vertical', label='Layers')
plt.savefig(f"bar.svg")