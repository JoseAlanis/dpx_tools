# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD-3-Clause

# -- WIP -- {
fig, ax = plt.subplots()
im = ax.imshow(np.median(channel_corrs, axis=0), cmap="YlGn", )
# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Channels correlation', rotation=-90, va="bottom")
# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(n_channels), labels=channels, rotation=-90)
ax.set_yticks(np.arange(n_channels), labels=channels)
# }