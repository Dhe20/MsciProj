import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

def sel_eff(x,D,c,k):
    return 1 - (1 + (D/(x*300000/70))**c)**(-k)

cs = [4.035, 6.382, 13.804, 28.878]
cs.sort(reverse=True)
unc = [5, 10, 20, 30]
x = np.linspace(0,0.15,10000)
investigated_values = unc


color = iter(cm.winter(np.linspace(0, 1, len(investigated_values))))
for i in range(len(cs)):
    y = sel_eff(x, 250, cs[i], 2)
    c = next(color)
    ax.plot(x,y,ls='dashed', lw=5, c=c, label=r'$\sigma_D/D = {}\%$'.format(unc[i]))

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.5)

#ax.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.legend(fontsize = 28, loc='upper right', framealpha=1)
ax.set_ylabel(r'$P_{\mathrm{det}}(D_{\mathrm{GW}}|z,H_0)$', fontsize=45, labelpad=15)
ax.set_xlabel(r'$z$', fontsize=45, labelpad=15)
# ax.set_ylim(0.003,0.15)
# ax.set_xlim(4,1100)
# ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//Pdet.svg'

plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)

plt.show()
plt.show()