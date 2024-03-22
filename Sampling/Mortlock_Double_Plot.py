import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
import matplotlib.patches as mpatches
plt.style.use("default")
from matplotlib.pyplot import cm


investigated_characteristic = 'trial_survey_completeness_0'
#investigated_values = [25,75,95]
investigated_values = np.array([1.0]) #,0.5])
investigated_characteristic = 'bvmf_proportional_p_det_many'
investigated_values = [False, True]
max_numbers = ['0']*2
def expected(data, sig):
    sig = np.array(sig)
    S = np.sum(1/sig**2)
    data = np.array(data)
    return np.sum(data/(sig**2))/S, np.sqrt(1/S)

def bias_dist(x, H=70):
    ps = []
    inc = x.index[1]-x.index[0]
    cut_x = x.loc[x.index<=H]
    for i in cut_x.columns:
        ps.append(inc*np.sum(cut_x[i]))
    return ps

def C_I_samp(x):
    H = np.random.uniform(50,100)
    H = 70
    ps = []
    inc = x.index[1]-x.index[0]
    cut_x = x.loc[x.index<=H]
    for i in cut_x.columns:
        ps.append(inc*np.sum(cut_x[i]))
    return ps


def axis_namer(s):
    index = s.find('_')
    if index != -1:
        title = s[0].upper()+s[1:index]+' '+s[index+1].upper()+s[index+2:]
    else:
        title = s[0].upper()+s[1:]
    return title

def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

fig = plt.figure(figsize = (12,8))
# create grid for different subplots
spec = gridspec.GridSpec(ncols=1, nrows=2,
                        wspace=0.2,
                         hspace=0.3)

ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])

meanss = []
stdss = []
pos = []

p_i_s = []
c_i_s = []

for i in range(len(investigated_values)):
    print(i)
    filename = "PosteriorData/SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
    df = pd.read_csv(filename, index_col = 0)
    means = []
    stds = []
    inc = df.index[1]-df.index[0]
    p_i_s.append(bias_dist(df))
    c_i_s.append(C_I_samp(df))
    for column in df.columns:
        if is_unique(df[column]):
            print('Gotcha')
            continue
        pdf_single = df[column]/(inc * df[column].sum())
        #print(df[column].sum())
        pdf_single.dropna(inplace=True)
        vals = np.array(pdf_single.index)
        mean = sum(inc * pdf_single*vals)
        # means or modes
        #mean = vals[np.argmax(pdf_single*vals)]
        if mean==0:
            continue
        means.append(mean)
        stds.append(np.sqrt(sum((inc*pdf_single*pdf_single.index**2))-mean**2))
    meanss.append(means)
    stdss.append(stds)
    pos.append(i+1)



for i in range(len(investigated_values)):
    bias, bias_err = expected(meanss[i], stdss[i])
    s = 'H_0 = {:.2f}+/-{:.2f}'.format(bias, bias_err)
    print(s)


fig = plt.figure(figsize = (12,9.5), layout="constrained")
# create grid for different subplots
spec = gridspec.GridSpec(ncols=1, nrows=2,
                         wspace=0.2,
                         hspace=0.)

ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])
ax = [ax1,ax2]

ax1.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax2.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)

for i in ax:
    i.set_ylim(0,0.49)
    i.set_xlim(66,75)

plt.subplots_adjust(wspace=0, hspace=0)

b = 20


#bias_0 = np.mean(meanss[0])-70
#bias_1 = np.mean(meanss[1])-70
#bias_err_0 = np.std(meanss[0])
#bias_err_1 = np.std(meanss[1])

bias_0, bias_err_0 = expected(meanss[0], stdss[0])
bias_1, bias_err_1 = expected(meanss[1], stdss[1])

bias_0 -= 70
bias_1 -= 70

color = iter(cm.autumn(np.linspace(0, 1, 20)))

for i in range(len(investigated_values)):
        filename = "PosteriorData/SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
        df = pd.read_csv(filename, index_col = 0)
        means = []
        stds = []
        inc = df.index[1]-df.index[0]
        p_i_s.append(bias_dist(df))
        c_i_s.append(C_I_samp(df))
        for column in df.columns:
            if int(column)>80 and int(column)<90:
                c = next(color)
                if is_unique(df[column]):
                    print('Gotcha')
                    continue
                pdf_single = df[column]/(inc * df[column].sum())
                ax1.plot(df.index, pdf_single, c = c)

ax2.hist(meanss[1], bins=b, histtype='step', edgecolor='#e16462', lw=3.5, density=1, hatch='//')
ax2.vlines(x=bias_1+70, ymin=0, ymax=0.49,ls='dashed', lw=3, label='$\hat H_0-H_0={:.2f}\pm{:.2f}$'.format(bias_1, bias_err_1), color=  'r')
ax2.vlines(x=70, ymin=0, ymax=0.49, color='k', ls='dashed', lw=3)
ax1.vlines(x=70, ymin=0, ymax=0.49, color='k', ls='dashed', lw=3)


#ax1.grid(axis='x', ls='dashed', alpha=0.5)
#ax2.grid(axis='x', ls='dashed', alpha=0.5)


# ax2.set_ylabel('Included', fontsize=35, labelpad=40)
# ax1.set_ylabel('Not Included', fontsize=35, labelpad=40)


ax2.set_xlabel(r'$\hat H_0$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=35, labelpad=15)

for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(3)
    ax2.spines[axis].set_linewidth(3)


ax1.set_ylabel( r'$P(H_{0}|\text{sample})$', fontsize=35)
ax2.set_ylabel( r'$P(\hat{H}_{0}|\text{sample})$', fontsize=35)

# ax1.legend(fontsize=22)
ax2.legend(fontsize=22)
ax1.set_xticklabels([])

ax1.set_title('Selection effects', fontsize=40, pad=30)
#plt.tight_layout()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//Mortlock_Double.svg'


plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)

plt.show()