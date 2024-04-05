import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm

def expected(data, sig):
    sig = np.array(sig)
    S = np.sum(1/sig**2)
    data = np.array(data)
    return np.sum(data/(sig**2))/S, np.sqrt(1/S)

title = 'Centroid'

investigated_characteristic = "CentroidSigma"
N_centroids = [10,15,20,25]
investigated_values = [0.04, 0.08, 0.12, 0.16, 0.2]
# investigated_values = [0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.40]
# max_numbers = ["0","0","0","0","0", "2", "0", "0", "0", "0", "2"]
max_numbers = ["0"]*len(investigated_values)
event_count_max_numbers =  ["0"]*len(investigated_values)

spectral_map = cm.get_cmap('winter', len(N_centroids))
colors = spectral_map(np.linspace(0, 1, len(N_centroids)))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0)

for j, N_gal in enumerate(N_centroids):
    means = []
    stds = []
    alphas = []
    # for column in df.columns:
    post_avg = []
    N = np.array(investigated_values)
    meanss = []
    stdss = []
    alphass = []
    pos = []
    df_N = pd.DataFrame()

    for i in range(len(investigated_values)):
        # print(i)
        filename = "SampleUniverse_" + str(investigated_characteristic) + "_" + str(N_gal)+ "_" +str(investigated_values[i]) + "_" + \
                   max_numbers[i] + ".csv"
        df = pd.read_csv(filename, index_col=0)
        df.dropna(inplace=True, axis=1)
        filename = "EventCount_SampleUniverse_" + str(investigated_characteristic) + "_" + str(N_gal)+ "_" +str(investigated_values[i]) + "_" + \
                   event_count_max_numbers[i] + ".csv"
        df_event_count = pd.read_csv(filename, index_col=0)
        df_event_count.dropna(inplace=True, axis=1)
        means = []
        stds = []
        alphas = []
        for i, column in enumerate(df.columns):
            pdf_single = df[column] / df[column].sum()  # * (df.index[1] - df.index[0])
            # pdf_single.dropna(inplace=True)
            vals = np.array(pdf_single.index)
            event_count = df_event_count[column][0]
            mean = sum(pdf_single * vals)
            # means or modes
            # mean = vals[np.argmax(pdf_single*vals)]
            means.append(mean)
            stds.append(np.sqrt(sum((pdf_single * pdf_single.index ** 2)) - mean ** 2))
            alphas.append(np.sqrt(sum((pdf_single * pdf_single.index ** 2)) - mean ** 2) * np.sqrt(event_count))
        # df_N[str(investigated_values[i])] = df.mean(axis=1)
        # df_N[str(investigated_values[i])] = df_N[str(investigated_values[i])] / df_N[
        #     str(investigated_values[i])].sum()
        if j == 0:
            good_samples = list(np.where(df_event_count.values[0]>=10)[0])
        means = [mean for i, mean in enumerate(means) if i in good_samples]
        stds = [std for i, std in enumerate(stds) if i in good_samples]
        alphas = [alpha for i, alpha in enumerate(alphas) if i in good_samples]
        # stdss = stdss[good_samples]
        # alphass = alphass[good_samples]

        meanss.append(means)
        stdss.append(stds)
        alphass.append(alphas)
        pos.append(i + 1)


    # print(len(meanss))


    stdss = [list(np.array(stds)/70) for stds in stdss]


    ax1.plot(investigated_values, [np.mean(alphas) for alphas in alphass] ,"--x", color = colors[j])
    ax2.plot(investigated_values, [np.mean(stds) for stds in stdss] ,"--x", color = colors[j], label = r'$N_c = {}$'.format(str(N_gal)))

ax1.set_ylabel(r"$\alpha_{\sigma_{H_0}}$", fontsize = 20)

ax2.set_xlabel(r"$\frac{\sigma_g}{s} (Mpc^{-1}) $", fontsize = 20)
# plt.ylabel(r"$\frac{\sigma_{\hat{H}_0}}{H_0}$", fontsize = 16)
ax2.set_ylabel(r"$\frac{\sigma_{\hat{H}_0}}{H_0}$", fontsize = 20)

ax2.legend()

plt.show()