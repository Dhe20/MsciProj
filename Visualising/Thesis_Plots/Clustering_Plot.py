#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm
from Tests.AverageLocVol import calc_centroid_volume

path = '/Users/daneverett/PycharmProjects/MSciProject/Sampling/WriteUpSamples/Centroid_Universes/'
path = 'c:\\Users\manco\OneDrive\Ambiente de Trabalho\Masters_Project\MsciProj\Sampling/WriteUpSamples/Centroid_Universes/'


def expected(data, sig):
    sig = np.array(sig)
    S = np.sum(1/sig**2)
    data = np.array(data)
    return np.sum(data/(sig**2))/S, np.sqrt(1/S)

loc_volumes = [214834, 3985608, 37758203]



title = 'Centroid'

investigated_characteristic = "CentroidSigma"
N_centroids = [10,15,20,25]#,20,25
investigated_values = [0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24]
investigated_values = [0.04, 0.08, 0.12, 0.16, 0.2, 0.24]
investigated_value_volumes = [calc_centroid_volume(value, gen_size=625) for value in investigated_values]

# investigated_values = [0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.40]
# max_numbers = ["0","0","0","0","0", "2", "0", "0", "0", "0", "2"]
max_numbers = ["0"]*len(investigated_values)
event_count_max_numbers = ["0"]*len(investigated_values)

spectral_map = cm.get_cmap('winter', len(N_centroids))
colors = spectral_map(np.linspace(0, 1, len(N_centroids)))

fig, (ax) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0)

markers = ['o', 'v', 's']

for axis_third, investigated_characteristic in enumerate(["CentroidSigma", "15LocVol_CentroidSigma","LargeLocVol_CentroidSigma"]):
    if axis_third==1:
        N_centroids = [20,10,15]  # ,20,25
    else:
        N_centroids = [10, 15, 20, 25]  # ,20,25
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
            filename = path + "SampleUniverse_" + str(investigated_characteristic) + "_" + str(N_gal)+ "_" +str(investigated_values[i]) + "_" + \
                       max_numbers[i] + ".csv"
            df = pd.read_csv(filename, index_col=0)
            df.dropna(inplace=True, axis=1)
            filename = path + "EventCount_SampleUniverse_" + str(investigated_characteristic) + "_" + str(N_gal)+ "_" +str(investigated_values[i]) + "_" + \
                       event_count_max_numbers[i] + ".csv"
            df_event_count = pd.read_csv(filename, index_col=0)
            df_event_count.dropna(inplace=True, axis=1)
            means = []
            stds = []
            alphas = []
            for _, column in enumerate(df.columns):
                pdf_single = df[column] / df[column].sum()  # * (df.index[1] - df.index[0])
                # pdf_single.dropna(inplace=True)
                vals = np.array(pdf_single.index)
                event_count = df_event_count[column][0]
                # if event_count<0:
                #     print(event_count)
                mean = sum(pdf_single * vals)
                # means or modes
                # mean = vals[np.argmax(pdf_single*vals)]
                means.append(mean)
                if np.sum((pdf_single * pdf_single.index ** 2)) - mean ** 2<0:
                    continue
                stds.append(np.sqrt(np.sum((pdf_single * pdf_single.index ** 2)) - mean ** 2))
                alphas.append(np.sqrt(np.sum((pdf_single * pdf_single.index ** 2)) - mean ** 2) * np.sqrt(event_count))
            # df_N[str(investigated_values[i])] = df.mean(axis=1)
            # df_N[str(investigated_values[i])] = df_N[str(investigated_values[i])] / df_N[
            #     str(investigated_values[i])].sum()
            if i == 0:
                good_samples = list(np.where(df_event_count.values[0]>=10)[0])
            means = [mean for k, mean in enumerate(means) if k in good_samples]
            stds = [std for k, std in enumerate(stds) if k in good_samples]
            alphas = [alpha for k, alpha in enumerate(alphas) if k in good_samples]
            # stdss = stdss[good_samples]
            # alphass = alphass[good_samples]

            meanss.append(means)
            stdss.append(stds)
            alphass.append(alphas)
            pos.append(i + 1)


        # print(len(meanss))


        stdss = [list(np.array(stds)/70) for stds in stdss]

        investigated_values_x = np.array(investigated_value_volumes)/loc_volumes[axis_third]

        if axis_third == 0:
            ax[0].plot(investigated_values_x, [np.mean(alphas) for alphas in alphass] ,"--"+markers[axis_third], color = colors[j], label = r'$N_c = {}$'.format(str(N_gal)))
            ax[1].plot(investigated_values_x, [np.mean(stds) for stds in stdss] ,"--"+markers[axis_third], color = colors[j])

        else:
            ax[0].plot(investigated_values_x, [np.mean(alphas) for alphas in alphass], "--" + markers[axis_third],
                       color=colors[j])
            ax[1].plot(investigated_values_x, [np.mean(stds) for stds in stdss], "--" + markers[axis_third],
                       color=colors[j])

ax[0].set_ylabel(r"$\alpha_{\sigma_{H_0}}$", fontsize = 20)

ax[1].set_xlabel(r"$\frac{\sigma_g}{s} (Mpc^{-1}) $", fontsize = 20)
ax[1].set_ylabel(r"$\frac{\sigma_{\hat{H}_0}}{H_0}$", fontsize = 20)

ax[0].legend(fontsize = 20)

ax[0].set_xscale("log")
ax[1].set_xscale("log")

plt.show()
# %%
