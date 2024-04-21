#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm
from Tests.AverageLocVol import calc_centroid_volume
plt.style.use("default")


path = '/Users/daneverett/PycharmProjects/MSciProject/Sampling/WriteUpSamples/Centroid_Universes/'
#path = 'c:\\Users\manco\OneDrive\Ambiente de Trabalho\Masters_Project\MsciProj\Sampling/WriteUpSamples/Centroid_Universes/'

loc_volumes = [37758203, 3985608, 214834 ]

cmap = ['winter']*3


title = 'Centroid'

investigated_characteristic = "CentroidSigma"
N_centroids = [10,15,20,25]#,20,25
investigated_values = [0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.72, 0.8]
investigated_value_volumes = [calc_centroid_volume(value, gen_size=625) for value in investigated_values]

# investigated_values = [0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.40]
# max_numbers = ["0","0","0","0","0", "2", "0", "0", "0", "0", "2"]
max_numbers = ["0"]*(len(investigated_values)+1)
event_count_max_numbers = ["0"]*(len(investigated_values)+1)



fig, (ax) = plt.subplots(2, 3, figsize = (13,7.5))
fig.subplots_adjust(hspace=0)

markers = ['o', 'v', 's']

for axis_third, investigated_characteristic in enumerate(["LargeLocVol_CentroidSigma", "15LocVol_CentroidSigma","CentroidSigma"]):

    if axis_third==1:
        # N_centroids = [20]
        investigated_values = [0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.72, 0.8]
    else:
        N_centroids = [10, 15, 20, 25]
        investigated_values = [0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24]

    spectral_map = cm.get_cmap(cmap[axis_third], len(N_centroids))
    colors = spectral_map(np.linspace(0, 1, len(N_centroids)))

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
                if _<10000:
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
        if axis_third==3:
            ax[1,axis_third].plot(investigated_values_x[0:len(investigated_values)], [np.mean(stds) for stds in stdss], ls = '--',
                       c = colors[j],
                       label = r'$N_c = {}$'.format(str(N_gal)))

        ax[0,axis_third].errorbar(investigated_values_x[0:len(investigated_values)], [np.mean(alphas) for alphas in alphass],
                       yerr = [np.std(alphas)/np.sqrt(len(stds)) for alphas in alphass], ls = "--", fmt = markers[axis_third],
                   c=colors[j])
        ax[1,axis_third].errorbar(investigated_values_x[0:len(investigated_values)], [np.mean(stds) for stds in stdss],
                       yerr = [np.std(stds)/np.sqrt(len(stds)) for stds in stdss], ls = "--", fmt = markers[axis_third],
                        c=colors[j])
    if axis_third!=3:
        ax[0, axis_third].set_xscale("log")
        ax[1, axis_third].set_xscale("log")

    ax[0,axis_third].set_ylabel(r"$\alpha_{\sigma_{H_0}}$", fontsize = 20)

    ax[0,axis_third].set_xlabel(r"$\frac{V_{\text{cluster}}}{V_{\text{loc}}} (Mpc^{-1}) $", fontsize = 20)
    ax[1,axis_third].set_ylabel(r"$\frac{\sigma_{\hat{H}_0}}{H_0}$", fontsize = 20)

ax[1,2].legend(fontsize = 20)




image_format = 'jpeg' # e.g .png, .svg, etc.
image_name = 'Clustering_LocVol' + "." +image_format

plt.savefig("HighRes/"+image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)
plt.savefig("LowRes/"+image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=120)

plt.show()
# %%
