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

title = 'Clustering'

investigated_characteristic = "clustering"
N_gals = [5000,2500,1250, 625.0,round(625/2,1), round(625/4,1)]
investigated_values = [0.01, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# max_numbers = ["0","0","0","0","0", "2", "0", "0", "0", "0", "2"]
max_numbers = ["0"]*11

spectral_map = cm.get_cmap('winter', len(N_gals))
colors = spectral_map(np.linspace(0, 1, len(N_gals)))


for j, N_gal in enumerate(N_gals):

    if N_gal <= 625.0:
        investigated_values = [0.01,1,2,3,4,5,6,7]
    means = []
    stds = []
    # for column in df.columns:
    post_avg = []
    N = np.array(investigated_values)
    meanss = []
    stdss = []
    pos = []
    df_N = pd.DataFrame()

    for i in range(len(investigated_values)):
        # print(i)
        filename = "SampleUniverse_" + str(investigated_characteristic) + "_" + str(N_gal)+ "_" +str(investigated_values[i]) + "_" + \
                   max_numbers[i] + ".csv"
        df = pd.read_csv(filename, index_col=0)
        df.dropna(inplace=True, axis=1)
        means = []
        stds = []
        for column in df.columns:
            pdf_single = df[column] / df[column].sum()  # * (df.index[1] - df.index[0])
            # pdf_single.dropna(inplace=True)
            vals = np.array(pdf_single.index)
            mean = sum(pdf_single * vals)
            # means or modes
            # mean = vals[np.argmax(pdf_single*vals)]
            if mean == 0:
                continue
            means.append(mean)
            stds.append(np.sqrt(sum((pdf_single * pdf_single.index ** 2)) - mean ** 2))
        df_N[str(investigated_values[i])] = df.mean(axis=1)
        df_N[str(investigated_values[i])] = df_N[str(investigated_values[i])] / df_N[
            str(investigated_values[i])].sum()
        meanss.append(means)
        stdss.append(stds)
        pos.append(i + 1)

    # print(len(meanss))


    stdss = np.array(stdss)/70


    plt.plot(investigated_values, np.mean(stdss, axis=1),"--x", color = colors[j], label = r"$N_{gal} = $" + str(N_gal))

plt.xlabel(r"$\zeta_0$")
plt.ylabel(r"$\frac{\sigma_{\hat{H}_0}}{H_0}$", fontsize = 16)
plt.legend()
plt.show()
# fig = plt.figure(figsize = (24,20))
# # create grid for different subplots
# spec = gridspec.GridSpec(ncols=2, nrows=len(investigated_values),
#                          wspace=0.2,
#                          hspace=0.)
#
