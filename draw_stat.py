import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.ticker as mtick
from matplotlib.patches import StepPatch
def show1(data, name):
    labels = 'image-wise 0to23(m) abs error, average=%.2f' % data.mean() + '(m)'

    fix, axs = plt.subplots(ncols=1, nrows=1, squeeze=False) # , figsize=(32,24))

    # perc = np.linspace(0,100,len(data))
    # axs[0, 0].plot(np.arange(1, data[0].shape[0] + 1), data[0])
    # axs[0, 0].plot(perc, data)

    n = data.shape[0]//20
    tmp = [data[n*i:n*(i+1)].mean() for i in range(20)]
    prec = np.arange(0, 20) * 5
    # axs[0, 0].stairs(tmp, prec, fill=True)
    axs[0, 0].plot((np.arange(0, data.shape[0]) /data.shape[0]) * 99.99999, data, marker='o', linestyle='None', markersize=3)

    axs[0, 0].xaxis.set_ticks([0, 5, 50, 95, 100])
    axs[0, 0].yaxis.set_ticks([0, 1, 2, 4, 8])
    axs[0, 0].set_xlabel("percentage (%)")
    axs[0, 0].set_ylabel("abs error")
    axs[0, 0].set_title(label=labels)

    axs[0, 0].axhline(y=1.0, color='brown', linestyle='-', linewidth=1, label='1m')
    # axs[0, 0].axhline(y=1.15, color='blue', linestyle='-', linewidth=1, label='5%')
    axs[0, 0].axvline(x=95.0, color='red', linestyle='-', linewidth=1, label='outlier')

    axs[0, 0].legend()

    fix.savefig(name)

def show2(data, name):
    labels = 'pixel-wise 0to23(m) abs error, average=%.2f' % data.mean() + '(m)'

    fix, axs = plt.subplots(ncols=1, nrows=1, squeeze=False) # , figsize=(32,24))

    # perc = np.linspace(0,100,len(data))
    # axs[0, 0].plot(np.arange(1, data[0].shape[0] + 1), data[0])
    # axs[0, 0].plot(perc, data)

    n = data.shape[0]//20
    tmp = [data[n*i:n*(i+1)].mean() for i in range(20)]
    prec = np.arange(0, 20) * 5
    # axs[0, 0].stairs(tmp, prec, fill=True)
    axs[0, 0].plot(np.arange(0, data.shape[0]) * (100/data.shape[0]), data)

    axs[0, 0].xaxis.set_ticks([0, 5, 50, 95, 100])
    axs[0, 0].yaxis.set_ticks([0, 1, 2, 4, 8])
    axs[0, 0].set_xlabel("percentage (%)")
    axs[0, 0].set_ylabel("abs error")
    axs[0, 0].set_title(label=labels)

    axs[0, 0].axhline(y=1.0, color='brown', linestyle='-', linewidth=1, label='1m')
    # axs[0, 0].axhline(y=1.15, color='blue', linestyle='-', linewidth=1, label='5%')
    axs[0, 0].axvline(x=95.0, color='red', linestyle='-', linewidth=1, label='outlier')

    axs[0, 0].legend() # title='')

    fix.savefig(name)

import glob
filenames = glob.glob('error/*.npy')

arr = []
for path in filenames:
    tmp = np.load(path)
    arr.append(tmp)

    #tmp = np.sort(tmp)
    #print("tmp.shape", tmp.shape)
    #show2(arr, path, 'pixel_wise_error.png')

img_arr = np.array([e.mean() for e in arr])
img_arr = np.sort(img_arr)
show1(img_arr, 'image_wise_error.png')
arr = np.concatenate(arr)
arr = np.sort(arr)
show2(arr, 'pixel_wise_error.png')
