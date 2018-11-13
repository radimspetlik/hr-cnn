import matplotlib
from matplotlib.ticker import FormatStrFormatter

matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image


def opencv_colordim_switch(frame):
    new_data = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]), dtype='uint8')
    new_data[:, :, 0] = frame[:, :, 2]
    new_data[:, :, 1] = frame[:, :, 1]
    new_data[:, :, 2] = frame[:, :, 0]

    return new_data


def plot_torch_array_to_summary(summary, name, values, epoch, target=None, ylim=None):
    dpi = 100.0
    width = 770.0
    height = 278.0
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    if target is not None:
        x = np.arange(0.67, 4.0, 3.33 / values.size()[0])
        x = x[x.size - values.size()[0]:]
        plt.plot(x, values.data.cpu().numpy().flatten())
        plt.plot([target, target], [values.min().data[0], values.max().data[0]], 'r')
    else:
        plt.plot(np.arange(0, values.size()[0]), values.data.cpu().numpy().flatten())
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.subplots_adjust(left=0.075, bottom=0.1, right=0.99, top=0.97, wspace=0, hspace=0)
    if ylim is not None:
        plt.gca().set_ylim(ylim)
    fig.canvas.draw()
    im = Image.frombuffer('RGBA', (int(width), int(height)), fig.canvas.buffer_rgba(), 'raw', 'RGBA', 0, 1)
    plt.close()
    if summary is not None:
        summary.add_image(name, np.array(im), epoch)
    else:
        return np.array(im)
