import matplotlib.pyplot as plt
import numpy as np

def save_figure(figure: np.ndarray ,title: str, filename: str):
    plt.imshow(figure, cmap='gray')
    plt.colorbar()
    plt.title(title)
    plt.savefig(filename)
    plt.clf()
    plt.close("all")
