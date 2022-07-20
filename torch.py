import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def test():
    path_to_save_img = os.path.join(os.path.dirname(os.path.realpath(__file__)),'static/images/img.png')

    x = np.array([1,2,4,5])
    y = np.array([2,7,11,8])

    plt.plot(x,y)
    plt.savefig(path_to_save_img)
    print('Image Saved..')


