import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def colorize_predictions(value, vmin=None, vmax=None, vmax_95=True, cmap='viridis'):

    value = value.cpu().numpy()[0, 0, :, :]
    # value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    if vmax_95:
        vmax_95 = np.percentile(value, 95)
        vmax = vmax_95 if vmax is None else vmax
    else:
        vmax = value.max() if vmax is None else vmax


    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]
    # tensor = torch.from_numpy(img.transpose((2, 0, 1)))/ 255.0

    return img