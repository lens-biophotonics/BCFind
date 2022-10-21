import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def get_slice_center_idxs_from_array(centers, sl, slice_width=1, axis=0):
    pts_idx = (centers[:, axis] >= sl - slice_width // 2) & (centers[:, axis] < sl + slice_width // 2)
    return pts_idx


def make_video(input, target=None, nn_pred=None, center_pred=None, slide_axis=0, out_filename=None, fps=2):
    plt_axes = [ax for ax in range(len(input.shape)) if ax != slide_axis]
    n_seconds = input.shape[slide_axis] // fps
    
    # Set image to show
    # if target is specified input and target are superimposed on the red and blue channels respectively
    # if nn_pred is specified input and nn_pred are superimposed on the red and blue channels respectively
    # if both target and nn_pred are specified the 2 superimposition above are concatenated on the y axis, first the input+target then the input+nn_pred
    x = input.astype('uint8')
    
    if target is not None or nn_pred is not None:
        zeros_shape = list(input.shape)
        zeros_shape[plt_axes[1]] = 3
        zeros = np.zeros(zeros_shape, dtype='uint8')

    if target is not None:
        x = np.concatenate([x, zeros, target.astype('uint8')], axis=plt_axes[1])
    if nn_pred is not None:
        x = np.concatenate([x, zeros, nn_pred.astype('uint8')], axis=plt_axes[1])
    
    # Set scatter to plot
    if center_pred is not None:
        try:
            print('Found RGB columns in center_pred. Using them in scatterplot')
            colors = np.array(center_pred.loc[:, ['R', 'G', 'B']]) / 255
        except KeyError:
            print('No RGB columns in center_pred. Using orange for all centers')
            colors = 'orange'
        
        center_pred = np.array(center_pred)[:, :3]
        pts_idx = get_slice_center_idxs_from_array(center_pred, 0, slice_width=6, axis=slide_axis)

    # Set figure
    ratio = x.shape[plt_axes[1]] / x.shape[plt_axes[0]]
    fig = plt.figure(figsize=(int(8 * ratio), 8))
    plt.grid(b=True, which="major", color="#4444aa", alpha=0.5)
    plt.grid(b=True, which="minor", color="#225500", alpha=0.5)
    plt.axis('off')

    img = plt.imshow(
        x.take(0, axis=slide_axis),
        interpolation="none",
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=255,
    )
    if center_pred is not None:
        sct = plt.scatter(
            center_pred[pts_idx, plt_axes[1]], 
            center_pred[pts_idx, plt_axes[0]], 
            edgecolors=colors[pts_idx, :], 
            s=200, 
            c='none'
            )

    def animate_func(z):
        img.set_data(x.take(z, axis=slide_axis))
        if center_pred is not None:
            pts_idx = get_slice_center_idxs_from_array(center_pred, z, slice_width=6, axis=slide_axis)
            sct.set_offsets(np.c_[center_pred[pts_idx, plt_axes[1]], center_pred[pts_idx, plt_axes[0]]])
            sct.set_edgecolors(c=colors[pts_idx, :])
            return img, sct
        else:
            return img

    anim = FuncAnimation(
        fig,
        animate_func,
        frames=n_seconds * fps,
        interval=1000 / fps,  # in ms
    )
    if out_filename:
        anim.save(out_filename, fps=fps)

    return anim
