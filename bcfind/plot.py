import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from matplotlib.ticker import MultipleLocator


def make_video(volume, out_filename):
    x = (255 * volume).astype(np.uint8)

    print("Making video of", out_filename, "maxval=", x.max())
    fps = 8
    n_seconds = x.shape[2] // fps
    ratio = x.shape[1] / x.shape[0]

    fig = plt.figure(figsize=(int(8 * ratio), 8))
    plt.grid(b=True, which="major", color="#4444aa", alpha=0.5)
    plt.grid(b=True, which="minor", color="#225500", alpha=0.5)

    im = plt.imshow(
        x[..., 0],
        interpolation="none",
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=255,
    )
    plt.colorbar()

    ax = fig.axes[0]
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(10))

    def animate_func(z):
        im.set_array(x[..., z])
        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=n_seconds * fps,
        interval=1000 / fps,  # in ms
    )
    anim.save(out_filename, fps=fps, extra_args=["-vcodec", "libx264"])


def make_video_with_predictions(
    volume, bcfind_pred, slide_axis=0, slice_width=10, fps=6, **kwargs
):
    volume = volume - volume.min()
    volume = (volume / volume.max()) * 255
    plt_axes = [ax for ax in range(len(volume.shape)) if ax != slide_axis]

    def get_slice_cells_idx(points, sl):
        pts_idx = (points[:, slide_axis] >= sl - slice_width // 2) & (
            points[:, slide_axis] < sl + slice_width // 2
        )
        return pts_idx

    # slice_pts = bcfind_pred.loc[pts_idx, :]
    colors = np.array(bcfind_pred.loc[:, ['R', 'G', 'B']]) / 255
    bcfind_pred = np.array(bcfind_pred)
    pts_idx = get_slice_cells_idx(bcfind_pred, 0)

    fig, ax = plt.subplots(**kwargs)
    img = ax.imshow(
        volume.take(0, axis=slide_axis),
        interpolation="none",
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=255,
    )
    sc = ax.scatter(
        bcfind_pred[pts_idx, plt_axes[1]], 
        bcfind_pred[pts_idx, plt_axes[0]], 
        edgecolors=colors[pts_idx, :], 
        s=150, 
        c='none'
        )
    
    plt.axis('off')
    plt.close()

    def init():
        pts_idx = get_slice_cells_idx(bcfind_pred, 0)
        slice_pts = bcfind_pred[pts_idx, :]

        img.set_data(volume.take(0, axis=slide_axis))
        sc.set_offsets(np.c_[slice_pts[:, plt_axes[1]], slice_pts[:, plt_axes[0]]])
        sc.set_edgecolors(c=colors[pts_idx, :])
        return img, sc

    def animate_func(sl):
        pts_idx = get_slice_cells_idx(bcfind_pred, sl)
        slice_pts = bcfind_pred[pts_idx, :]

        img.set_data(volume.take(sl, axis=slide_axis))
        sc.set_offsets(np.c_[slice_pts[:, plt_axes[1]], slice_pts[:, plt_axes[0]]])
        sc.set_edgecolors(c=colors[pts_idx, :])

        return img, sc

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        init_func=init,
        frames=volume.shape[slide_axis],
        interval=1000 / fps,
        blit=True,
    )

    # rc("animation", html="html5")
    return anim
