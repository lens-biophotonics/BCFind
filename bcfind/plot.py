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


def sliding_volume_with_cells(
    volume, cell_coord, cell_colors=None, slice_width=10, fps=6, **kwargs
):
    volume = (volume * 255).astype("uint8")

    if cell_colors is not None:
        cell_colors = np.array(cell_colors)
    else:
        cell_colors = np.repeat("red", cell_coord.shape[0])

    def get_slice_cells_idx(points, z, slice_width):
        pts_idx = (points["z"] >= z - slice_width // 2) & (
            points["z"] < z + slice_width // 2
        )
        return pts_idx

    pts_idx = get_slice_cells_idx(cell_coord, 0, slice_width)
    slice_pts = cell_coord.loc[pts_idx]

    fig, ax = plt.subplots(**kwargs)
    img = ax.imshow(
        volume[..., 0],
        interpolation="none",
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=255,
    )
    sc = ax.scatter(slice_pts["y"], slice_pts["x"], c=cell_colors[pts_idx], s=15)
    plt.close()

    def init():
        pts_idx = get_slice_cells_idx(cell_coord, 0, slice_width)
        slice_pts = cell_coord.loc[pts_idx]

        img.set_data(volume[..., 0])
        sc.set_offsets(np.c_[slice_pts["y"], slice_pts["x"]])
        sc.set_color(c=cell_colors[pts_idx])
        return img, sc

    def animate_func(z):
        pts_idx = get_slice_cells_idx(cell_coord, z, slice_width)
        slice_pts = cell_coord.loc[pts_idx]

        img.set_data(volume[..., z])
        sc.set_offsets(np.c_[slice_pts["y"], slice_pts["x"]])
        sc.set_color(c=cell_colors[pts_idx])

        return img, sc

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        init_func=init,
        frames=volume.shape[2],
        interval=1000 / fps,
        blit=True,
    )

    rc("animation", html="html5")
    return anim
