import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


def get_idxs_in_section(array, section_center, section_width=1, axis=0):
    in_section = (array[:, axis] >= section_center - section_width // 2) & (
        array[:, axis] < section_center + section_width // 2 + 1
    )
    return in_section


def make_video(
    input,
    target=None,
    nn_pred=None,
    centers=None,
    center_persistance=5,
    slide_axis=0,
    out_filename=None,
    fps=2,
):
    plt_axes = [ax for ax in range(len(input.shape)) if ax != slide_axis]
    n_frames = input.shape[slide_axis]
    x = input.astype("uint8")

    # create separator between input, target and nn_pred
    if target is not None or nn_pred is not None:
        sep_shape = list(input.shape)
        sep_shape[plt_axes[1]] = 3
        sep = np.empty(sep_shape, dtype="uint8") * np.nan

    # concatenate target and/or nn_pred to input
    if target is not None:
        x = np.concatenate([x, sep, target.astype("uint8")], axis=plt_axes[1])

    if nn_pred is not None:
        if centers is not None:
            # create shifted coordinate to plot over nn_pred
            nn_pred_centers = np.array(centers)[:, :3]
            nn_pred_centers[:, plt_axes[1]] += x.shape[plt_axes[1]] + 3

        x = np.concatenate([x, sep, nn_pred.astype("uint8")], axis=plt_axes[1])

    # Set scatter to plot
    if centers is not None:
        try:
            colors = np.array(centers.loc[:, ["R", "G", "B"]]) / 255
            print("Found RGB columns in center_pred. Using them in scatterplot")
        except KeyError:
            print("No RGB columns in center_pred. Using orange for all centers")
            colors = np.repeat("orange", centers.shape[0])

        centers = np.array(centers)[:, :3]
        in_section = get_idxs_in_section(
            centers, section_center=0, section_width=center_persistance, axis=slide_axis
        )

    # Set figure
    ratio = x.shape[plt_axes[1]] / x.shape[plt_axes[0]]
    fig = plt.figure(figsize=(int(8 * ratio), 8))
    plt.grid(b=True, which="major", color="#4444aa", alpha=0.5)
    plt.grid(b=True, which="minor", color="#225500", alpha=0.5)
    plt.axis("off")

    img = plt.imshow(
        x.take(0, axis=slide_axis),
        interpolation="none",
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=255,
    )
    if centers is not None:
        sct = plt.scatter(
            centers[in_section, plt_axes[1]],
            centers[in_section, plt_axes[0]],
            edgecolors=colors[in_section],
            s=200,
            c="none",
        )
        if nn_pred is not None:
            sct2 = plt.scatter(
                nn_pred_centers[in_section, plt_axes[1]],
                nn_pred_centers[in_section, plt_axes[0]],
                edgecolors=colors[in_section],
                s=200,
                c="none",
            )

    def animate_func(z):
        img.set_data(x.take(z, axis=slide_axis))

        if centers is not None:
            in_section = get_idxs_in_section(
                centers,
                section_center=z,
                section_width=center_persistance,
                axis=slide_axis,
            )

            sct.set_offsets(
                np.c_[
                    centers[in_section, plt_axes[1]], centers[in_section, plt_axes[0]]
                ]
            )
            sct.set_edgecolors(c=colors[in_section])

            if nn_pred is not None:
                sct2.set_offsets(
                    np.c_[
                        nn_pred_centers[in_section, plt_axes[1]],
                        nn_pred_centers[in_section, plt_axes[0]],
                    ]
                )
                sct2.set_edgecolors(c=colors[in_section])
                return img, sct, sct2
            else:
                return img, sct
        else:
            return img

    anim = FuncAnimation(
        fig,
        animate_func,
        frames=n_frames,
        interval=1000 / fps,  # in ms
    )
    if out_filename:
        anim.save(out_filename, fps=fps)

    plt.close(fig)
    return anim
