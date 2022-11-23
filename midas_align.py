from glob import glob

import cv2
import numpy as np

from midas_vis_postprocess import read_pfm, draw_depth_map


def align_points():
    data = [
        [272.25, 1/3],
        [305.5, 1/3],
        [338.25, 1/3],
        [251, 1/5],
        [248, 1/5],
        [569.5, 1/1],
        [535, 1/1],
        [642, 1/1],
        [494, 1/0.5],
        [464.5, 1/0.5],
        [545.5, 1/0.5]
    ]

    data = np.array(data)
    # X = np.concatenate([data[:, 0].reshape((-1, 1)), np.ones((len(data), 1))], axis=1)
    # y = data[:, 1]
    X = 1 / np.concatenate([data[:, 1].reshape((-1, 1)), np.ones((len(data), 1))], axis=1)
    y = data[:, 0]
    # X = np.concatenate([data[:, 1].reshape((-1, 1)), np.ones((len(data), 1))], axis=1)
    # y = data[:, 0]

    s, t = np.linalg.lstsq(X, y, rcond=None)[0]
    print(s, t)


def read_sgbm_pfm(pfm_file, num_disparities=4*16) -> np.ndarray:
    data, _ = read_pfm(pfm_file)
    data = data[:, num_disparities:]
    data = data.flatten()
    return data


def align_pfm(sgbm_depth_file, midas_depth_file):
    # sgbm_depth_file = "input/p30_220720-2/1/sgbm_right/left_ELE-AL00_1658283379323_SGBM.pfm"
    # midas_depth_file = "output/p30_220720-2/1/processed_static/left_ELE-AL00_1658283379323.pfm"

    sgbm_depth = read_sgbm_pfm(sgbm_depth_file)
    midas_depth = read_sgbm_pfm(midas_depth_file)

    midas_depth = midas_depth[sgbm_depth < 10]
    sgbm_depth = sgbm_depth[sgbm_depth < 10]

    X = np.vstack([sgbm_depth, np.ones(len(sgbm_depth))]).T
    y = midas_depth

    # idx = np.random.choice(sgbm_depth.shape[0], 1000)
    # X = X[idx]
    # y = y[idx]

    s, t = np.linalg.lstsq(X, y, rcond=None)[0]
    print(s, t)


def align_pfm_batch():
    sgbm_depth_file_folder = "input/p30_220720-2/1/sgbm_right"
    midas_depth_file_folder = "output/p30_220720-2/1/processed_static"

    sgbm_depth_files = sorted(glob(sgbm_depth_file_folder + "/left_*.pfm"))
    midas_depth_files = sorted(glob(midas_depth_file_folder + "/left_*.pfm"))
    assert len(sgbm_depth_files) == len(midas_depth_files)

    sgbm_depth_data = [read_sgbm_pfm(file) for file in sgbm_depth_files]
    midas_depth_data = [read_sgbm_pfm(file) for file in midas_depth_files]

    X = np.concatenate(sgbm_depth_data)
    X = np.vstack([X, np.ones_like(X)]).T
    y = np.concatenate(midas_depth_data)

    s, t = np.linalg.lstsq(X, y, rcond=None)[0]
    print(s, t)


def align_pfm_per_image():
    sgbm_depth_file_folder = "input/p30_220720-2/1/sgbm_right"
    midas_depth_file_folder = "output/p30_220720-2/1/processed_static"

    sgbm_depth_files = sorted(glob(sgbm_depth_file_folder + "/left_*.pfm"))
    midas_depth_files = sorted(glob(midas_depth_file_folder + "/left_*.pfm"))
    assert len(sgbm_depth_files) == len(midas_depth_files)

    for sgbm_depth_file, midas_depth_file in zip(sgbm_depth_files, midas_depth_files):
        print(f"{midas_depth_file}: ", end="")
        align_pfm(sgbm_depth_file, midas_depth_file)


if __name__ == "__main__":
    align_pfm_batch()
    align_pfm_per_image()

