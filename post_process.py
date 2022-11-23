import glob
from itertools import product
import os

import cv2
import numpy as np
import numpy.ma as ma

from utils import read_pfm


def convert_disparity_to_depth(disparity, reprojection_matrix, min_depth=0, max_depth=20):
    points_3d = cv2.reprojectImageTo3D(disparity, reprojection_matrix)
    depth = points_3d[:, :, -1]
    # depth = np.clip(depth, min_depth, max_depth)
    return depth


def convert_inv_depth_to_abs_depth(inv_depth, s=0.0036052485398046924, t=-0.5836624602235267):
    return 1 / (inv_depth * s + t)


def load_stereo_coefficients(path):
    """ Loads stereo matrix coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    R1 = cv_file.getNode("R1").mat()
    R2 = cv_file.getNode("R2").mat()
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()

    cv_file.release()
    return K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q


def load_roi_file(roi_file_path):
    result = {}
    with open(roi_file_path, "r") as f:
        result = {i.split()[0]: [int(j) if j.isdigit() else float(j) for j in i.split()[1:]] for i in f.readlines()}
    
    return result


def draw_depth_map(depth):
    depth_map = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map = 255 - depth_map
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_TURBO)
    return depth_map


def draw_depth_map_batch(result_folder, calibration_file):
    disparity_files = glob.glob(f"{result_folder}/*.pfm")
    Q = load_stereo_coefficients(calibration_file)[-1]

    for disparity_file in disparity_files:
        disp, _ = read_pfm(disparity_file)
        depth = convert_disparity_to_depth(disp, Q)
        depth_map = draw_depth_map(depth)
        
        filename = os.path.split(disparity_file)[1].split(".")[0]
        cv2.imwrite(f"{result_folder}/{filename}_depth.png", depth_map)


def get_scale_translation(pred_inv_depth: np.ndarray, gt_depth: float):
    """
    calculate relationship between predicted inverse depth and absolute depth
    
    pred_inv_depth (np.ndarray): predicted inverse depth, shape: (n,)
    gt_depth (float): ground truth depth
    """

    n = pred_inv_depth.shape[0]
    pred_inv_depth = pred_inv_depth.reshape((n, 1)) / 100
    # X = pred_inv_depth
    X = np.concatenate([pred_inv_depth, np.ones_like(pred_inv_depth)], axis=1)
    y = np.ones((n, 1)) * (1 / gt_depth)

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X1, X2, y1, y2 = train_test_split(X, y, train_size=4/n)
    
    # s, t = np.linalg.lstsq(X1, y1, rcond=None)[0]
    # print(s, t)
    # return s, t

    # reg = LinearRegression().fit(X1, y1)
    # print(f"score: {reg.score(X1, y1)}, coef: {reg.coef_}, intercept: {reg.intercept_}")
    # return reg.coef_

    theta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
    theta = theta.reshape((-1,))
    return theta


inv_depths = []
inv_gts = []


def calculate_precision(result_folder, calibration_file, roi_file, min_depth=0, max_depth=8):
    disparity_files = glob.glob(f"{result_folder}/*.pfm")
    Q = load_stereo_coefficients(calibration_file)[-1]

    filename_to_bbox = load_roi_file(roi_file)
    result = []

    for disparity_file in disparity_files:
        disp, _ = read_pfm(disparity_file)
        # depth = convert_disparity_to_depth(disp, Q)
        depth = convert_inv_depth_to_abs_depth(disp)
        depth = np.clip(depth, min_depth, max_depth)
        depth_map = draw_depth_map(depth)
        
        filename = os.path.split(disparity_file)[1].split(".")[0]
        cv2.imwrite(f"{result_folder}/{filename}_depth.png", depth_map)

        if not filename.startswith('left'): continue

        filename_ext = "%s.png" % filename
        x, y, w, h, gt = filename_to_bbox[filename_ext]
        roi_depth = depth[y:y+h, x:x+w]
        roi_depth_mask = ma.masked_outside(roi_depth, min_depth, max_depth)

        error = np.abs(roi_depth_mask - gt) / gt
        result.append([
            filename_ext, gt,
            np.min(roi_depth_mask), np.max(roi_depth_mask),np.mean(roi_depth_mask), 
            np.min(error), np.max(error), np.mean(error)
        ])

        # roi_disp = disp[y:y+h, x:x+w]
        # print(f"baseline*focal = {np.mean(roi_disp)}*{gt} = {np.mean(roi_disp)*gt}")

        # theta = get_scale_translation(roi_depth.reshape((-1,)), gt)
        # print(f"scale and translation for {gt}: {theta}")

        inv_depth = disp.reshape((-1,))
        inv_gt = np.ones_like(inv_depth) / gt
        inv_depths.append(inv_depth)
        inv_gts.append(inv_gt)
    
    return result


def summarize_precision(precision_results):
    depth_gts = [0.5, 1, 3, 5]
    gt_to_error_means = {depth_gt: [] for depth_gt in depth_gts}
    results = []

    for precision_result in precision_results:
        gt = precision_result[1]
        if gt not in depth_gts: continue

        gt_to_error_means[gt].append(precision_result[-1])
    
    for gt, error_means in gt_to_error_means.items():
        if len(error_means) is 0: continue

        results.append([
            None, str(gt), 0, 0, 0, 0, 0, np.mean(error_means)
        ])
    
    return results


def save_precision_results(precision_results, result_filepath):
    precision_results = [[str(i) for i in precision_result] for precision_result in precision_results]
    
    with open(result_filepath, "w") as f:
        f.write("filename,gt,min_depth,max_depth,avg_depth,min_error,max_error,avg_error\n")
        for line in precision_results:
            f.write(",".join(line))
            f.write("\n")


if __name__ == "__main__":
    # result_folder = "output/221005-1_1m_left"
    # result_folder = "output/221005-1_1m_left_small"
    # # result_folder = "output/221119-1_test1_left_small"
    # # result_folder = "output/rgb_1663320740807"

    # calibration_file = "input/v30p_right-crop_1.26014215_1.25802757.yml"
    # # calibration_file = "input/p30_1.03855444_1.05697224.yml"

    # draw_depth_map_batch(result_folder, calibration_file)

    depth_gts = [0.5, 1, 3, 5]
    moving_states = ['static', 'slow', 'fast']
    calibration_file = "input/p30_1.03855444_1.05697224.yml"

    results = {moving_state: [] for moving_state in moving_states}

    for depth_gt, moving_state in product(depth_gts, moving_states):
        result_folder = f"output/p30_220720-2/{depth_gt}/processed_{moving_state}"
        roi_file = f"input/p30_220720-2/{depth_gt}/processed_{moving_state}_{depth_gt}.roi"

        res = calculate_precision(result_folder, calibration_file, roi_file)
        results[moving_state].extend(res)
    
    for moving_state, moving_results in results.items():
        precision_summary = summarize_precision(moving_results)
        moving_results.extend(precision_summary)

        save_precision_results(moving_results, f"output/test_{moving_state}.csv")

    X = np.concatenate(inv_depths)
    X = np.vstack([X, np.ones(len(X))]).T
    y = np.concatenate(inv_gts)
    # X = 1 / np.concatenate(inv_gts)
    # X = np.vstack([X, np.ones(len(X))]).T
    # y = np.concatenate(inv_depths)

    # from sklearn.model_selection import train_test_split
    # X, _, y, _ = train_test_split(X, y, train_size=1000/len(y))

    s, t = np.linalg.lstsq(X, y, rcond=None)[0]
    print(s, t)
