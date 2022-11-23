import copy
import glob
import re
import sys
import os

import cv2
import numpy as np
import numpy.ma as ma


def on_mouse_moving(event, x, y, flags, param):
    depth_map, showing_img = param
    if event == cv2.EVENT_MOUSEMOVE:
        xy = "%d,%d" % (x, y)
        # print(x,y)
        copyed = copy.deepcopy(showing_img)
        cv2.putText(copyed, fr"(x:{x},y:{y}){depth_map[y][x]}", (0, showing_img.shape[0]), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255),2)

        print(depth_map[y][x], end="\r")
        cv2.imshow("depth",copyed)


def on_mouse_moving_idepth(event, x, y, flags, param):
    depth_map, showing_img, inv_depth = param
    if event == cv2.EVENT_MOUSEMOVE:
        xy = "%d,%d" % (x, y)
        # print(x,y)
        copyed = copy.deepcopy(showing_img)
        cv2.putText(copyed, f"(x:{x},y:{y}){depth_map[y][x]:.2f}, {inv_depth[y][x]:.2f}", (0, showing_img.shape[0]), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255),2)

        print(depth_map[y][x], end="\r")
        cv2.imshow("depth",copyed)


def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


def write_pfm(path, image, scale=1):
    """Write pfm file.

    Args:
        path (str): pathto file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)


def convert_disparity_to_depth(disparity, reprojection_matrix, min_depth=0, max_depth=20):
    points_3d = cv2.reprojectImageTo3D(disparity, reprojection_matrix)
    depth = points_3d[:, :, -1]
    # depth = np.clip(depth, min_depth, max_depth)
    return depth


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


def get_scale_translation(pred_inv_depth: np.ndarray, gt_depth: float):
    """
    calculate relationship between predicted inverse depth and absolute depth
    
    pred_inv_depth (np.ndarray): predicted inverse depth, shape: (n,)
    gt_depth (float): ground truth depth
    """

    n = pred_inv_depth.shape[0]
    pred_inv_depth.reshape((n, 1))
    x = np.concatenate([pred_inv_depth, np.ones_like(pred_inv_depth)], axis=1)
    y = np.ones((n, 1)) * (1 / gt_depth)

    theta = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
    return theta


def draw_depth_map(depth):
    depth_map = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map = 255 - depth_map
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_TURBO)
    return depth_map


def convert_inv_depth_to_abs_depth(inv_depth, s=0.0009079487065604722, t=0.5743572543646746):
    return 1 / (inv_depth * s + t)


def convert_inv_depth_to_abs_depth2(inv_depth, s=-58.758263, t=631.6803):
    return (inv_depth - t) / s


def convert_inv_depth_to_abs_depth3(inv_depth, filepath):
    data = {
        "output/p30_220720-2/1/processed_static\\left_ELE-AL00_1658283379323.pfm": 
            [-83.04552275153364,596.2423836018882],
        "output/p30_220720-2/1/processed_static\\left_ELE-AL00_1658283381269.pfm":
            [-77.17128888139224,582.7661308499473],
        "output/p30_220720-2/1/processed_static\\left_ELE-AL00_1658283385571.pfm":
            [-105.06694444071529,717.1507384482996],
        "output/p30_220720-2/1/processed_static\\left_ELE-AL00_1658283386747.pfm":
            [-102.62229091168298,726.0620349606471],
        "output/p30_220720-2/1/processed_static\\left_ELE-AL00_1658283390650.pfm":
            [-93.9300920084905,739.3502776632162],
        "output/p30_220720-2/1/processed_static\\left_ELE-AL00_1658283391693.pfm":
            [-94.55958189124121,736.9124624146514]
    }

    print(filepath, data[filepath])
    return convert_inv_depth_to_abs_depth2(inv_depth, s=data[filepath][0], t=data[filepath][1])


if __name__ == "__main__":
    # result_folder = "output/221005-1_1m_left"
    # result_folder = "output/221005-1_1m_left_small"
    result_folder = "output/221119-1_test1_left_small"
    # result_folder = "output/221119-1_test3_left_small"
    result_folder = "output/221119-1_test4_left_small"
    # result_folder = "output/221119-2_test1_left_small"
    result_folder = "output/221119-2_test2_left_small"
    # result_folder = "output/221119-2_test3_left_small"
    # result_folder = "output/221119-2_test4_left_small"
    # result_folder = "output/221119-2_test5_left_small"
    # result_folder = "output/221119-2_test6_left_small"
    # result_folder = "output/rgb_1663320740807"
    # result_folder = "output/model_illustration/outdoor/005"
    result_folder = "output/p30_220720-2/1/processed_static"
    disparity_files = sorted(glob.glob(f"{result_folder}/*.pfm"))

    # calibration_file = "output/v30p_right-crop_1.26014215_1.25802757.yml"
    calibration_file = "output/p30_1.03855444_1.05697224.yml"
    coefs = load_stereo_coefficients(calibration_file)
    K1, Q = coefs[0], coefs[-1]

    gt = 3

    for idx, disparity_file in enumerate(disparity_files):
        data, scale = read_pfm(disparity_file)
        # depth = convert_inv_depth_to_abs_depth(data)
        # depth = convert_inv_depth_to_abs_depth2(data)
        depth = convert_inv_depth_to_abs_depth3(data, disparity_file)
        # depth = convert_disparity_to_depth(data, Q) * 100
        # depth = np.clip(depth, 0, 8)

        depth_map = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_map = 255 - depth_map
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_TURBO)
        
        filename = os.path.split(disparity_file)[1].split(".")[0]
        cv2.imwrite(f"{result_folder}/{filename}_depth.png", depth_map)

        if not filename.startswith("left"): continue

        cv2.imshow("depth", depth_map)
        # cv2.setMouseCallback('depth', on_mouse_moving, [depth, depth_map])
        cv2.setMouseCallback('depth', on_mouse_moving_idepth, [depth, depth_map, data])
        cv2.waitKey(0)

