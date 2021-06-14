# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
import os.path as osp

from numpy.core.numeric import ones_like
import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import pickle
import open3d
import matplotlib.pyplot as plt

############# input parameters  #############
from demo.demo_options import DemoOptions
from bodymocap.body_mocap_api import BodyMocap
from handmocap.hand_mocap_api import HandMocap
import mocap_utils.demo_utils as demo_utils
import mocap_utils.general_utils as gnu
from mocap_utils.timer import Timer
from datetime import datetime

from bodymocap.body_bbox_detector import BodyPoseEstimator
from handmocap.hand_bbox_detector import HandBboxDetector
from integration.copy_and_paste import integration_copy_paste

import renderer.image_utils as imu
from renderer.viewer2D import ImShow


def __filter_bbox_list(body_bbox_list, hand_bbox_list, single_person):
    # (to make the order as consistent as possible without tracking)
    bbox_size = [(x[2] * x[3]) for x in body_bbox_list]
    idx_big2small = np.argsort(bbox_size)[::-1]
    body_bbox_list = [body_bbox_list[i] for i in idx_big2small]
    hand_bbox_list = [hand_bbox_list[i] for i in idx_big2small]

    if single_person and len(body_bbox_list) > 0:
        body_bbox_list = [
            body_bbox_list[0],
        ]
        hand_bbox_list = [
            hand_bbox_list[0],
        ]

    return body_bbox_list, hand_bbox_list


def run_regress(
    args,
    img_original_bgr,
    body_bbox_list,
    hand_bbox_list,
    bbox_detector,
    body_mocap,
    hand_mocap,
):
    cond1 = len(body_bbox_list) > 0 and len(hand_bbox_list) > 0
    cond2 = not args.frankmocap_fast_mode

    # use pre-computed bbox or use slow detection mode
    if cond1 or cond2:
        if not cond1 and cond2:
            # run detection only when bbox is not available
            (
                body_pose_list,
                body_bbox_list,
                hand_bbox_list,
                _,
            ) = bbox_detector.detect_hand_bbox(img_original_bgr.copy())
        else:
            print("Use pre-computed bounding boxes")
        assert len(body_bbox_list) == len(hand_bbox_list)

        if len(body_bbox_list) < 1:
            return list(), list(), list()

        # sort the bbox using bbox size
        # only keep on bbox if args.single_person is set
        body_bbox_list, hand_bbox_list = __filter_bbox_list(
            body_bbox_list, hand_bbox_list, args.single_person
        )

        # hand & body pose regression
        pred_hand_list = hand_mocap.regress(
            img_original_bgr, hand_bbox_list, add_margin=True
        )
        pred_body_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(hand_bbox_list) == len(pred_hand_list)
        assert len(pred_hand_list) == len(pred_body_list)

    else:
        _, body_bbox_list = bbox_detector.detect_body_bbox(img_original_bgr.copy())

        if len(body_bbox_list) < 1:
            return list(), list(), list()

        # sort the bbox using bbox size
        # only keep on bbox if args.single_person is set
        hand_bbox_list = [None,] * len(body_bbox_list)
        body_bbox_list, _ = __filter_bbox_list(
            body_bbox_list, hand_bbox_list, args.single_person
        )

        # body regression first
        pred_body_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(body_bbox_list) == len(pred_body_list)

        # get hand bbox from body
        hand_bbox_list = body_mocap.get_hand_bboxes(
            pred_body_list, img_original_bgr.shape[:2]
        )
        assert len(pred_body_list) == len(hand_bbox_list)

        # hand regression
        pred_hand_list = hand_mocap.regress(
            img_original_bgr, hand_bbox_list, add_margin=True
        )
        assert len(hand_bbox_list) == len(pred_hand_list)

    # integration by copy-and-paste
    integral_output_list = integration_copy_paste(
        pred_body_list, pred_hand_list, body_mocap.smpl, img_original_bgr.shape
    )

    return body_bbox_list, hand_bbox_list, integral_output_list


def run_frank_mocap(args, bbox_detector, body_mocap, hand_mocap, visualizer):
    # Setup input data to handle different types of inputs
    input_type, input_data, dmap_data = demo_utils.setup_input(args)

    cur_frame = args.start_frame
    video_frame = 0
    while True:
        # load data        o3d_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=200)
        load_bbox = False

        if input_type == "image_dir":
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                dmap_path = dmap_data[cur_frame]
                img_original_bgr = cv2.imread(image_path)
                dmap = cv2.imread(dmap_path, cv2.CV_16UC1)
            else:
                img_original_bgr = None

        elif input_type == "bbox_dir":
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]["image_path"]
                hand_bbox_list = input_data[cur_frame]["hand_bbox_list"]
                body_bbox_list = input_data[cur_frame]["body_bbox_list"]
                img_original_bgr = cv2.imread(image_path)
                load_bbox = True
            else:
                img_original_bgr = None

        elif input_type == "video":
            _, img_original_bgr = input_data.read()
            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)

        elif input_type == "webcam":
            _, img_original_bgr = input_data.read()

            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        else:
            assert False, "Unknown input_type"

        cur_frame += 1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break
        print("--------------------------------------")

        # bbox detection
        if not load_bbox:
            body_bbox_list, hand_bbox_list = list(), list()

        # regression (includes integration)
        body_bbox_list, hand_bbox_list, pred_output_list = run_regress(
            args,
            img_original_bgr,
            body_bbox_list,
            hand_bbox_list,
            bbox_detector,
            body_mocap,
            hand_mocap,
        )

        # save the obtained body & hand bbox to json file
        if args.save_bbox_output:
            demo_utils.save_info_to_json(
                args, image_path, body_bbox_list, hand_bbox_list
            )

        if len(body_bbox_list) < 1:
            print(f"No body deteced: {image_path}")
            continue

        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)
        # TODO: handle multiple meshes
        # Now, just assume there is only one good mesh
        # Furthermore, we're ignoring vertices that are not facing the camera
        vertices = pred_mesh_list[0]["vertices"]
        vertices = vertices[np.where(vertices[:, 2] <= 0)]
        verts_u_float = vertices[:, 0]
        verts_v_float = vertices[:, 1]
        verts_u_int = verts_u_float.astype(int)
        verts_v_int = verts_v_float.astype(int)

        ## Visualize the mesh projected onto the original image
        # plt.imshow(img_original_bgr.astype(np.uint8))
        # plt.plot(verts_u_int, verts_v_int, "r.")
        # plt.show()
        # B4 the 3D reconstruction
        b4_img = img_original_bgr.copy()
        b4_img[verts_v_int, verts_v_int] = (120, 50, 50)  # Red

        # Visualize the pointcloud
        # Plot the Zero Z plane
        zX, zY = np.meshgrid(range(-100, 1000), range(-100, 1000))
        zX = zX.reshape(-1, 1)
        zY = zY.reshape(-1, 1)
        zZ = np.zeros_like(zX)
        zero_z_plane = np.hstack([zX, zY, zZ])
        zero_z_plane = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(zero_z_plane)
        )
        # Origin
        o3d_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=200)
        # Plot only the camera-facing side of the pointcloud
        front_mesh_vertices = open3d.utility.Vector3dVector(vertices)
        front_mesh_pcl = open3d.geometry.PointCloud(front_mesh_vertices)
        # open3d.visualization.draw_geometries([front_mesh_pcl, o3d_frame, zero_z_plane])

        #############################################3
        # Investigate the depth
        breakpoint()
        K_kinect = np.array(
            [[613.095, 0, 636.84], [0, 612.806, 368.059], [0, 0, 1],]
        )  # Obtained from Kinect's factory calib and correspond to 720p images
        print("\n------> Visualize the pointcloud obtained from the depthmap")
        # u, v should be the same as those of the fankmocap
        kinect_verts_u_int = verts_u_int  # integer
        kinect_verts_v_int = verts_v_int  # integer
        kinect_verts_z = dmap[kinect_verts_v_int, kinect_verts_u_int]
        # u, v float values
        kinect_verts_u_float = verts_u_float.reshape([1, -1])
        kinect_verts_v_float = verts_v_float.reshape([1, -1])

        kinect_uv1_points = np.vstack(
            [
                kinect_verts_u_float,
                kinect_verts_v_float,
                np.ones_like(kinect_verts_v_float),
            ]
        )
        kinect_xyz_points = kinect_verts_z * kinect_uv1_points  # 3 x N
        kinect_XYZ_points = np.linalg.inv(K_kinect) @ kinect_xyz_points  # 3 x N
        # Visualize these XYZ points
        kinect_pcl = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(kinect_XYZ_points.T)  # Need to transpose
        )
        # open3d.visualization.draw_geometries([o3d_frame, zero_z_plane, kinect_pcl])

        #######################################3
        # Obtain XYZ w.r.t the camera world from the Frankmocap result
        flength = 613  # Imagine a large focal length
        print(
            f"\n------> Imagine a long focal length {flength}, convert UV1 to XYZ w.r.t the world"
        )
        img_h, img_w = img_original_bgr.shape[:2]
        py, px = img_h / 2.0, img_w / 2.0
        K = np.array([[flength, 0, px], [0, flength, py], [0, 0, 1],])

        verts_X_w = (verts_u_float - img_w / 2) * 1.0 / flength
        verts_Y_w = (verts_v_float - img_h / 2) * 1.0 / flength
        verts_Z_w = (vertices[:, 2] * 1.0 / flength) + 1
        verts_XYZ_w = np.hstack(
            [
                verts_X_w.reshape([-1, 1]),
                verts_Y_w.reshape([-1, 1]),
                verts_Z_w.reshape([-1, 1]),
            ]
        )  # N x 3
        pcl_w = open3d.utility.Vector3dVector(verts_XYZ_w)
        pcl_w = open3d.geometry.PointCloud(pcl_w)
        # open3d.visualization.draw_geometries([pcl_w])

        # breakpoint()
        ## Project back XYZ to the image frame
        # print("------> Project back to the image frame")
        # verts_xyz_cam = K @ verts_XYZ_w.T  # 3 x N
        # verts_uv1_img = verts_xyz_cam / verts_xyz_cam[2, :]  # 3 x N
        # verts_u = verts_uv1_img[0, :].astype(int)
        # verts_v = verts_uv1_img[1, :].astype(int)

        # plt.imshow(img_original_bgr.astype(np.uint8))
        # plt.plot(verts_u, verts_v, "r.")
        # plt.title("Reprojection from XYZ_w to Image")
        # plt.show()

        # after_img = img_original_bgr.copy()
        # after_img[verts_v, verts_u] = (20, 120, 20)

        # blended_img = b4_img * 0.4 + after_img * 0.6
        # plt.imshow(blended_img.astype(np.uint8)[...,::-1])
        # plt.title("Diffence between before and after the 3D reconstruction")
        # plt.show()

        ########################################################
        # Find the correct scale. kinect_pcl = alpha * pcl_w
        # kinect_XYZ_points = alpha * verts_XYZ_w
        breakpoint()
        # alphas = kinect_XYZ_points / verts_XYZ_w.T
        alphas = kinect_XYZ_points[2, :] / verts_XYZ_w[:, 2]
        print(
            "[Info] Alphas stats: \n Mean: ",
            np.mean(alphas),
            "\nStd: ",
            np.std(alphas),
            "| median: ",
            np.median(alphas),
        )
        print("[Info] Examples: ", alphas[:10])
        # Test this alpha by scaling the pcl_w by 1/alpha and project the two pointclouds onto the space
        # scaled_verts_XYZ_w = verts_XYZ_w * np.mean(alphas)
        scaled_verts_XYZ_w = verts_XYZ_w * np.median(alphas)
        scaled_pcl_w = open3d.utility.Vector3dVector(scaled_verts_XYZ_w)
        scaled_pcl_w = open3d.geometry.PointCloud(scaled_pcl_w)

        ## Now, scale the point cloud by some factor, let's say 1000
        # print("\n------> Now, scale the point cloud by some factor, let's say 1000")
        # verts_XYZ_w *= 1000
        # pcl_w = open3d.utility.Vector3dVector(verts_XYZ_w)
        # pcl_w = open3d.geometry.PointCloud(pcl_w)
        # open3d.visualization.draw_geometries([pcl_w, o3d_frame, zero_z_plane])

        print(
            "\n------> Now, I put the scaled pointcloud alongside with the kinect one"
        )
        # Paint colors for pointclouds
        kinect_pcl.paint_uniform_color(
            np.array([0.5, 0.5, 0.5]).reshape([3, 1])
        )  # white
        scaled_pcl_w.paint_uniform_color(
            np.array([0.9, 0.2, 0.3]).reshape([3, 1])
        )  # Red
        pcl_w.paint_uniform_color(np.array([0.2, 0.2, 0.9]).reshape([3, 1]))  # Red
        open3d.visualization.draw_geometries(
            [scaled_pcl_w, o3d_frame, zero_z_plane, kinect_pcl]
            # [o3d_frame, kinect_pcl, pcl_w]
        )

        # breakpoint()
        ## Project back XYZ to the image frame to ensure that the XYZ calculation using the Frankmocap works
        print("------> Project back the scaled PCL to the image frame")
        verts_xyz_cam = K @ scaled_verts_XYZ_w.T  # 3 x N
        verts_uv1_img = verts_xyz_cam / verts_xyz_cam[2, :]  # 3 x N
        verts_u = verts_uv1_img[0, :].astype(int)
        verts_v = verts_uv1_img[1, :].astype(int)
        plt.imshow(img_original_bgr.astype(np.uint8)[..., ::-1])
        plt.plot(verts_u, verts_v, "r.")
        plt.plot(verts_u_int, verts_v_int, "b.")
        plt.legend(["Red: frankmocap reprojection", "Blue: original"])
        plt.title("Reprojection from scaled XYZ_w to Image")
        plt.show()

        print(f"Processed : {image_path}")

    # save images as a video
    if not args.no_video_out and input_type in ["video", "webcam"]:
        demo_utils.gen_video_out(args.out_dir, args.seq_name)

    if input_type == "webcam" and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()


def main():
    args = DemoOptions().parse()
    args.use_smplx = True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert torch.cuda.is_available(), "Current version only supports GPU"

    hand_bbox_detector = HandBboxDetector("third_view", device)

    # Set Mocap regressor
    body_mocap = BodyMocap(
        args.checkpoint_body_smplx, args.smpl_dir, device=device, use_smplx=True
    )
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device=device)

    # Set Visualizer
    if args.renderer_type in ["pytorch3d", "opendr"]:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)

    run_frank_mocap(args, hand_bbox_detector, body_mocap, hand_mocap, visualizer)


if __name__ == "__main__":
    main()
