# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
import os.path as osp
from matplotlib import scale
from numpy.core.fromnumeric import sort

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
        # only keep a single bbox - the largest one if args.single_person is set
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

    #############################################3
    K_kinect = np.array(
        [[613.095, 0, 636.84], [0, 612.806, 368.059], [0, 0, 1],]
    )  # Obtained from Kinect's factory calib and correspond to 720p images

    # Debug
    # Non-blocking visualization
    o3d_vis = open3d.visualization.Visualizer()
    o3d_vis.create_window()
    # Change view angle (http://www.open3d.org/docs/0.9.0/tutorial/Advanced/customized_visualization.html)
    o3d_vis_ctr = o3d_vis.get_view_control()
    pi = 3.14159265359
    o3d_vis_ctr.rotate(0, pi)

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
        cur_frame_id = image_path.split("/")[-1].split(".")[0]  # name of the image file

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

        # Get an image + Dmap
        # res_dmap: np.uint16 # H x W
        res_img, res_dmap, res_mesh = visualizer.visualize(
            img_original_bgr,
            pred_mesh_list=pred_mesh_list,
            body_bbox_list=body_bbox_list,
            hand_bbox_list=hand_bbox_list,
        )

        ## show result in the screen
        # if not args.no_display:
        #    # Overlay depth
        #    res_dmap_8bit = res_dmap / res_dmap.max() * 255
        #    res_dmap_8bit = res_dmap.astype(np.uint8)
        #    # plt.imshow(res_dmap_8bit)
        #    # plt.show()
        #    dmap_overlay = (
        #        cv2.applyColorMap(res_dmap_8bit, cv2.COLORMAP_JET) * 0.7
        #        + 0.3 * img_original_bgr
        #    )
        #    res_img = np.hstack([res_img, dmap_overlay])
        #    res_img = res_img.astype(np.uint8)
        #    # Resize
        #    sh, sw = res_img.shape[:2]
        #    sh, sw = int(0.4 * sh), int(0.4 * sw)
        #    res_img = cv2.resize(res_img, (sw, sh))
        #    ImShow(res_img)

        # Get the depth scale by comparing the frankmocap dmap with that of the kinect
        # Get indices
        dmap[np.where(res_dmap <= 0)] = 0  # Consider only points on the person
        valid_dmap_indices = np.where(res_dmap * dmap > 0)
        valid_dmap = dmap[valid_dmap_indices]
        valid_res_dmap = res_dmap[valid_dmap_indices]
        # Reshift and rescale valid_dmap
        reshifted_res_dmap = valid_res_dmap - 5
        reshifted_res_dmap *= 112  # TODO:  for now, not consider *(-1)

        # Now, transform this dmap to 3D using an orthographic camera model
        valid_dmap_u = valid_dmap_indices[1]
        valid_dmap_v = valid_dmap_indices[0]  # (133898, )

        # Obtain XYZ w.r.t the camera world from the Frankmocap result
        flength = 613  # Imagine a large focal length
        print(
            f"\n------> Imagine a long focal length {flength}, convert UV1 to XYZ w.r.t the world"
        )
        img_h, img_w = img_original_bgr.shape[:2]
        py, px = img_h / 2.0, img_w / 2.0
        K = np.array([[flength, 0, px], [0, flength, py], [0, 0, 1],])

        res_dmap_X = (valid_dmap_u - img_w / 2) * 1.0 / flength
        res_dmap_Y = (valid_dmap_v - img_w / 2) * 1.0 / flength
        res_dmap_Z = (
            reshifted_res_dmap * 1.0 / flength + 1  # + 112 * 5.0 / flength
        )  # 112 and 5 are numbers used to shift and scale Z values in p3d_renderer.py
        res_dmap_XYZ = np.hstack(
            [
                res_dmap_X.reshape([-1, 1]),
                res_dmap_Y.reshape([-1, 1]),
                res_dmap_Z.reshape([-1, 1]),
            ]
        )  # N x 3
        # Obtain XYZ w.r.t the camera world from the Kinect depth map
        kinect_dmap_uv1 = np.vstack(
            [valid_dmap_u, valid_dmap_v, np.ones_like(valid_dmap_v)]
        )
        kinect_dmap_xyz = valid_dmap * kinect_dmap_uv1  # 3 x N
        kinect_dmap_XYZ = np.linalg.inv(K_kinect) @ kinect_dmap_xyz  # 3 x N
        # Get scale
        scales = kinect_dmap_XYZ[2, :] / res_dmap_XYZ[:, 2]
        print(f"[Info] Scales stats: ")
        print(
            f"[Info] Mean: {np.mean(scales)}| Std: {np.std(scales)}| Median: {np.median(scales)}"
        )
        dmap_scale = np.median(scales)

        # Debug: show depth errors using the scale obtained directly from depth map scaling
        # dmap_scale = np.median(valid_dmap/valid_res_dmap)
        # scaled_res_dmap = res_dmap * dmap_scale
        # scaled_dmap_errors = scaled_res_dmap - dmap  # Error image
        # print(
        #    f"[Info] Dmap Error: {np.mean(scaled_dmap_errors)}| Std: {np.std(scaled_dmap_errors)}| Median: {np.median(scaled_dmap_errors)}"
        # )
        # res_dmap_8bit = res_dmap / res_dmap.max() * 255
        # res_dmap_8bit = cv2.applyColorMap(res_dmap.astype(np.uint8), cv2.COLORMAP_JET)
        # kinect_dmap_8bit = dmap / dmap.max() * 255
        # kinect_dmap_8bit = cv2.applyColorMap(dmap.astype(np.uint8), cv2.COLORMAP_JET)
        # scaled_res_dmap_8bit = scaled_res_dmap / scaled_res_dmap.max() * 255
        # scaled_res_dmap_8bit = cv2.applyColorMap(
        #    scaled_res_dmap.astype(np.uint8), cv2.COLORMAP_JET
        # )
        # scaled_dmap_errors = np.abs(scaled_dmap_errors)
        # scaled_dmap_errors_8bit = scaled_dmap_errors / scaled_dmap_errors.max() * 255
        # scaled_dmap_errors_8bit = cv2.applyColorMap(
        #    scaled_dmap_errors_8bit.astype(np.uint8), cv2.COLORMAP_JET
        # )

        # dmap_overlay = np.hstack(
        #    [
        #        kinect_dmap_8bit,
        #        res_dmap_8bit,
        #        scaled_res_dmap_8bit,
        #        scaled_dmap_errors_8bit,
        #    ]
        # )
        # sh, sw = dmap_overlay.shape[:2]
        # sh, sw = int(0.4 * sh), int(0.4 * sw)
        # dmap_overlay = cv2.resize(dmap_overlay, (sw, sh))
        # ImShow(dmap_overlay)
        # plt.imshow(dmap_overlay)
        # plt.title("[1] Kinect vs [2] Res vs [3] Scaled Res vs [4] Error")
        # cmap = plt.cm.get_cmap("jet")
        # plt.colorbar(cmap=cmap)
        # plt.axis("off")
        # plt.show()

        # TODO: handle multiple meshes
        # Now, just assume there is only one good mesh
        # Furthermore, we're ignoring vertices that are not facing the camera

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
        o3d_vis.add_geometry(o3d_frame)
        o3d_vis.add_geometry(zero_z_plane)
        # open3d.visualization.draw_geometries([front_mesh_pcl, o3d_frame, zero_z_plane])

        #######################################3
        # Obtain XYZ w.r.t the camera world from the Frankmocap result
        vertices = pred_mesh_list[0]["vertices"]
        faces = pred_mesh_list[0]["faces"]
        verts_u_float = vertices[:, 0]
        verts_v_float = vertices[:, 1]
        verts_z_float = vertices[:, 2]

        verts_uv1_float = np.vstack(
            [verts_u_float, verts_v_float, np.ones_like(verts_u_float)]
        )

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
        verts_Z_w = (
            vertices[:, 2] * 1.0 / flength + 1  # 112 * 5.0 / flength
        )  # 112 and 5 are numbers used to shift and scale Z values in p3d_renderer.py
        verts_XYZ_w = np.hstack(
            [
                verts_X_w.reshape([-1, 1]),
                verts_Y_w.reshape([-1, 1]),
                verts_Z_w.reshape([-1, 1]),
            ]
        )  # N x 3
        # Scale up pcl
        scaled_verts_XYZ_w = verts_XYZ_w * dmap_scale  # * flength / 112
        scaled_verts = open3d.utility.Vector3dVector(scaled_verts_XYZ_w)
        scaled_faces = open3d.utility.Vector3iVector(faces)
        scaled_pcl_w = open3d.geometry.PointCloud(scaled_verts)
        scaled_mesh = open3d.geometry.TriangleMesh(scaled_verts, scaled_faces)
        mesh_path = os.path.join(args.out_dir, cur_frame_id + ".obj")
        open3d.io.write_triangle_mesh(mesh_path, scaled_mesh)
        print(f"[Info] Written raw mesh to file {mesh_path}")

        ## Project back XYZ to the image frame
        # print("------> Project back to the image frame")
        # verts_xyz_cam = K_kinect @ XYZ_points  # 3 x N
        # verts_uv1_img = verts_xyz_cam / verts_xyz_cam[2, :]  # 3 x N
        # verts_u = verts_uv1_img[0, :].astype(int)
        # verts_v = verts_uv1_img[1, :].astype(int)

        # plt.imshow(img_original_bgr.astype(np.uint8))
        # plt.plot(verts_u, verts_v, "r.")
        # plt.title("Reprojection from XYZ_w to Image")
        # plt.show()

        ########################################################
        print(
            "\n------> Now, I put the scaled pointcloud alongside with the kinect one"
        )

        # Debug: display two PCL onto the space

        # Get Kinect 3D points
        kinect_verts_u_int = verts_u_float.astype(int)  # integer
        kinect_verts_v_int = verts_v_float.astype(int)  # integer
        # Make sure nothing is out of bound
        kinect_verts_v_int[np.where(kinect_verts_v_int >= dmap.shape[0])] = (
            dmap.shape[0] - 1
        )
        kinect_verts_u_int[np.where(kinect_verts_v_int >= dmap.shape[1])] = (
            dmap.shape[1] - 1
        )
        kinect_verts_z = dmap[kinect_verts_v_int, kinect_verts_u_int]

        kinect_uv1_points = np.vstack(
            [kinect_verts_u_int, kinect_verts_v_int, np.ones_like(kinect_verts_v_int),]
        )
        kinect_xyz_points = kinect_verts_z * kinect_uv1_points  # 3 x N
        kinect_XYZ_points = np.linalg.inv(K_kinect) @ kinect_xyz_points  # 3 x N
        # Visualize these XYZ points
        kinect_pcl = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(kinect_XYZ_points.T)  # Need to transpose
        )
        # Paint colors for pointclouds
        kinect_pcl.paint_uniform_color(
            np.array([0.5, 0.5, 0.5]).reshape([3, 1])
        )  # white
        # pcl_w.paint_uniform_color(np.array([0.9, 0.2, 0.3]).reshape([3, 1]))  # Red

        if cur_frame == args.start_frame + 1:
            scaled_pcl_w_ = scaled_pcl_w
            scaled_mesh_ = scaled_mesh
            kinect_pcl_ = kinect_pcl
            # o3d_vis.add_geometry(scaled_pcl_w_)
            o3d_vis.add_geometry(kinect_pcl_)
            o3d_vis.add_geometry(scaled_mesh_)
        else:
            scaled_mesh_.vertices = scaled_mesh.vertices
            scaled_pcl_w_.points = scaled_pcl_w.points
            kinect_pcl_.points = kinect_pcl.points
            # o3d_vis.update_geometry(scaled_pcl_w_)
            o3d_vis.update_geometry(scaled_mesh_)
            o3d_vis.update_geometry(kinect_pcl_)
            o3d_vis.poll_events()
            o3d_vis.update_renderer()

        # breakpoint()
        ## Project back XYZ to the image frame to ensure that the XYZ calculation using the Frankmocap works
        print("------> Project back the scaled PCL to the image frame")
        verts_xyz_cam = K @ scaled_verts_XYZ_w.T  # 3 x N
        verts_uv1_img = verts_xyz_cam / verts_xyz_cam[2, :]  # 3 x N
        verts_u = verts_uv1_img[0, :].astype(int)
        verts_v = verts_uv1_img[1, :].astype(int)
        plt.imshow(img_original_bgr.astype(np.uint8)[..., ::-1])
        plt.plot(verts_u, verts_v, "r.")
        plt.plot(verts_u_float.astype(int), verts_v_float.astype(int), "b.")
        plt.legend(["Red: frankmocap reprojection", "Blue: original"])
        plt.title("Reprojection from scaled XYZ_w to Image")
        plt.axis("off")
        plt.show()

        print(f"Processed : {image_path}")

    # save images as a video
    if not args.no_video_out and input_type in ["video", "webcam"]:
        demo_utils.gen_video_out(args.out_dir, args.seq_name)

    if input_type == "webcam" and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()
    o3d_vis.destroy_window()


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
    visualizer = Visualizer(args.renderer_type, is_get_dmap=True, is_get_all=True)

    run_frank_mocap(args, hand_bbox_detector, body_mocap, hand_mocap, visualizer)


if __name__ == "__main__":
    main()
