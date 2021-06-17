"""
Extract joints predicted by Frankmocap alongside with their depth from Kinect
"""
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

    # DEBUG: what is this integeral_output_list
    # integration by copy-and-paste
    integral_output_list = integration_copy_paste(
        pred_body_list, pred_hand_list, body_mocap.smpl, img_original_bgr.shape
    )

    return body_bbox_list, hand_bbox_list, integral_output_list


def run_frank_mocap(args, bbox_detector, body_mocap, hand_mocap, visualizer):
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
    o3d_vis = open3d.visualization.Visualizer()
    o3d_vis.create_window()
    o3d_vis.add_geometry(o3d_frame)
    o3d_vis.add_geometry(zero_z_plane)

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
        cur_frame_id = image_path.split("/")[-1].split(".")[0]  # name of the image file

        if img_original_bgr is None or cur_frame > args.end_frame:
            break
        print("--------------------------------------")
        print(f"[Debug] SOurce Img: {image_path}")

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
        # Save meshes
        breakpoint()
        raw_mesh_path = os.path.join(args.out_dir, cur_frame_id + ".obj")
        raw_vertices = open3d.utility.Vector3dVector(pred_mesh_list[0]["vertices"])
        raw_faces = open3d.utility.Vector3iVector(pred_mesh_list[0]["faces"])
        raw_mesh = open3d.geometry.TriangleMesh(raw_vertices, raw_faces)
        open3d.io.write_triangle_mesh(raw_mesh_path, raw_mesh)
        print(f"[Info] Written raw mesh to file {raw_mesh_path}")

        # TODO: handle multiple meshes
        # Now, just assume there is only one good mesh
        # Furthermore, we're ignoring vertices that are not facing the camera
        vertices = pred_mesh_list[0]["vertices"].copy()
        vertices = vertices[np.where(vertices[:, 2] <= 0)]
        verts_u_float = vertices[:, 0]
        verts_v_float = vertices[:, 1]

        # TODO: debug this step. why verts_u_int can be out of the image
        # For now, just simply prevent this happen
        verts_u_float[np.where(verts_u_float >= dmap.shape[1])] = dmap.shape[1] - 1
        verts_v_float[np.where(verts_v_float >= dmap.shape[0])] = dmap.shape[0] - 1

        verts_u_int = verts_u_float.astype(int)
        verts_v_int = verts_v_float.astype(int)

        ## Visualize the mesh projected onto the original image
        # plt.imshow(img_original_bgr.astype(np.uint8))
        # plt.plot(verts_u_int, verts_v_int, "r.")
        # plt.show()
        # B4 the 3D reconstruction
        b4_img = img_original_bgr.copy()
        b4_img[verts_v_int, verts_v_int] = (120, 50, 50)  # Red

        # Plot only the camera-facing side of the pointcloud
        front_mesh_vertices = open3d.utility.Vector3dVector(vertices)
        front_mesh_pcl = open3d.geometry.PointCloud(front_mesh_vertices)
        # open3d.visualization.draw_geometries([front_mesh_pcl, o3d_frame, zero_z_plane])

        #############################################3
        # Investigate the depth
        K_kinect = np.array(
            [[611.99, 0, 612.067], [0, 637.85, 372.33], [0, 0, 1],]
        )  # Obtained from Kinect's factory calib and correspond to 720p images
        print("\n------> Visualize the pointcloud obtained from the depthmap")
        # u, v should be the same as those of the fankmocap
        kinect_verts_u_int = verts_u_int.copy()  # integer
        kinect_verts_v_int = verts_v_int.copy()  # integer
        kinect_verts_z = dmap[kinect_verts_v_int, kinect_verts_u_int]
        # u, v float values
        kinect_verts_u_float = verts_u_float.copy().reshape([1, -1])
        kinect_verts_v_float = verts_v_float.copy().reshape([1, -1])

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
        kinect_pcl_ = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(kinect_XYZ_points.T)  # Need to transpose
        )
        print(f"[Debug] Kinect_pct: {np.array(kinect_pcl_.points)[:2]}")
        if cur_frame > args.start_frame + 1:
            print(f"[Debug] cur_frame > start_frame. Updating the pcl")
            kinect_pcl.points = kinect_pcl_.points
            o3d_vis.update_geometry(kinect_pcl)
            o3d_vis.update_renderer()
            o3d_vis.poll_events()
        elif cur_frame == args.start_frame + 1:
            print(f"[Debug] cur_frame == start_frame. Just add the pcl")
            kinect_pcl = kinect_pcl_
            o3d_vis.add_geometry(kinect_pcl)

        o3d_vis.capture_screen_image("tmp/temp_%s.jpg" % cur_frame_id)

        # Save this kinect point cloud
        kinect_pcl_path = os.path.join(args.out_dir, cur_frame_id + ".pcd")
        open3d.io.write_point_cloud(kinect_pcl_path, kinect_pcl)
        print(f"[Info] Written kinect pcl to file {kinect_pcl_path}")
        print(f"Processed : {image_path}")
        # TODO: remove
        # cur_frame += 40

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
    visualizer = Visualizer(args.renderer_type)

    run_frank_mocap(args, hand_bbox_detector, body_mocap, hand_mocap, visualizer)


if __name__ == "__main__":
    main()
