# Copyright (c) Facebook, Inc. and its affiliates.

# Part of code is modified from https://github.com/facebookresearch/pytorch3d

import cv2
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
from pytorch3d.renderer import (
    PerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    BlendParams,
    MeshRasterizer,
    SoftPhongShader,
)


class Pytorch3dRenderer(object):
    def __init__(self, img_size, mesh_color, is_get_dmap=False, is_get_all=False):
        self.is_get_dmap = is_get_dmap  # return depth map?
        self.is_get_all = is_get_all  # return depth map, pytorch3d mesh?
        self.device = torch.device("cuda:0")
        # self.render_size = 1920

        self.img_size = img_size

        # mesh color
        mesh_color = np.array(mesh_color)[::-1]
        self.mesh_color = (
            torch.from_numpy(mesh_color.copy()).view(1, 1, 3).float().to(self.device)
        )

        # renderer for large objects, such as whole body.
        self.render_size_large = 700
        lights = PointLights(
            ambient_color=[[1.0, 1.0, 1.0],],
            diffuse_color=[[1.0, 1.0, 1.0],],
            device=self.device,
            location=[[1.0, 1.0, -30]],
        )
        self.renderer_large = self.__get_renderer(self.render_size_large, lights)

        # renderer for small objects, such as whole body.
        self.render_size_medium = 400
        lights = PointLights(
            ambient_color=[[0.5, 0.5, 0.5],],
            diffuse_color=[[0.5, 0.5, 0.5],],
            device=self.device,
            location=[[1.0, 1.0, -30]],
        )
        self.renderer_medium = self.__get_renderer(self.render_size_medium, lights)

        # renderer for small objects, such as whole body.
        self.render_size_small = 200
        lights = PointLights(
            ambient_color=[[0.5, 0.5, 0.5],],
            diffuse_color=[[0.5, 0.5, 0.5],],
            device=self.device,
            location=[[1.0, 1.0, -30]],
        )
        self.renderer_small = self.__get_renderer(self.render_size_small, lights)

    def __get_renderer(self, render_size, lights):
        self.min_z = 0.1  # same as znear used in the cameras model
        cameras = FoVOrthographicCameras(
            device=self.device,
            znear=0.1,
            zfar=10.0,
            max_y=1.0,
            min_y=-1.0,
            max_x=1.0,
            min_x=-1.0,
            scale_xyz=((1.0, 1.0, 1.0),),  # (1, 3)
        )

        ## Debug:
        ## Computer proj matrix
        # """
        # array([[[ 1.        ,  0.        ,  0.        , -0.        ],
        # [ 0.        ,  1.        ,  0.        , -0.        ],
        # [ 0.        ,  0.        ,  0.1010101 , -0.01010101],
        # [ 0.        ,  0.        ,  0.        ,  1.        ]]],
        #    dtype=float32)

        # """
        # proj_mat = cameras.compute_projection_matrix(
        #    znear=0.1,
        #    zfar=10.0,
        #    max_x=1.0,
        #    min_x=-1.0,
        #    max_y=1.0,
        #    min_y=-1.0,
        #    scale_xyz=torch.tensor([[1.0, 1.0, 1.0]]),
        # )
        # print(f"[Info] Orthographic cam proj mat: \n {proj_mat}")

        raster_settings = RasterizationSettings(
            image_size=render_size, blur_radius=0, faces_per_pixel=1,
        )
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0, 0, 0))

        self.rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings
        )
        renderer = MeshRenderer(
            rasterizer=(self.rasterizer),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_params,
            ),
        )

        return renderer

    def render(self, verts, faces, bg_img):
        """
        @inputs:
            bg_img: 1920 canvas
        """
        orig_verts = verts
        verts = verts.copy()
        faces = faces.copy()

        # bbox for verts
        x0 = int(np.min(verts[:, 0]))
        x1 = int(np.max(verts[:, 0]))
        y0 = int(np.min(verts[:, 1]))
        y1 = int(np.max(verts[:, 1]))
        width = x1 - x0
        height = y1 - y0

        bbox_size = max(height, width)
        if bbox_size <= self.render_size_small:
            # print("Using small size renderer")
            render_size = self.render_size_small
            renderer = self.renderer_small
        else:
            if bbox_size <= self.render_size_medium:
                # print("Using medium size renderer")
                render_size = self.render_size_medium
                renderer = self.renderer_medium
            else:
                # print("Using large size renderer")
                render_size = self.render_size_large
                renderer = self.renderer_large

        # padding the tight bbox
        margin = int(max(width, height) * 0.1)
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(self.img_size, x1 + margin)
        y1 = min(self.img_size, y1 + margin)

        print("old X0: ", x0)
        print("old verts: ", verts[:4])
        print("Margin: ", margin)

        # move verts to be in the bbox
        verts[:, 0] -= x0
        verts[:, 1] -= y0

        # Test
        bbox_crop = bg_img[y0:y1, x0:x1].astype(np.uint8)
        verts_x = verts[:, 0].astype(int)
        verts_y = verts[:, 1].astype(int)
        # bbox_crop[verts_x, verts_y] = (220, 120, 120)
        # plt.imshow(bbox_crop)
        # plt.plot(verts_x, verts_y, "r.")
        # plt.show()

        # normalize verts to (-1, 1)
        bbox_size = max(y1 - y0, x1 - x0)
        half_size = bbox_size / 2
        verts[:, 0] = (verts[:, 0] - half_size) / half_size
        verts[:, 1] = (verts[:, 1] - half_size) / half_size

        # the coords of pytorch-3d is (1, 1) for upper-left and (-1, -1) for lower-right
        # so need to multiple minus for vertices
        verts[:, :2] *= -1

        # STEP 2
        # shift verts along the z-axis
        verts[:, 2] /= 112
        verts[:, 2] += 5
        verts_tensor = torch.from_numpy(verts).float().unsqueeze(0).cuda()
        faces_tensor = torch.from_numpy(faces.copy()).long().unsqueeze(0).cuda()

        # set color
        mesh_color = self.mesh_color.repeat(1, verts.shape[0], 1)
        textures = Textures(verts_rgb=mesh_color)

        # rendering mesh
        mesh = Meshes(
            verts=verts_tensor, faces=faces_tensor, textures=textures
        )  # vertices here have X, Y in [-1, 1]

        ########################
        # DEBUG
        # Plot the Zero Z plane
        # zX, zY = np.meshgrid(range(-100, 1000), range(-100, 1000))
        # zX = zX.reshape(-1, 1)
        # zY = zY.reshape(-1, 1)
        # zZ = np.zeros_like(zX)
        # zero_z_plane = np.hstack([zX, zY, zZ])
        # zero_z_plane = open3d.geometry.PointCloud(
        #    open3d.utility.Vector3dVector(zero_z_plane)
        # )
        ## Convert pytorch3d meshes to open3d pcl
        # o3d_verts = mesh.verts_list()[0].cpu().numpy()
        # o3d_verts = open3d.utility.Vector3dVector(o3d_verts)
        # o3d_pcl = open3d.geometry.PointCloud(o3d_verts)
        ## Origin
        # o3d_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=200)
        # open3d.visualization.draw_geometries([o3d_frame, zero_z_plane, o3d_pcl])

        # rendering depth map size 200 x 200, float 32 bit
        mesh_dmap = self.rasterizer(mesh).zbuf.cpu().squeeze().numpy()
        mesh_dmap[np.where(mesh_dmap < self.min_z)] = 0  # TODO: uncomment this

        # DONOT CONVERT
        # Convert to uint16
        # mesh_dmap = mesh_dmap / mesh_dmap.max() * 65535
        # mesh_dmap = mesh_dmap.astype(np.uint16)

        # mesh_dmap = mesh_dmap / mesh_dmap.max() * 255
        # mesh_dmap = mesh_dmap.astype(np.uint8)
        # import matplotlib.pyplot as plt

        # plt.imshow(mesh_dmap)
        # plt.title("Rended Dmap")
        # plt.show()

        # blending rendered mesh with background image
        rend_img = renderer(mesh)
        rend_img = rend_img[0].cpu().numpy()

        rend_img = rend_img / (rend_img.max()) * 255
        rend_img = rend_img.astype(np.uint8)
        import matplotlib.pyplot as plt

        # plt.imshow(rend_img)
        # plt.title("Rended Image")
        # plt.show()

        # Render size 700
        scale_ratio = render_size / bbox_size
        img_size_new = int(self.img_size * scale_ratio)
        bg_img_new = cv2.resize(bg_img, (img_size_new, img_size_new))

        x0 = max(int(x0 * scale_ratio), 0)
        y0 = max(int(y0 * scale_ratio), 0)
        x1 = min(int(x1 * scale_ratio), img_size_new)
        y1 = min(int(y1 * scale_ratio), img_size_new)

        h0 = min(y1 - y0, render_size)
        w0 = min(x1 - x0, render_size)

        y1 = y0 + h0
        x1 = x0 + w0
        rend_img_new = np.zeros((img_size_new, img_size_new, 4))
        rend_img_new[y0:y1, x0:x1, :] = rend_img[:h0, :w0, :]
        rend_img = rend_img_new

        alpha = rend_img[:, :, 3:4]
        alpha[alpha > 0] = 1.0

        rend_img = rend_img[:, :, :3]
        maxColor = rend_img.max()
        rend_img *= 255 / maxColor  # Make sure <1.0
        rend_img = rend_img[:, :, ::-1]

        res_img = alpha * rend_img + (1.0 - alpha) * bg_img_new

        #################
        # Get the depth map image of the same size
        mesh_dmap = cv2.resize(mesh_dmap, (render_size, render_size))
        rend_dmap_new = np.zeros((img_size_new, img_size_new, 1))
        rend_dmap_new[y0:y1, x0:x1, :] = mesh_dmap[:h0, :w0][..., None]
        mesh_dmap = rend_dmap_new

        import matplotlib.pyplot as plt

        # plt.imshow(mesh_dmap)
        # plt.title("Mesh Depth Map after Resizing")
        # plt.show()

        # Debug: Show res image and depth image at the same time
        # breakpoint()
        # depth_8bit = mesh_dmap / mesh_dmap.max() * 255
        # depth_8bit = depth_8bit.astype(np.uint8)
        # depth_8bit = np.concatenate([depth_8bit, depth_8bit, depth_8bit], -1)
        # depth_vs_img = np.hstack([res_img.astype(np.uint8)[..., ::-1], depth_8bit])
        # plt.imshow(depth_vs_img)
        # plt.title("Depth vs Res Img")
        # plt.show()

        res_img = cv2.resize(res_img, (self.img_size, self.img_size))
        mesh_dmap = cv2.resize(mesh_dmap, (self.img_size, self.img_size))
        if self.is_get_all:
            return res_img, mesh_dmap, mesh
        elif self.is_get_dmap:
            return res_img, mesh_dmap
        else:
            return res_img
