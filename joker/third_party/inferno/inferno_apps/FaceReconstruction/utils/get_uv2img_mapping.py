import itertools

import numpy as np
from trimesh import Trimesh
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.interpolate import RegularGridInterpolator
import cv2


def get_uv2img_mapping(renderer, K, Rt, depth, uvmap_size, subdivide_render=10, depth_tolerance=1e-3, subdivide_margin=0.1):
    """

    :param mesh:
    :param K:
    :param Rt:
    :param depth:
    :param uvmap_size:
    :param subdivide_render:
    :param depth_tolerance:
    :return:
        uv2img_mapping as np.array uvmapsize x uvmapsize x 3
        first two channels define x and y coordinates in screen space; (0,0) refers to center of top left pixel; (w-1,h-1) to center of bottom-left pixel
        last channel contains visibility map with values either 0. or 1.
    """
    uv_h, uv_w = uvmap_size
    uvgrid_coords_uv = np.stack(np.meshgrid(np.arange(uv_h), np.arange(uv_w)), axis=-1)
    uvgrid_coords_uv = (uvgrid_coords_uv + 0.5) / np.array([[uv_w, uv_h]])  # pixel coordinates H x W x 2 normalized from 0 ... 1 (0 is top left corner of top left uv pixel)
    uvgrid_coords_uv[..., 1] = 1 - uvgrid_coords_uv[..., 1]  # vertical flip
    vert_coords_uv = mesh.visual.uv
    faces = mesh.faces
    vert_coords_world = mesh.vertices
    face_coords_uv = vert_coords_uv[faces]  # nfaces x 3 x 2

    # # visualization
    # fig, ax = plt.subplots()
    # for i in range(len(face_coords_uv)):
    #     quad = face_coords_uv[i]
    #     polygon = Polygon(quad, fill=False, edgecolor=f"C1")
    #     ax.add_patch(polygon)
    # ax.scatter(np.reshape(uvgrid_coords_uv, (-1, 2))[:, 0], np.reshape(uvgrid_coords_uv, (-1, 2))[:, 1], s=.5)
    # plt.show()
    # plt.close()

    uvgrid_bc = np.zeros((uv_h, uv_w, 3), dtype=uvgrid_coords_uv.dtype) - 1
    uvgrid_fidx = np.zeros((uv_h, uv_w), dtype=int) - 1

    # tiled rasterization of uvmap to reduce memory footprint
    tile_vertical_edges = np.linspace(0, uv_h, subdivide_render + 1).astype(int)
    tile_horizontal_edges = np.linspace(0, uv_w, subdivide_render + 1).astype(int)
    for i, j in itertools.product(range(subdivide_render), range(subdivide_render)):
        tile_idxbbx = tile_vertical_edges[i], tile_vertical_edges[i + 1], tile_horizontal_edges[j], tile_horizontal_edges[j + 1]
        tile_height = tile_idxbbx[1] - tile_idxbbx[0]
        tile_width = tile_idxbbx[3] - tile_idxbbx[2]

        # filtering uvgrid_coords_uv and faces within tile
        tile_uvgrid_coords_uv = uvgrid_coords_uv[tile_idxbbx[0]: tile_idxbbx[1], tile_idxbbx[2]: tile_idxbbx[3]]
        tile_uvgrid_coords_uv = np.reshape(tile_uvgrid_coords_uv, (-1, 2))  # N_pts_tile x 2
        tile_limits = np.min(tile_uvgrid_coords_uv, axis=0), np.max(tile_uvgrid_coords_uv, axis=0)
        x_margin = (tile_limits[1][0] - tile_limits[0][0]) * subdivide_margin
        y_margin = (tile_limits[1][1] - tile_limits[0][1]) * subdivide_margin
        tile_face_mask = np.any((tile_limits[0][0] - x_margin <= face_coords_uv[:, :, 0]) & (face_coords_uv[:, :, 0] <= tile_limits[1][0] + x_margin) & (tile_limits[0][1] - y_margin <= face_coords_uv[:, :, 1]) & (face_coords_uv[:, :, 1] <= tile_limits[1][1] + y_margin), axis=1)  # filter out triangles that intersect with uv coordinate bbx
        tile_face_idcs = np.where(tile_face_mask)[0]
        tile_uv_face_coords = face_coords_uv[tile_face_mask]  # N_faces_tile x 3 x 2

        if len(tile_face_idcs) > 0:  # only need to rasterize if any faces in tile at all
            tile_uvgrid_bc, tile_uvgrid_localfidx = twoD_rasterize(tile_uvgrid_coords_uv, tile_uv_face_coords)  # fidx ... face index
            tile_uvgrid_globalfidx = tile_face_idcs[tile_uvgrid_localfidx]
            tile_uvgrid_globalfidx[tile_uvgrid_localfidx == -1] = -1  # no face
            uvgrid_bc[tile_idxbbx[0]:tile_idxbbx[1], tile_idxbbx[2]: tile_idxbbx[3]] = np.reshape(tile_uvgrid_bc, (tile_height, tile_width, 3))
            uvgrid_fidx[tile_idxbbx[0]:tile_idxbbx[1], tile_idxbbx[2]: tile_idxbbx[3]] = np.reshape(tile_uvgrid_globalfidx, (tile_height, tile_width))

            # # visualization of tile
            # fig, ax = plt.subplots()
            # uvgrid_fidx_ = np.zeros_like(uvgrid_fidx) - 1
            # uvgrid_fidx_[tile_idxbbx[0]:tile_idxbbx[1], tile_idxbbx[2]: tile_idxbbx[3]] = np.reshape(tile_uvgrid_localfidx, (tile_height, tile_width))
            # ax.imshow(uvgrid_fidx_ != -1)
            # ax.scatter(tile_uvgrid_coords_uv[:, 0] * uv_w - 0.5, (1 - tile_uvgrid_coords_uv[:, 1]) * uv_h - 0.5, )
            # for i in range(len(tile_uv_face_coords)):
            #     quad = tile_uv_face_coords[i]
            #     quad[:, 1] = 1 - quad[:, 1]
            #     quad = quad * uv_w - 0.5
            #     polygon = Polygon(quad, fill=False, edgecolor=f"C1")
            #     ax.add_patch(polygon)
            # plt.xlim(tile_idxbbx[2], tile_idxbbx[3])
            # plt.ylim(tile_idxbbx[1], tile_idxbbx[0])
            # plt.show()
            # plt.close()

    uvgrid_visibility = uvgrid_fidx != -1

    # projection to screen space
    uvgrid_faces = faces[uvgrid_fidx]  # hgrid x wgrid x 3, idcs of vertices of the face that uvgrid point projects onto
    uvgrid_face_coords_world = vert_coords_world[uvgrid_faces]  # hgrid x wgrid x 3 x 3, world-space vertex coordinates of the face that uvgrid point projects onto
    uvgrid_coords_world = np.einsum("abcd,abc->abd", uvgrid_face_coords_world, uvgrid_bc)  # barycentric-cordinate-based vertex coordinate interpolation; results in shape hgrid x wgrid x 3 with world-space coordinates
    uvgrid_coords_cam = np.einsum("ab,cdb->cda", Rt, np.concatenate([uvgrid_coords_world, np.ones_like(uvgrid_coords_world[..., :1])], axis=-1))
    uvgrid_coords_screen = np.einsum("ab,cdb->cda", K, uvgrid_coords_cam)
    uvgrid_coords_screen = uvgrid_coords_screen[..., :2] / uvgrid_coords_screen[..., 2:]

    # # visualization of uvgrid points in world space
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # vis_pts = np.reshape(uvgrid_coords_world[75:150, 75:150], (-1, 3))
    # ax.scatter(vis_pts[:, 0], vis_pts[:, 1], vis_pts[:,2])
    # plt.show()

    # # visualization of projection of uvgrid to image space
    # fig, ax = plt.subplots()
    # vis_pts_idcs = np.random.permutation(uv_h * uv_w)[:1000]
    # vis_pts = np.reshape(uvgrid_coords_screen, (-1, 2))[vis_pts_idcs]
    # ax.scatter(vis_pts[:, 0], vis_pts[:, 1])
    # ax.imshow(depth, cmap="gray")
    # plt.show()

    # check visibility through depth consistency
    depth_interpolator = RegularGridInterpolator(
        points=(np.arange(depth.shape[0]), np.arange(depth.shape[1])),
        values=depth,
        method="nearest",
        fill_value=0,
        bounds_error=False
    )
    uvgrid_depths = uvgrid_coords_cam[..., -1]
    uvgrid_depths_interp = depth_interpolator(uvgrid_coords_screen[..., ::-1])  # have to swap x,y coordinates during sampling
    uvgrid_visibility = uvgrid_visibility & (np.abs(uvgrid_depths - uvgrid_depths_interp) < depth_tolerance)

    # # visualize depth reprojection error
    # fig, ax = plt.subplots()
    # ax.imshow(np.abs(uvgrid_depths - uvgrid_depths_interp), vmax=depth_tolerance)
    # plt.show()

    # applying visibility
    uvgrid_coords_screen[~uvgrid_visibility] = -1

    uv2img_mapping = np.concatenate([uvgrid_coords_screen, uvgrid_visibility.astype(uvgrid_coords_screen.dtype)[..., None]], axis=-1)
    # first two channels define x and y coordinates in screen space; (0,0) refers to center of top left pixel; (w-1,h-1) to center of bottom-left pixel
    # last channel contains visibility map with values either 0. or 1.
    return uv2img_mapping


def twoD_rasterize(sample_points, triangles):
    npoints = len(sample_points)
    ntriangles = len(triangles)

    if ntriangles == 0:
        bc = np.zeros((npoints, 3), dtype=sample_points.dtype) - 1
        face_idcs = np.zeros(npoints, dtype=int) - 1
    else:
        bc_all = get_barycentric_coordinates(sample_points, triangles)  # npoints x ntriangles x 3
        point_on_face_mask = np.all(bc_all > 0, axis=-1)  # npoints x ntriangles

        # getting face index
        point_on_face_mask_ = np.concatenate((point_on_face_mask, np.ones_like(point_on_face_mask[:, :1])), axis=-1)  # adding one aux face that all points lie on to detect the points that don't lie on any face
        face_idcs = np.argmax(point_on_face_mask_, axis=1)
        no_face_mask = face_idcs == ntriangles  # points that dont lie on any face
        face_idcs[no_face_mask] = -1

        bc = bc_all[np.arange(npoints), face_idcs]
        bc[no_face_mask] = -1

    return bc, face_idcs


def get_barycentric_coordinates(points, triangles):
    """
    gets barycentric coordinates of points wrt triangles
    taken from https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle

    :param points: N_points x 2
    :param triangles: N_triangles x 3 x 2
    :return: barycentric coordinates of shape N_points x N_triangles x 3
    """
    npoints = len(points)
    ntriangles = len(triangles)
    bc = np.zeros((npoints, ntriangles, 3), dtype=points.dtype) - 1

    signed_areas = 0.5 * (-triangles[:, 1, 1] * triangles[:, 2, 0] + triangles[:, 0, 1] * (-triangles[:, 1, 0] + triangles[:, 2, 0]) + triangles[:, 0, 0] * (triangles[:, 1, 1] - triangles[:, 2, 1]) + triangles[:, 1, 0] * triangles[:, 2, 1])

    # filtering out zero-area triangles
    non_zero_area_mask = signed_areas != 0
    triangles_nonzero = triangles[non_zero_area_mask]
    ntriangles_nonzero = len(triangles_nonzero)
    signed_areas_nonzeros = signed_areas[non_zero_area_mask]

    points_expanded = np.broadcast_to(np.expand_dims(points, 1), (npoints, ntriangles_nonzero, 2))
    triangles_nonzero_expanded = np.broadcast_to(np.expand_dims(triangles_nonzero, 0), (npoints, ntriangles_nonzero, 3, 2))
    signed_areas_nonzero_expanded = np.broadcast_to(np.expand_dims(signed_areas_nonzeros, 0), (npoints, ntriangles_nonzero))
    a0 = 1 / (2 * signed_areas_nonzero_expanded) * (triangles_nonzero_expanded[:, :, 0, 1] * triangles_nonzero_expanded[:, :, 2, 0] - triangles_nonzero_expanded[:, :, 0, 0] * triangles_nonzero_expanded[:, :, 2, 1] + (triangles_nonzero_expanded[:, :, 2, 1] - triangles_nonzero_expanded[:, :, 0, 1]) * points_expanded[:, :, 0] + (triangles_nonzero_expanded[:, :, 0, 0] - triangles_nonzero_expanded[:, :, 2, 0]) * points_expanded[:, :, 1])
    a1 = 1 / (2 * signed_areas_nonzero_expanded) * (triangles_nonzero_expanded[:, :, 0, 0] * triangles_nonzero_expanded[:, :, 1, 1] - triangles_nonzero_expanded[:, :, 0, 1] * triangles_nonzero_expanded[:, :, 1, 0] + (triangles_nonzero_expanded[:, :, 0, 1] - triangles_nonzero_expanded[:, :, 1, 1]) * points_expanded[:, :, 0] + (triangles_nonzero_expanded[:, :, 1, 0] - triangles_nonzero_expanded[:, :, 0, 0]) * points_expanded[:, :, 1])

    bc[:, non_zero_area_mask, 0] = a0
    bc[:, non_zero_area_mask, 1] = a1
    bc[:, non_zero_area_mask, 2] = 1 - a0 - a1

    return bc


def check_uv_uvinv(uvpath, uvinv_path):
    uv = cv2.imread(str(uvpath)).astype(float)[..., ::-1] / 255
    uvinv = cv2.imread(str(uvinv_path)).astype(float)[..., ::-1] / 255

    h_uv, w_uv = uv.shape[:2]
    h_uvinv, w_uvinv = uvinv.shape[:2]
    pts_x_uv = (np.arange(w_uv) + 0.5) / w_uv
    pts_y_uv = (np.arange(h_uv) + 0.5) / h_uv
    pts_x_uvinv = (np.arange(w_uvinv) + 0.5) / w_uvinv
    pts_y_uvinv = (np.arange(h_uvinv) + 0.5) / h_uvinv
    grid_uv = np.stack(np.meshgrid(pts_x_uv, pts_y_uv), axis=-1)
    grid_uv = np.concatenate([grid_uv, np.ones_like(grid_uv[..., :1])], axis=-1)
    grid_uvinv = np.stack(np.meshgrid(pts_x_uvinv, pts_y_uvinv), axis=-1)
    grid_uvinv = np.concatenate([grid_uvinv, np.ones_like(grid_uvinv[..., :1])], axis=-1)

    uv_interp = RegularGridInterpolator(points=(pts_y_uv, pts_x_uv),
                                        values=uv, bounds_error=False, fill_value=0)
    uvinv_interp = RegularGridInterpolator(points=(pts_y_uvinv, pts_x_uvinv),
                                           values=uvinv, bounds_error=False, fill_value=0)

    # sampling each other should always yield the coordinate grid!
    pred1 = uv_interp(uvinv[..., [1, 0]])
    pred2 = uvinv_interp(uv[..., [1, 0]])
    pred1_diff = pred1 - grid_uvinv
    pred2_diff = pred2 - grid_uv
    pred1_diff[pred1[..., 0] == 0] = 0  # mask out background
    pred2_diff[pred2[..., 0] == 0] = 0

    fig, axes = plt.subplots(ncols=3, nrows=3)
    axes[0, 0].imshow(pred1)
    axes[0, 1].imshow(pred2)
    axes[0, 2].imshow(grid_uv)
    axes[1, 0].imshow(pred1_diff[..., 0])
    axes[1, 1].imshow(pred1_diff[..., 1])
    axes[1, 2].imshow(pred1_diff[..., 2])
    axes[2, 0].imshow(pred2_diff[..., 0])
    axes[2, 1].imshow(pred2_diff[..., 1])
    axes[2, 2].imshow(pred2_diff[..., 2])
    plt.show()


if __name__ == "__main__":
    check_uv_uvinv("/tmp/panohead10/0000/sequences/STATIC/frame_00000/flame_uvs-512/cam_02.png",
                   "/tmp/panohead10/0000/sequences/STATIC/frame_00000/flame_uvinvs-512/cam_02.png")
    points = np.array([[0.6, 0.6]])
    triangles = np.array([[[0, 0], [1, 0], [0, 1]], [[1, 1], [1, 0], [0, 1]]])
    bc = get_barycentric_coordinates(points, triangles)
    print(bc)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots()
    for i in range(len(triangles)):
        quad = triangles[i]
        polygon = Polygon(quad, fill=False, edgecolor=f"C{i}")
        ax.add_patch(polygon)
    ax.scatter(points[:, 0], points[:, 1])
    plt.show()
    plt.close()
