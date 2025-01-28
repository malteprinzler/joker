from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--cam_dir', type=Path, default='data/DISTILLATION_EXAMPLE_DATA/000/sequences/CAM_SWEEP/frame_00000/cam-512')
parser.add_argument('--pred_dir', type=Path, default='experiments/distillation/example_data/example_data_000_nerfstudio/save/it020000-test')
parser.add_argument('--out_file', type=Path, default='experiments/distillation/example_data/example_data_000_nerfstudio/nerfstudio_ds.json')
args = parser.parse_args()
cam_dir = args.cam_dir
pred_dir = args.pred_dir
out_file = args.out_file

out_file.parent.mkdir(exist_ok=True, parents=True)
out_dict = {
    "camera_model": "OPENCV",  # camera model type [OPENCV, OPENCV_FISHEYE]
    "frames": []

}
for img_path in [p for p in sorted(pred_dir.iterdir()) if p.name.endswith('.png')]:
    stem = img_path.stem
    cam_path = cam_dir / f'cam_{stem}.json'
    with open(cam_path, 'r') as f:
        cam_params = json.load(f)
    intrinsics = np.array(cam_params['intrinsics'])[:3,:3]  # 3x3
    extrinsics = np.array(cam_params['extrinsics'])  # 4 x 4
    img = Image.open(img_path)

    # converting camera intrinsics:
    # joker intrinsics assume +x: right, +y: down, origin at top-left corner of top-left pixel
    # nerfstudio instrinsics assume +x: right, +y: up, origin at center of bottom-left pixel  https:#docs.nerf.studio/quickstart/data_conventions.html
    intrinsics_ns = intrinsics.copy()
    intrinsics_ns[1, 2] = img.height - intrinsics[1, 2]  # flipping y axis from +y: down to +y: up
    intrinsics_ns[:2, -1] -= 0.5  # going from origin at corner of pixel to origin at center of pixel

    # converting camera extrinsics
    # joker extrinsics assume: +x = right, +y = down, +z = look-at
    # nerfstudio extrinsics assume +x = right, +y = up, -z = look-at  https:#docs.nerf.studio/quickstart/data_conventions.html
    joker_2_ns = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]])
    extrinsics_ns = joker_2_ns @ extrinsics

    # converting world
    # joker world assumes: +x = left face side, +y = up, +z = face front
    # nerfstudio assumes: x = left, -y = look-at , +z = up
    world_rotate = np.array([
        [-1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.]])
    extrinsics_ns = extrinsics_ns @ world_rotate

    out_dict['frames'].append({
      "fl_x": intrinsics_ns[0,0], # focal length x
      "fl_y": intrinsics_ns[1,1], # focal length y
      "cx": intrinsics_ns[0,2], # principal point x
      "cy": intrinsics_ns[1,2], # principal point y
      "w": img.width, # image width
      "h": img.height, # image height
      "file_path": str(img_path.absolute()),
        'transform_matrix': np.linalg.inv(extrinsics_ns).tolist(),
    })

with open(out_file, 'w') as f:
    json.dump(out_dict, f, indent='\t')

print("Nerfstudio dataset successfully written to {}".format(out_file))

    

    

