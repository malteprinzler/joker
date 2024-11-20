from pathlib import Path
import glob
import tqdm
from argparse import ArgumentParser
from joker.prior.data.preprocess.deep3dfacerecon import deep3dface, predict_bfm_from_img
from joker.prior.data.preprocess.blip import predict_img_caption, naive_caption_reenactment
from PIL import Image


def preprocess_reenactment_folder(in_dir: Path, out_dir: Path):
    """
    assuming input folder with format
    000000_ref.jpg/png
    000000_drive.jpg/png
    000001_ ....

    will perform correct cropping + estimation of bfm + bfm reenactment and render out the reenacted bfm normal map
    output dir will look like
    000000_ref.jpg
    000000_drive.jpg
    000000_ctrl.jpg
    000000_prompt.txt
    000001_ ....

    """

    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    out_dir.mkdir(exist_ok=True, parents=True)
    in_ref_imgs = [Path(p) for p in sorted(glob.glob(str(in_dir) + '/*_ref.png') + glob.glob(str(in_dir) + '/*_ref.jpg') + glob.glob(str(in_dir) + '/*_ref.jpeg'))]
    in_drive_imgs = [p.parent / p.name.replace("ref", "drive") for p in in_ref_imgs]

    for ref_path, drive_path in tqdm.tqdm(zip(in_ref_imgs, in_drive_imgs), total=len(in_ref_imgs)):
        # cropping and estimating bfm parameters + generating reenacted bfm normal map
        ref_results = predict_bfm_from_img(ref_path, blur_pad=True, crop=True)
        drive_results = predict_bfm_from_img(drive_path, blur_pad=True, crop=True)
        reenacted_coeffs = drive_results["head_coeffs"]
        reenacted_coeffs["id"] = ref_results["head_coeffs"]["id"]
        reenacted_normal = deep3dface.model.render_bfm_normals(reenacted_coeffs, persc_proj=reenacted_coeffs["facemodel_perc_proj"], rasterize_size=(512, 512), ndc_proj=reenacted_coeffs["ndc_proj"])

        # image captioning
        ref_caption = predict_img_caption(ref_results["img"])
        drive_caption = predict_img_caption(drive_results["img"])
        reenacted_caption = naive_caption_reenactment(ref_caption, drive_caption)
        # we recommend to use chatgpt_prompt_reenactment (see joker/prior/data/preprocess/chatgpt_prompt_reenactment) but since that requires a paid chatgpt api license, we use the naive implementation by default

        stem = ref_path.name.split("_")[0]
        out_ref_img_path = out_dir / (stem + "_ref.jpg")
        out_drive_img_path = out_dir / (stem + "_drive.jpg")
        out_ctrl_img_path = out_dir / (stem + "_ctrl.jpg")
        out_prompt_path = out_dir / (stem + "_prompt.txt")

        Image.fromarray(ref_results["img"]).save(out_ref_img_path)
        Image.fromarray(drive_results["img"]).save(out_drive_img_path)
        Image.fromarray(reenacted_normal).save(out_ctrl_img_path)
        with open(out_prompt_path, "w") as f:
            f.write(reenacted_caption)


if __name__ == '__main__':
    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace('10.1.5.248', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    parser = ArgumentParser()
    parser.add_argument("--in_dir", type=Path, default="data/REENACTMENT/IMGS_RAW")
    parser.add_argument("--out_dir", type=Path, default="data/REENACTMENT/IMGS_PROCESSED")
    args = parser.parse_args()
    preprocess_reenactment_folder(args.in_dir, args.out_dir)
