import gradio as gr
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
import numpy as np
import torch
import time
from argparse import ArgumentParser

from PIL import Image
from joker.prior.data.preprocess.deep3dfacerecon import deep3dface, predict_bfm_from_img
from joker.prior.data.preprocess.blip import predict_img_caption, naive_caption_reenactment
from joker.prior.models import Joker2DPrior

joker_prior = None
device = torch.device("cuda")
CKPT_PATH = "assets/joker/pretrained/bfm_ft_NersembleCelebvtext_230000.bin"
AUTO_CAPTIONING = False

def reenact_img(reference_image:np.ndarray=None, crop_reference=True, driving_image:np.ndarray=None, crop_driving=True, seed:int=20, num_inference_steps=100, guidance_scale=3.0, prompt='', mixed_precision=False) -> np.ndarray:
    """
    images are H x W x 3 np.ndarrays RGB, 0...255
    """

    a = time.time()

    # cropping and estimating bfm parameters + generating reenacted bfm normal map
    ref_results = predict_bfm_from_img(reference_image, blur_pad=False, crop=crop_reference)
    drive_results = predict_bfm_from_img(driving_image, blur_pad=False, crop=crop_driving)
    reenacted_coeffs = drive_results["head_coeffs"]
    reenacted_coeffs["id"] = ref_results["head_coeffs"]["id"]
    reenacted_normal = deep3dface.model.render_bfm_normals(reenacted_coeffs, persc_proj=reenacted_coeffs["facemodel_perc_proj"], rasterize_size=(512, 512), ndc_proj=reenacted_coeffs["ndc_proj"])

    b = time.time()

    # image captioning
    if prompt == '' and AUTO_CAPTIONING:
        ref_caption = predict_img_caption(ref_results["img"])
        drive_caption = predict_img_caption(drive_results["img"])
        prompt = naive_caption_reenactment(ref_caption, drive_caption, raise_error=True)
        # we recommend to use chatgpt_prompt_reenactment (see joker/prior/data/preprocess/chatgpt_prompt_reenactment) but since that requires a paid chatgpt api license, we use the naive implementation by default
    else:
        pass

    c = time.time()

    # inference
    control_map = torch.from_numpy(reenacted_normal).float().to(device).permute(2,0,1)[None]/255 *2 -1
    ref_imgs = torch.from_numpy(ref_results['img']).float().to(device).permute(2,0,1)[None][None] / 255
    generator = torch.Generator(device).manual_seed(seed)
    init_joker_prior()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=mixed_precision):
        output_images = joker_prior.infer(prompt=prompt,
                                          control_map=control_map,
                                          ref_imgs=ref_imgs,
                                          generator=generator,
                                          num_inference_steps=num_inference_steps,
                                          guidance_scale=guidance_scale,
                          )
    output_np = np.round(np.clip(output_images[0].permute(1,2,0).float().cpu().numpy(), 0, 1)*255).astype(np.uint8)

    d = time.time()

    total_time = d-a
    print(f'time measures:\ncropping & bfm estimation: {b-a} ({(b-a)/total_time*100:.2f}%)\ncaptioning: {c-b} ({(c-b)/total_time*100:.2f}%)\ninference: {d-c} ({(d-c)/total_time*100:.2f}%)\n')
    return output_np


def init_joker_prior():
    global joker_prior
    if joker_prior is None:
        joker_prior = Joker2DPrior.from_pretrained()
        joker_prior.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
        joker_prior.to(device).eval()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--autocaptioning', action='store_true')
    args = parser.parse_args()
    AUTO_CAPTIONING = args.autocaptioning

    demo = gr.Interface(
        title='Joker: 2D Diffusion Prior Demo',
        description='This demo lets you play around with the 2D diffusion prior of the 3DV paper \'Joker: Conditional 3D Head Synthesis with Extreme Facial Expressions\'. You can use this model to transfer an expression from a driving image to the subject visible in the reference image. For more details, please visit our project webpage: https://malteprinzler.github.io/projects/joker',
        fn=reenact_img,
        inputs=[
            gr.Image(),
            gr.Checkbox(True),
            gr.Image(),
            gr.Checkbox(True),
            gr.Slider(0,1000, 0, step=1),
            gr.Slider(0, 200, 25, step=1),
            gr.Slider(0, 20, 3.0, step=0.1),
            gr.Textbox('', placeholder='e.g. \'a man looking very angry\'' + (' (Optional: if not specified, will be extracted from driving image)' if AUTO_CAPTIONING else '')),
            gr.Checkbox(True, label='use mixed precision (speeding up the inference process while causing negligible quality reduction)')
        ],
        outputs=[gr.Image(height=512, width=512)],

        examples=[
            ['demo/2d_prior_demo/example_data/000020_ref.jpg',  # reference_image
             False,  # crop reference
             'demo/2d_prior_demo/example_data/000020_drive.jpg',  # driving_image
             False,  # crop_driving
             20,  # seed
             25,  # num_inference_steps
             3.0,  # guidance_scale
            'a man with a big smile',  # prompt
             True
             ],
            ['demo/2d_prior_demo/example_data/000020_ref.jpg',  # reference_image
             False,  # crop reference
             'demo/2d_prior_demo/example_data/000020_drive.jpg',  # driving_image
             False,  # crop_driving
             20,  # seed
             25,  # num_inference_steps
             3.0,  # guidance_scale
             'a man looking very scared',  # prompt
             True,  # mixed_precision
             ],
            ['demo/2d_prior_demo/example_data/000060_ref.jpg',  # reference_image
             False,  # crop reference
             'demo/2d_prior_demo/example_data/000060_drive.jpg',  # driving_image
             False,  # crop_driving
             61,  # seed
             25,  # num_inference_steps
             3.0,  # guidance_scale
            'a woman with her tongue sticking out',  # prompt
             True,  # mixed_precision
             ],
            ['demo/2d_prior_demo/example_data/000021_ref.png',  # reference_image
             False,  # crop reference
             'demo/2d_prior_demo/example_data/000021_drive.png',  # driving_image
             False,  # crop_driving
             21,  # seed
             25,  # num_inference_steps
             3.0,  # guidance_scale
             'a man with an angry expression and an open mouth',  # prompt
             True, # mixed_precision
             ],
            ['demo/2d_prior_demo/example_data/000027_ref.png',  # reference_image
             False,  # crop reference
             'demo/2d_prior_demo/example_data/000027_drive.png',  # driving_image
             False,  # crop_driving
             27,  # seed
             25,  # num_inference_steps
             3.0,  # guidance_scale
             'a woman sticking her tongue out',  # prompt
             True,  # mixed_precision
             ],
            ['demo/2d_prior_demo/example_data/000031_ref.png',  # reference_image
             False,  # crop reference
             'demo/2d_prior_demo/example_data/000031_drive.png',  # driving_image
             False,  # crop_driving
             31,  # seed
             25,  # num_inference_steps
             3.0,  # guidance_scale
             'a man is smiling',  # prompt
             True,  # mixed_precision
             ],
            ['demo/2d_prior_demo/example_data/000040_ref.png',  # reference_image
             False,  # crop reference
             'demo/2d_prior_demo/example_data/000040_drive.png',  # driving_image
             False,  # crop_driving
             40,  # seed
             25,  # num_inference_steps
             3.0,  # guidance_scale
             'a wooden head of a buddha is smiling',  # prompt
             True,  # mixed_precision
             ],
            ['demo/2d_prior_demo/example_data/000072_ref.png',  # reference_image
             False,  # crop reference
             'demo/2d_prior_demo/example_data/000072_drive.png',  # driving_image
             False,  # crop_driving
             72,  # seed
             25,  # num_inference_steps
             3.0,  # guidance_scale
             'a man with his tongue sticking out',  # prompt
             True,  # mixed_precision
             ],
        ]
    )

    # demo.launch(share=True)
    demo.launch(server_name="0.0.0.0", server_port=7860)