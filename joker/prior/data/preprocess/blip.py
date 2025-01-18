import warnings
from logging import warning

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image
import torch

blip2_processor = None
blip2_model = None


def init_blip_model():
    global blip2_model, blip2_processor
    if blip2_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blip2_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24")
        blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, revision="51572668da0eb669e01a189dc22abe6088589a24")
        blip2_model.to(device).eval()


@torch.no_grad()
def predict_img_caption(img: np.ndarray):
    init_blip_model()

    img_pil = Image.fromarray(img)
    device = blip2_model.device

    blip_inputs = blip2_processor(img_pil, return_tensors="pt").to(device, torch.float16)
    generated_ids = blip2_model.generate(**blip_inputs, max_new_tokens=50)
    blip2_caption = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
    return blip2_caption


person_identifier_genders = {
    "a man": "m",
    "a woman": "f",
    "a girl": "f",
    "a boy": "m",
    "a young man": "m",
    "a young woman":"f"
}

def get_caption_person_identifier(caption):
    caption = caption.lower()
    for k in person_identifier_genders:
        if k in caption:
            return k
    return None


def naive_caption_reenactment(ref_caption, drive_caption, raise_error=False):
    ref_person_identifier = get_caption_person_identifier(ref_caption)
    drive_person_identifier = get_caption_person_identifier(drive_caption)

    if ref_person_identifier is not None and drive_person_identifier is not None:
        reenacted_caption = drive_caption.replace(drive_person_identifier, ref_person_identifier)
        if person_identifier_genders[ref_person_identifier] != person_identifier_genders[drive_person_identifier]:
            if person_identifier_genders[ref_person_identifier] == "f":
                reenacted_caption = reenacted_caption.replace(" his ", " her ")
            elif person_identifier_genders[ref_person_identifier] == "m":
                reenacted_caption = reenacted_caption.replace(" her ", " his ")
            else:
                raise ValueError("Unknown person identifier gender")
        return reenacted_caption
    else:
        msg = ("Couldnt parse one of the captions naively. Please use more sophisticated methods.\n"
               f"Ref Caption: {ref_caption}\n"
               f"Drive Caption: {drive_caption}\n"
               f"Returning Drive Caption.")
        if raise_error:
            raise ValueError(msg)
        else:
            print("Warning: " + msg)
        return drive_caption


