
import argparse
import matplotlib.pyplot as plt
import warnings
import io, base64
warnings.filterwarnings("ignore")
from PIL import Image
import cv2
from colorizers import *

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

colorizer_eccv16.cuda()
colorizer_siggraph17.cuda()

def tensor_to_pillow_image(tensor: np.array) -> Image:
    tensor_np = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor_np)

def pil_to_base64(pil_img, format='PNG'):
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def inference_colorizer(img_path:str, use_gpu: bool =True, output_result: str='./test.png'):
    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    img = load_img(img_path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    if use_gpu:
        tens_l_rs = tens_l_rs.cuda()

    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel

    # img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    # out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    print(type(out_img_eccv16))
    # plt.imsave(output_result, out_img_eccv16)

    out_img_pillow = tensor_to_pillow_image(out_img_eccv16)
    # out_img_pillow.save(output_result)
    return pil_to_base64(out_img_pillow)


# inference_colorizer()
