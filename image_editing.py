from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from utils import dict2namespace
from runners.utils import save_single_image

import torch
import torchvision.transforms as transforms
import argparse
import yaml
from PIL import Image
import pdb
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from torchvision.utils import save_image


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('-c', '--config', type=str, default='BB_base.yml', help='Path to the config file')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, 0,1,2,3 cpu=-1')
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args

    return namespace_config

def get_model_from_checkpoint(config):

    net = LatentBrownianBridgeModel(config.model).to(
            config.training.device[0]
        )
    
    if config.model.__contains__('model_load_path') and config.model.model_load_path is not None:
        model_states = torch.load(config.model.model_load_path, map_location='cpu')

    net.load_state_dict(model_states['model'])

    return net

def get_image_and_mask(img_path,franka_mask_path,xarm_mask_path):
    
    transform = transforms.Compose(
        [
            transforms.Resize((256,256)),
            transforms.ToTensor(), # scales it from 0 to 1 
        ]
    )

    try:
        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = transform(image)
        image = (image - 0.5) * 2.0
        image.clamp_(-1.0, 1.0)

    except BaseException as e:
        print(img_path)

    try:
        franka_mask = Image.open(franka_mask_path)
        if franka_mask and (not franka_mask.mode == "L"):
            franka_mask = franka_mask.convert("L")
        franka_mask = transform(franka_mask)

    except BaseException as e:
        print(franka_mask_path)

    try:
        xarm_mask = Image.open(xarm_mask_path)
        if xarm_mask and (not xarm_mask.mode == "L"):
            xarm_mask = xarm_mask.convert("L")
        xarm_mask = transform(xarm_mask)

    except BaseException as e:
        print(xarm_mask_path)

    return image,franka_mask,xarm_mask

def get_inpaint_area(xarm_mask,franka_mask):
    xarm_bool = xarm_mask.bool()
    franka_bool = franka_mask.bool()
    diff_mask = franka_bool & ~xarm_bool
    
    return diff_mask.to(dtype=franka_mask.dtype)        

if __name__ == "__main__":
    
    img_path = "sim_2_real_split_filtered/train/B/image_0005.png"
    franka_mask_path = "sim_2_real_split_filtered/train/B-masks/image_0005.png"
    xarm_mask_path = "sim_2_real_split_filtered/train/A-masks/image_0005.png"
    condition_path = "scratchbook"


    config = parse_args_and_config()
    args = config.args
    gpu_ids = args.gpu_ids
    gpu_list = gpu_ids.split(",")
    config.training.use_DDP = False
    config.training.device = [torch.device(f"cuda:{gpu_list[0]}")]

    net = get_model_from_checkpoint(config=config)

    img, franka_mask, xarm_mask = get_image_and_mask(img_path, franka_mask_path, xarm_mask_path)

    img = img.to(f"cuda:{gpu_list[0]}")
    img = img.unsqueeze(0)
    out = net.sample(img)

    out = out[0]

    save_single_image(
        out,
        condition_path,
        "test.png",
        to_normal=True,
    )

    save_single_image(
        img[0],
        condition_path,
        "input.png",
        to_normal=True,
    )

    pipeline = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
    ).to(f"cuda:{gpu_list[0]}")

    init_image = load_image(img_path)
    generator = torch.Generator("cuda").manual_seed(92)
    prompt = ""
    mask = get_inpaint_area(xarm_mask,franka_mask)
    save_image(mask, "scratchbook/mask.png")
    image = pipeline(prompt=prompt, image=init_image, mask_image=mask, generator=generator).images[0]
    image.save("scratchbook/inpaint.png")