import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from Register import Registers
from datasets.base import ImagePathDataset
from datasets.utils import get_image_paths_from_dir
from PIL import Image
import cv2
import os


@Registers.datasets.register_with_name("custom_single")
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage="train"):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, stage)
        )
        self.flip = dataset_config.flip if stage == "train" else False
        self.to_normal = dataset_config.to_normal

        self.imgs = ImagePathDataset(
            image_paths, self.image_size, flip=self.flip, to_normal=self.to_normal
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.imgs[i]


# NOTE: dataloader in default config
@Registers.datasets.register_with_name("custom_aligned")
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage="train"):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, f"{stage}/B")
        )
        mask_paths_ori = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, f"{stage}/B-masks")
        )

        image_paths_cond = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, f"{stage}/A")
        )
        mask_paths_cond = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, f"{stage}/A-masks")
        )

        self.flip = dataset_config.flip if stage == "train" else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = ImagePathDataset(
            image_paths_ori,
            mask_paths_ori,
            self.image_size,
            flip=self.flip,
            to_normal=self.to_normal,
        )
        self.imgs_cond = ImagePathDataset(
            image_paths_cond,
            mask_paths_cond,
            self.image_size,
            flip=self.flip,
            to_normal=self.to_normal,
        )

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]

@Registers.datasets.register_with_name("custom_aligned_camera")
class CustomAlignedDatasetCamera(Dataset):
    def __init__(self, dataset_config, stage="train"):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, f"{stage}/B")
        )
        mask_paths_ori = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, f"{stage}/B-masks")
        )
        image_paths_cond = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, f"{stage}/A")
        )
        mask_paths_cond = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, f"{stage}/A-masks")
        )
        self.flip = dataset_config.flip if stage == "train" else False
        self.to_normal = dataset_config.to_normal
        self.imgs_ori = ImagePathDataset(
            image_paths_ori, mask_paths_ori, self.image_size,
            flip=self.flip, to_normal=self.to_normal
        )
        self.imgs_cond = ImagePathDataset(
            image_paths_cond, mask_paths_cond, self.image_size,
            flip=self.flip, to_normal=self.to_normal
        )
        # ---- load all camera parameters once ----
        if dataset_config.use_cam_params:
            self.cam_vals = []
            cam_dir = os.path.join(dataset_config.dataset_path, f"{stage}/cam_pose")
            for path in image_paths_ori:
                stem = Path(path).stem
                txt_file = os.path.join(cam_dir, f"{stem}.txt")
                with open(txt_file, "r") as f:
                    vals = [float(x) for x in f.read().strip().split()]
                if len(vals) != 6:
                    raise ValueError(f"{txt_file}: expected 6 values, got {len(vals)}")
                self.cam_vals.append(torch.tensor(vals, dtype=torch.float32))
        else:
            self.cam_vals = None
    def __len__(self):
        return len(self.imgs_ori)
    def __getitem__(self, idx):
        if self.cam_vals is not None:
            cam_vals = self.cam_vals[idx]  # already a tensor
            return self.imgs_ori[idx], self.imgs_cond[idx], cam_vals
        return self.imgs_ori[idx], self.imgs_cond[idx]

@Registers.datasets.register_with_name("custom_aligned_camera")
class CustomAlignedDistanceFields(Dataset):
    def __init__(self, dataset_config, stage="train"):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, f"{stage}/B")
        )
        mask_paths_ori = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, f"{stage}/B-masks")
        )
        image_paths_cond = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, f"{stage}/A")
        )
        mask_paths_cond = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, f"{stage}/A-masks")
        )
        self.flip = dataset_config.flip if stage == "train" else False
        self.to_normal = dataset_config.to_normal
        self.imgs_ori = ImagePathDataset(
            image_paths_ori, mask_paths_ori, self.image_size,
            flip=self.flip, to_normal=self.to_normal
        )
        self.imgs_cond = ImagePathDataset(
            image_paths_cond, mask_paths_cond, self.image_size,
            flip=self.flip, to_normal=self.to_normal
        )
        # ---- load all camera parameters once ----
        if dataset_config.use_cam_params:
            self.cam_vals = []
            cam_dir = os.path.join(dataset_config.dataset_path, f"{stage}/cam_pose")
            for path in image_paths_ori:
                stem = Path(path).stem
                txt_file = os.path.join(cam_dir, f"{stem}.txt")
                with open(txt_file, "r") as f:
                    vals = [float(x) for x in f.read().strip().split()]
                if len(vals) != 6:
                    raise ValueError(f"{txt_file}: expected 6 values, got {len(vals)}")
                self.cam_vals.append(torch.tensor(vals, dtype=torch.float32))
        else:
            self.cam_vals = None
    def __len__(self):
        return len(self.imgs_ori)
    def __getitem__(self, idx):
        if self.cam_vals is not None:
            cam_vals = self.cam_vals[idx]  # already a tensor
            return self.imgs_ori[idx], self.imgs_cond[idx], cam_vals
        return self.imgs_ori[idx], self.imgs_cond[idx]




@Registers.datasets.register_with_name("custom_colorization_LAB")
class CustomColorizationLABDataset(Dataset):
    def __init__(self, dataset_config, stage="train"):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, stage)
        )
        self.flip = dataset_config.flip if stage == "train" else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        img_path = self.image_paths[index]
        image = None
        try:
            image = cv2.imread(img_path)
            if self.to_lab:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        except BaseException as e:
            print(img_path)

        if p:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1).contiguous()

        if self.to_normal:
            image = (image - 127.5) / 127.5
            image.clamp_(-1.0, 1.0)

        L = image[0:1, :, :]
        ab = image[1:, :, :]
        cond = torch.cat((L, L, L), dim=0)
        return image, cond


@Registers.datasets.register_with_name("custom_colorization_RGB")
class CustomColorizationRGBDataset(Dataset):
    def __init__(self, dataset_config, stage="train"):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, stage)
        )
        self.flip = dataset_config.flip if stage == "train" else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=p),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        cond_image = image.convert("L")
        cond_image = cond_image.convert("RGB")

        image = transform(image)
        cond_image = transform(cond_image)

        if self.to_normal:
            image = (image - 0.5) * 2.0
            image.clamp_(-1.0, 1.0)
            cond_image = (cond_image - 0.5) * 2.0
            cond_image.clamp_(-1.0, 1.0)

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


@Registers.datasets.register_with_name("custom_inpainting")
class CustomInpaintingDataset(Dataset):
    def __init__(self, dataset_config, stage="train"):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, stage)
        )
        self.flip = dataset_config.flip if stage == "train" else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=p),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.0
            image.clamp_(-1.0, 1.0)

        height, width = self.image_size
        mask_width = random.randint(128, 180)
        mask_height = random.randint(128, 180)
        mask_pos_x = random.randint(0, height - mask_height)
        mask_pos_y = random.randint(0, width - mask_width)
        mask = torch.ones_like(image)
        mask[
            :,
            mask_pos_x : mask_pos_x + mask_height,
            mask_pos_y : mask_pos_y + mask_width,
        ] = 0

        cond_image = image * mask

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)
