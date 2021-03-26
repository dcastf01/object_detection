import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


# get from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/albumentations_tutorial/full_pytorch_example.py
def get_transform_from_aladdinpersson():
    transform = A.Compose(
        [
            A.Resize(width=1920, height=1080),
            A.RandomCrop(width=1280, height=720,p=0.5),
            A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
            A.
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.ColorJitter(p=0.5),
            ], p=1.0),
            A.Normalize(
                mean=[0, 0, 0],
                std=[1, 1, 1],
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ]
    )
    return transform