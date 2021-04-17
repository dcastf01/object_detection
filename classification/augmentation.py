import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


# get from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/albumentations_tutorial/full_pytorch_example.py
def get_transform_from_aladdinpersson()->dict:
    train_transform = A.Compose(
        [
            A.Resize(width=750, height=424), #proporcion 1,77
            A.RandomCrop(width=500, height=283,p=0.5),#proporcion 1.98
            A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.ColorJitter(p=0.5),
            ], p=1.0),
            A.Resize(224,224),
            A.Normalize(
                mean=[0, 0, 0],
                std=[1, 1, 1],
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(224,224),
            # A.CenterCrop(height=128, width=128),
            A.Normalize(
                mean=[0, 0, 0],
                std=[1, 1, 1],
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ]
        )
    
    transforms = {
        "train": train_transform,
        "val": val_transform,
        "test": val_transform,
    }
    return transforms