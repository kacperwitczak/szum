import cv2
import albumentations as A

def build_spatial_transform(is_train: bool = True, p_apply: float = 0.9) -> A.Compose | None:
    if not is_train:
        return None
        
    return A.ReplayCompose([
        A.Affine(
            scale=(0.90, 1.10),
            translate_percent=(-0.08, 0.08),
            rotate=(-5, 5),
            p=0.7,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.10,
            p=0.3,
        ),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.1),
    ], p=p_apply)
