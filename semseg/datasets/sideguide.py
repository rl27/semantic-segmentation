import torch 
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


class SideGuide(Dataset):
    CLASSES = [
        'background', 'sidewalk_blocks', 'sidewalk_cement', 'sidewalk_urethane', 'sidewalk_asphalt', 'sidewalk_soil_stone', 'sidewalk_damaged', 'sidewalk_other', 'braille_guide_blocks_normal', 'braille_guide_blocks_damaged', 'roadway_normal', 'roadway_crosswalk', 'alley_normal', 'alley_crosswalk', 'alley_speed_bump', 'alley_damaged', 'bike_lane_normal', 'caution_zone_stairs', 'caution_zone_manhole', 'caution_zone_tree_zone', 'caution_zone_grating', 'caution_zone_repair_zone'
    ]
    PALETTE = torch.tensor([[0, 0, 0], [0, 0, 255], [217, 217, 217], [198, 89, 17], [128, 128, 128], [255, 230, 153], [55, 86, 35], [110, 168, 70], [255, 255, 0], [128, 96, 0], [255, 128, 255], [255, 0, 255], [230, 170, 255], [208, 88, 255], [138, 60, 200], [88, 38, 128], [255, 155, 155], [255, 192, 0], [255, 0, 0], [0, 255, 0], [255, 128, 0], [105, 105, 255]])
    
    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        split = 'training' if split == 'train' else 'validation'
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255  

        img_path = Path(root) / split / 'images'
        self.files = list(img_path.glob("*.jpg"))
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('images', 'labels').replace('.jpg', '.png')

        image = io.read_image(img_path, io.ImageReadMode.RGB)
        label = io.read_image(lbl_path)

        if self.transform:
            image, label = self.transform(image, label)
        return image, label.squeeze().long()


# if __name__ == '__main__':
#     from semseg.utils.visualize import visualize_dataset_sample
#     visualize_dataset_sample(MapillaryVistas, '/home/sithu/datasets/Mapillary')