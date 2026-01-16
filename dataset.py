import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path).convert("RGB"))
        input_image = image[:, :256, :]
        target_image = image[:, 256:, :]

        x = Image.fromarray((input_image).astype(np.uint8)).convert('RGB')
        y = Image.fromarray((target_image).astype(np.uint8)).convert('RGB')

        x = config.transforms(x)
        y = config.transforms(y)

        return x, y
    
class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = Image.open(img_path).convert("RGB")

        x = config.transforms(image)

        return x, os.path.splitext(img_file)[0]   


if __name__ == "__main__":
    dataset = MapDataset(config.TRAIN_DIR)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    x, y = next(iter(loader))

    print(x.shape, y.shape)

    save_image(x*.5+.5, "result/x.png")
    save_image(y*.5+.5, "result/y.png")
