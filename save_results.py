from dataset import MyDataset, MapDataset
from torch.utils.data import DataLoader
from generator_model import Generator
import torch.optim as optim
import config
from utils import load_checkpoint
import torch
from torchvision.utils import save_image
from tqdm import tqdm




def main():
    path_src = "result/"
    path_desc = ""
    ds = MyDataset(root_dir=path_src)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    print(len(dl))

    
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    # opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    # if config.LOAD_MODEL:
    #     load_checkpoint(
    #         config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    #     )
    
    loop = tqdm(dl, leave=True)
    for idx, (x, name) in enumerate(loop):
        x = x.to(config.DEVICE)
        gen.eval()
        with torch.no_grad():
            y = gen(x)
            y = y * 0.5 + 0.5  # remove normalization#
            x = x * 0.5 + 0.5 
            z = torch.cat((x,y), 3)
            save_image(z, f"E:/COCO_depth/{name[0]}.png")

    # x, name = next(iter(dl))
    # x = x.to(config.DEVICE)
    # print(name[0])
    # gen.eval()
    # with torch.no_grad():
    #     y = gen(x)
    #     y = y * 0.5 + 0.5  # remove normalization#
    #     x = x * 0.5 + 0.5 
    #     z = torch.cat((x,y), 3)
    #     save_image(z, f"F:/result/{name[0]}.png")


def test():
    ds = MyDataset("D:/datasets/COCO dataset/train")
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    
    x, _ = next(iter(dl))
    
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    load_checkpoint(
        config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    )

    x = x.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        fake = gen(x)
        fake = fake * 0.5 + 0.5  # remove normalization#
        x = x * 0.5 + 0.5 
        z = torch.cat([x, fake], 0)
        save_image(z, f"result/gen_coco.png")


if __name__ == "__main__":
    test()