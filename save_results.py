from dataset import MyDataset
from torch.utils.data import DataLoader
from model import Generator
import torch.optim as optim
import config
import torch
from torchvision.utils import save_image
from tqdm import tqdm

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def main():
    ds = MyDataset(root_dir=config.Image_DIR)
    dl = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    
    load_checkpoint(
        config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    )
    
    loop = tqdm(dl, leave=True)
    for idx, (x) in enumerate(loop):
        x = x.to(config.DEVICE)
        gen.eval()
        with torch.no_grad():
            y = gen(x)
            y = y * 0.5 + 0.5  # remove normalization#
            x = x * 0.5 + 0.5 
            z = torch.cat((x,y), 3)
            save_image(z, f"{config.LEARNING_RATE}/clean_{idx}.png")


if __name__ == "__main__":
    main()