import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import get_loader,check_accuracy,load_checkpoint,save_predictions_as_imgs,save_checkpoint

LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
BATCH_SIZE = 16
EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train"
TRAIN_MASK_DIR = "data/train_masks"
TEST_IMG_DIR = "data/test"

def train(loader , model , optimiser , loss_fn , scaler):
    loop = tqdm(loader)

    for batch_idx  , (data,targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)

        # forward pass

        with torch.cuda.amp.autocast():
            predictions = model(data)

            loss = loss_fn(predictions , targets)

        # backward pass

        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        # update tqdm loop

        loop.set_postfix(loss = loss.item())

def main():
    train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT , width=IMAGE_WIDTH),
            A.Rotate(limit = 35 , p = 1.0),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.1),
            A.Normalize(
                mean = [0.0,0.0,0.0],
                std = [1.0,1.0,1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    model = UNET(3,1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = optim.Adam(params=model.parameters() , lr = LR)

    train_loader = get_loader(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        BATCH_SIZE,
        train_transforms
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("saved_model.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        train(train_loader , model , optimiser , loss_fn , scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimiser.state_dict(),
        }

        save_checkpoint(checkpoint)

        check_accuracy(train_loader , model , DEVICE)

        save_predictions_as_imgs(
            train_loader, model, folder="saved_images/", device=DEVICE
        )

if __name__ == "__main__":
    main()