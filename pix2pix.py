import constants
from networks import SkipGenerator, PatchDiscriminator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

class Pix2Pix:
    def __init__(
            self, 
            train_dataset, 
            test_dataset,
            checkpoint_filename,
            examples_folder,
            discriminator = PatchDiscriminator,
            generator = SkipGenerator,
            num_examples = 1
        ):
        self.train_dataset = train_dataset
        self.test_dataset  = test_dataset
        self.checkpoint_filename = checkpoint_filename
        self.examples_folder = examples_folder
        self.num_examples = num_examples

        self.train_loader  = DataLoader(
            self.train_dataset,
            batch_size = constants.BATCH_SIZE,
            shuffle = True,
            num_workers = constants.NUM_WORKERS,
        )
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        self.D = discriminator(net_in_channels=3).to(constants.DEVICE)
        self.G = generator(net_in_channels=3, net_out_channels=3).to(constants.DEVICE)

        self.opt_D = optim.Adam(
            self.D.parameters(), lr=constants.LEARNING_RATE, betas=constants.BETAS)
        self.opt_G = optim.Adam(
            self.G.parameters(), lr=constants.LEARNING_RATE, betas=constants.BETAS)

        self.D_scaler = torch.cuda.amp.GradScaler()
        self.G_scaler = torch.cuda.amp.GradScaler()


        self.bce     = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

        if constants.LOAD_CHECKPOINT:
            self.load_checkpoint()

    def train(self):
        for epoch in range(constants.NUM_EPOCHS):
            self.train_one_epoch()

            if constants.SAVE_CHECKPOINT and epoch % 5 == 0:
                self.save_checkpoint()

            self.save_examples(epoch)

    def train_one_epoch(self):
        loop = tqdm(self.train_loader, leave=True)

        for idx, (X, Y) in enumerate(loop):
            X = X.to(constants.DEVICE)
            Y = Y.to(constants.DEVICE)

            with torch.cuda.amp.autocast():
                D_real = self.D(X, Y)
                D_real_loss = self.bce(D_real, torch.ones_like(D_real))

                Y_fake = self.G(X)
                D_fake = self.D(X, Y_fake.detach()) 
                D_fake_loss = self.bce(D_fake, torch.zeros_like(D_fake))

                D_loss = (D_real_loss + D_fake_loss) / 2

            self.opt_D.zero_grad()
            self.D_scaler.scale(D_loss).backward()
            self.D_scaler.step(self.opt_D)
            self.D_scaler.update()

            with torch.cuda.amp.autocast():
                D_fake = self.D(X, Y_fake)
                G_fake_loss = self.bce(D_fake, torch.ones_like(D_fake))
                L1 = self.l1_loss(Y_fake, Y) * constants.L1_LAMBDA
                G_loss = G_fake_loss + L1

            self.opt_G.zero_grad()
            self.G_scaler.scale(G_loss).backward()
            self.G_scaler.step(self.opt_G)
            self.G_scaler.update()

            if idx % 10 == 0:
                loop.set_postfix(
                    D_real=torch.sigmoid(D_real).mean().item(),
                    D_fake=torch.sigmoid(D_fake).mean().item(),
                )
    
    def save_examples(self, epoch):
        folder = self.examples_folder
        i = iter(self.test_loader)
        for n in range(self.num_examples):
            X, Y = next(i)
            X, Y = X.to(constants.DEVICE), Y.to(constants.DEVICE)

            self.G.eval()
            with torch.no_grad():
                Z = self.G(X)
                save_image(Z*0.5 + 0.5, folder + f"/{epoch}_{n}_Y_fake.png")
                save_image(X*0.5 + 0.5, folder + f"/{epoch}_{n}_X.png")
                save_image(Y*0.5 + 0.5, folder + f"/{epoch}_{n}_Y.png")
            self.G.train()
    

    def save_checkpoint(self):
        filename = self.checkpoint_filename
        checkpoint = {
            "G_state_dict":     self.G.state_dict(),
            "opt_G_state_dict": self.opt_G.state_dict(),

            "D_state_dict":     self.D.state_dict(),
            "opt_D_state_dict": self.opt_D.state_dict(),
        }
        torch.save(checkpoint, filename)
        print(">> Checkpoint Saved")


    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_filename, map_location=constants.DEVICE)
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.opt_G.load_state_dict(checkpoint['opt_G_state_dict'])

        self.D.load_state_dict(checkpoint['D_state_dict'])
        self.opt_D.load_state_dict(checkpoint['opt_D_state_dict'])

        for optimizer in (self.opt_G, self.opt_D):
            for param_group in optimizer.param_groups:
                param_group["lr"] = constants.LEARNING_RATE
        print('>> Checkpoint Loaded')
