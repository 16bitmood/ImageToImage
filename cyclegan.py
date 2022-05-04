import constants
from networks import ResNetGenerator, PatchDiscriminator

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

class CycleGAN:
    def __init__(
            self, 
            train_dataset, 
            test_dataset,
            checkpoint_filename,
            examples_folder,
            discriminator = PatchDiscriminator,
            generator = ResNetGenerator,
            loss_type = 'mse',
            num_examples = 1):

        self.train_dataset = train_dataset
        self.test_dataset  = test_dataset
        self.checkpoint_filename = checkpoint_filename
        self.examples_folder = examples_folder
        self.num_examples = num_examples

        self.train_loader  = DataLoader(
            self.train_dataset,
            batch_size = constants.BATCH_SIZE,
            shuffle = True,
            num_workers = constants.NUM_WORKERS)

        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


        # Discrimator of the domain X
        self.D_X  = discriminator(net_in_channels=3, conditional=False).to(constants.DEVICE)

        # Discrimator of the domain Y
        self.D_Y  = discriminator(net_in_channels=3, conditional=False).to(constants.DEVICE)

        # Generator from Y -> X
        self.G_YX = generator(net_in_channels=3, net_out_channels=3).to(constants.DEVICE)

        # Generator from X -> Y
        self.G_XY = generator(net_in_channels=3, net_out_channels=3).to(constants.DEVICE)

        self.opt_D = optim.Adam(
            list(self.D_X.parameters()) + list(self.D_Y.parameters()),
            lr=constants.LEARNING_RATE, betas=constants.BETAS)

        self.opt_G = optim.Adam(
            list(self.G_YX.parameters()) + list(self.G_XY.parameters()),
            lr=constants.LEARNING_RATE, betas=constants.BETAS)

        # Float-16 Training
        self.D_scaler = torch.cuda.amp.GradScaler()
        self.G_scaler = torch.cuda.amp.GradScaler()

        self.loss = nn.MSELoss() if loss_type == 'mse' else nn.BCELoss() 
        self.l1_loss = nn.L1Loss()

        if constants.LOAD_CHECKPOINT:
            self.load_checkpoint()

    def train(self):
        for epoch in range(constants.NUM_EPOCHS):
            self.train_one_epoch(epoch)

            if constants.SAVE_CHECKPOINT and epoch % 5 == 0:
                self.save_checkpoint()

            self.save_examples(epoch)

    def train_one_epoch(self, epoch):
        pbar = tqdm(self.train_loader, leave=True)

        for idx, (image_X, image_Y) in enumerate(pbar):
            image_X = image_X.to(constants.DEVICE)
            image_Y = image_Y.to(constants.DEVICE)

            # Train Discriminators
            with torch.cuda.amp.autocast():
                # D_X
                fake_X = self.G_YX(image_Y)

                D_X_real = self.D_X(image_X)
                D_X_fake = self.D_X(fake_X.detach())

                D_X_real_loss = self.loss(D_X_real, torch.ones_like(D_X_real))
                D_X_fake_loss = self.loss(D_X_fake, torch.zeros_like(D_X_fake))

                D_X_loss = D_X_real_loss + D_X_fake_loss
                
                # D_Y
                fake_Y = self.G_XY(image_X)

                D_Y_real = self.D_Y(image_Y)
                D_Y_fake = self.D_Y(fake_Y.detach())

                D_Y_real_loss = self.loss(D_Y_real, torch.ones_like(D_Y_real))
                D_Y_fake_loss = self.loss(D_Y_fake, torch.zeros_like(D_Y_fake))

                D_Y_loss = D_Y_real_loss + D_Y_fake_loss

                D_loss = (D_X_loss + D_Y_loss)/2

            self.opt_D.zero_grad()
            self.D_scaler.scale(D_loss).backward()
            self.D_scaler.step(self.opt_D)
            self.D_scaler.update()

            # Train Generators
            with torch.cuda.amp.autocast():
                # Adversarial Loss
                D_X_fake = self.D_X(fake_X)
                D_Y_fake = self.D_Y(fake_Y)

                G_YX_loss = self.loss(D_X_fake, torch.ones_like(D_X_fake))
                G_XY_loss = self.loss(D_Y_fake, torch.ones_like(D_Y_fake))


                # Cycle Loss
                cycle_X = self.G_YX(fake_Y)
                cycle_Y = self.G_XY(fake_X)

                cycle_X_loss = self.l1_loss(image_X, cycle_X)
                cycle_Y_loss = self.l1_loss(image_Y, cycle_Y)

                G_loss = (
                    G_XY_loss + G_YX_loss
                    + constants.L1_LAMBDA_CYCLE*(cycle_X_loss + cycle_Y_loss)
                )

                # Identity Loss
                if constants.L1_LAMBDA_IDENTITY > 0:
                    identitiy_loss_X = self.l1_loss(image_X, self.G_YX(image_X))
                    identitiy_loss_Y = self.l1_loss(image_Y, self.G_XY(image_Y))
                    G_loss += constants.L1_LAMBDA_IDENTITY*(identitiy_loss_X+identitiy_loss_Y)

            self.opt_G.zero_grad()
            self.G_scaler.scale(G_loss).backward()
            self.G_scaler.step(self.opt_G)
            self.G_scaler.update()

            if idx % 200 == 0:
                self.save_examples(f'{epoch}_{idx}')

            if idx % 10 == 0:
                pbar.set_postfix(
                    D_X_real=torch.sigmoid(D_X_real).mean().item(),
                    D_Y_real=torch.sigmoid(D_Y_real).mean().item(),
                    D_X_fake=torch.sigmoid(D_X_fake).mean().item(),
                    D_Y_fake=torch.sigmoid(D_Y_fake).mean().item(),
                )
    
    def save_examples(self, epoch, n = None):
        folder = self.examples_folder
        i = iter(self.test_loader)
        for n in range(self.num_examples if n is None else n):
            X, Y = next(i)
            X, Y = X.to(constants.DEVICE), Y.to(constants.DEVICE)

            self.G_XY.eval()
            self.G_YX.eval()
            with torch.no_grad():
                fake_Y = self.G_XY(X)
                fake_X = self.G_YX(Y)
                save_image(Y*0.5 + 0.5, os.path.join(folder, f"{epoch}_{n}_Y.png"))
                save_image(X*0.5 + 0.5, os.path.join(folder, f"{epoch}_{n}_X.png"))
                save_image(fake_Y*0.5 + 0.5, os.path.join(folder, f"{epoch}_{n}_Y_fake.png"))
                save_image(fake_X*0.5 + 0.5, os.path.join(folder, f"{epoch}_{n}_X_fake.png"))

            self.G_XY.train()
            self.G_YX.train()

    def save_checkpoint(self):
        filename = self.checkpoint_filename
        checkpoint = {
            "D_X_state_dict": self.D_X.state_dict(),
            "D_Y_state_dict": self.D_Y.state_dict(),
            "opt_D_state_dict": self.opt_D.state_dict(),

            "G_XY_state_dict":  self.G_XY.state_dict(),
            "G_YX_state_dict":  self.G_YX.state_dict(),
            "opt_G_state_dict": self.opt_G.state_dict(),
        }
        torch.save(checkpoint, filename)
        print(">> Checkpoint Saved")


    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_filename, map_location=constants.DEVICE)
        self.G_XY.load_state_dict(checkpoint['G_XY_state_dict'])
        self.G_YX.load_state_dict(checkpoint['G_YX_state_dict'])
        self.opt_G.load_state_dict(checkpoint['opt_G_state_dict'])

        self.D_X.load_state_dict(checkpoint['D_X_state_dict'])
        self.D_Y.load_state_dict(checkpoint['D_Y_state_dict'])
        self.opt_D.load_state_dict(checkpoint['opt_D_state_dict'])

        for optimizer in (self.opt_G, self.opt_D):
            for param_group in optimizer.param_groups:
                param_group["lr"] = constants.LEARNING_RATE
        print('>> Checkpoint Loaded')
