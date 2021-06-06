import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional
import torch.utils.data
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.dis(x)
        x = torch.squeeze(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, out_dim),
            nn.Tanh())

    def forward(self, x):
        x = self.gen(x)
        x = torch.squeeze(x)
        return x


class conditionalGAN(nn.Module):
    def __init__(self, in_dim, out_dim, lr=0.005, batchsize=32):
        super(conditionalGAN, self).__init__()
        self.batch_size = batchsize
        self.discriminator = Discriminator(out_dim).to('cuda:0')
        self.generator = Generator(in_dim, out_dim).to('cuda:0')
        self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.dataloader = torch.utils.data.DataLoader(
                                        dataset=datasets.MNIST(root='./data/',
                                                               train=True,
                                                               transform=transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize((0.5,), (0.5,))]),
                                                               download=True),
                                        batch_size=batch_size,
                                        shuffle=True
                                                    )
        self.loss = nn.BCELoss().to('cuda:0')
        self.d_loss, self.g_loss = [], []

    def to_img(self, x):
        out = 0.5 * (x + 1)
        out = out.clamp(0, 1)
        out = out.view(-1, 1, 28, 28)
        return out

    def save_model(self):
        torch.save(self.state_dict(), './conditionalGAN.pth')
        torch.save(self.discriminator.state_dict(), './discriminator.pth')
        torch.save(self.generator.state_dict(), './generator.pth')

    def train_gan(self, epochs):
        for epoch in range(epochs):
            for i, (img, _) in enumerate(self.dataloader):
                num_img = img.size(0)
                img = img.view(num_img, -1)
                real_img = Variable(img).cuda()
                real_label = Variable(torch.ones(num_img)).cuda()
                fake_label = Variable(torch.zeros(num_img)).cuda()
                real_out = self.discriminator(real_img)
                d_loss_real = self.loss(real_out, real_label)
                real_scores = real_out

                z = Variable(torch.randn(num_img, 100)).cuda()
                fake_img = self.generator(z).detach()
                fake_out = self.discriminator(fake_img)
                d_loss_fake = self.loss(fake_out, fake_label)
                fake_scores = fake_out

                d_loss = d_loss_real + d_loss_fake
                self.d_loss.append(d_loss)
                self.d_optim.zero_grad()
                d_loss.backward()
                self.d_optim.step()

                z = Variable(torch.randn(num_img, 100)).cuda()
                fake_img = self.generator(z)
                output = self.discriminator(fake_img)
                g_loss = self.loss(output, real_label)
                self.g_loss.append(g_loss)
                self.g_optim.zero_grad()
                g_loss.backward()
                self.g_optim.step()

                if (i + 1) % 100 == 0:
                    print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                          'D real: {:.6f},D fake: {:.6f}'.format(
                        epoch, epochs, d_loss.data.item(), g_loss.data.item(),
                        real_scores.data.mean(), fake_scores.data.mean()
                    ))
                if epoch == 0:
                    real_images = self.to_img(real_img.cpu().data)
                    save_image(real_images, './img/real_images.png')
            fake_images = self.to_img(fake_img.cpu().data)
            save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))

    def plot_loss(self):
        pass


if __name__ == '__main__':
    img_size = (1, 28, 28)
    batch_size = 32
    epochs = 100
    lr = 0.00015
    z_dims = 100
    flatten_size = img_size[1] * img_size[2]
    gan = conditionalGAN(in_dim=z_dims, out_dim=flatten_size, batchsize=batch_size, lr=lr)
    # gan.train_gan(epochs)
    gan.save_model()
