# -*- coding: utf-8 -*-
"""
Created on Jan 24 21:18:08 2023

@author: LanGuipeng
"""
import torch
import numpy
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.utils import make_grid
from PIL import Image
import torchvision.transforms as transforms
import dlib
import torch.nn as nn

# trans_identity = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5,0.5)])
detector = dlib.get_frontal_face_detector()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def identity_real_pic(real_pic):
    image_detector_face = real_pic
    image_detector_face = 0.5*(image_detector_face+1.0)
    grid_img_face = make_grid(image_detector_face.data, nrow=8, padding=2, pad_value=0,normalize=False, range=None, scale_each=False)
    image_detector_face = grid_img_face.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    image_detector_face = Image.fromarray(numpy.uint8(image_detector_face))
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    aligned = []
    x_aligned, _ = mtcnn(image_detector_face, return_prob=True)
    aligned.append(x_aligned)
    aligned = torch.stack(aligned).to(device)
    embeddings_face = resnet(aligned)
    # embeddings_face = trans_identity(embeddings_face.cpu().detach().numpy()).cuda()
    
    return embeddings_face

def identity_fake_pic_addition(fake_pic):
    image_detector_fake = fake_pic
    image_detector_fake = 0.5*(image_detector_fake+1.0)
    grid_img_fake = make_grid(image_detector_fake.data, nrow=8, padding=2, pad_value=0,normalize=False, range=None, scale_each=False)
    image_detector_fake = grid_img_fake.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    image_detector_fake = Image.fromarray(numpy.uint8(image_detector_fake))
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,device=device)
    image_detector_fake_for_detect_number = image_detector_fake
    _, probs, _ = mtcnn.detect(image_detector_fake_for_detect_number, landmarks=True)
    if probs.all() == None:
        # embeddings3 = trans_id(torch.ones(1,896,8,8).cuda())
        embeddings_fake_face = torch.zeros(1,512).cuda()
    else:
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        aligned = []
        x_aligned, _ = mtcnn(image_detector_fake, return_prob=True)
        aligned.append(x_aligned)
        aligned = torch.stack(aligned).to(device)
        embeddings_fake_face = resnet(aligned)
        # embeddings_fake_face = trans_identity(embeddings_fake_face.cpu().detach().numpy()).cuda()
        
    return embeddings_fake_face

def identity_fake_pic(fake_pic):
    image_detector_fake = fake_pic
    image_detector_fake = 0.5*(image_detector_fake+1.0)
    grid_img_fake = make_grid(image_detector_fake.data, nrow=8, padding=2, pad_value=0,normalize=False, range=None, scale_each=False)
    image_detector_fake = grid_img_fake.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    image_detector_fake = Image.fromarray(numpy.uint8(image_detector_fake))
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,device=device)
    image_detector_fake_for_detect_number = image_detector_fake
    batch_boxes, batch_probs, batch_points = mtcnn.detect(image_detector_fake_for_detect_number, landmarks=True)
    if batch_probs.all() == None:
        # embeddings3 = trans_id(torch.ones(1,896,8,8).cuda())
        embeddings_fake_face = torch.zeros(1,512).cuda()       
    else:
        batch_boxes, batch_probs, batch_points = mtcnn.select_boxes(
                batch_boxes, batch_probs, batch_points, image_detector_fake, method=mtcnn.selection_method
                )
        faces = mtcnn.extract(image_detector_fake, batch_boxes, save_path=None)   
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        aligned = []
        aligned.append(faces)
        aligned = torch.stack(aligned).to(device)
        embeddings_fake_face = resnet(aligned)
    return embeddings_fake_face        

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
