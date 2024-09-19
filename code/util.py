# -*- coding: utf-8 -*-
"""

Created in March, 2024
@author: Ni Chen

"""

import gc
import os
import cv2

import numpy as np

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


####################################################### System setting ##############################################
def cuda_empty_cache():
    '''
    Releases all unoccupied cached memory currently held by the caching allocator
    so that those can be used in other GPU application and visible in nvidia-smi.
    '''
    gc.collect()
    torch.cuda.empty_cache()


def reset():
    try:
        from IPython import get_ipython
        get_ipython().magic('clear')
        get_ipython().magic('reset -f')

        gc.collect()
    except:
        pass


def set_device():
    n_gpu = torch.cuda.device_count()
    gpu = n_gpu - 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)

    return device


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def to_numpy(x):
    return x.cpu().numpy() if x.device != 'cpu' else x


################################################ Metrics #################################################

def minAngMSE(x_gt, x_est):
    x_gt = x_gt.flatten()
    x_est = x_est.flatten().to(x_gt.dtype).to(x_gt.device)
    
    mse_phase = (torch.abs(torch.vdot(x_gt, x_gt)) + torch.abs(torch.vdot(x_est, x_est)) - 2 * torch.abs(torch.vdot(x_gt, x_est)))
    mse_phase = mse_phase/torch.numel(x_gt)

    return mse_phase.detach().cpu().numpy()


def minAmpMSE(x_gt,x_est):
    x_gt = x_gt.flatten()
    x_est = x_est.flatten().to(x_gt.dtype).to(x_gt.device)
    
    mse_amp = torch.abs(torch.vdot(x_est,x_est)) - torch.abs(torch.vdot(x_gt, x_est))**2 / torch.abs(torch.vdot(x_gt,x_gt))
    mse_amp = mse_amp/torch.numel(x_gt)

    return mse_amp.detach().cpu().numpy()


def minCpxMSE(x_gt,x_est):
    x_gt = x_gt.flatten()
    x_est = x_est.flatten().to(x_gt.dtype).to(x_gt.device)
    
    mse_cpx = torch.abs(torch.vdot(x_est - x_gt, x_est - x_gt)) / torch.abs(torch.vdot(x_gt,x_gt))

    return mse_cpx.detach().cpu().numpy()


##########################################################################################################
def norm_01(x):
    return (x - x.min()) / (x.max() - x.min())  # Normalize to 0~1


def PSNR(x_gt, x_est):

    x_gt = norm_01(x_gt)*255
    x_est = norm_01(x_est)*255
    
    return cv2.PSNR(x_gt, x_est)
    

##########################################################################################################
def PSNR(x_gt, x_est):
    x_gt = norm_01(x_gt)*255
    x_est = norm_01(x_est)*255
    
    return cv2.PSNR(x_gt, x_est)
    
    
def FT(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))


def iFT(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x)))



def rect1d(x):
    y = (x.abs() < 1 / 2).type(torch.float32)

    idx = (x == 1 / 2)
    y[idx] = 1 


    return y
############################################ Image display ##########################################
def imshow_list(img_list, crop_size=None):
    N = len(img_list)

    Nx, Ny = np.shape(img_list[0])
    Nyc, Nxc = (Ny // 2), (Nx // 2)
    
    fig, axes = plt.subplots(1, N, figsize=(12, 10 / N), sharex=False, sharey=False)

    if isinstance(axes, plt.Axes):
        axes = [axes]

    for img, ax in zip(img_list, axes):
        if crop_size is not None:
            img = img[Nyc - crop_size//2:Nyc + crop_size//2, Nxc - crop_size//2:Nxc + crop_size//2]


        im = ax.imshow(img)
        if crop_size is not None:
            ax.axvline(x = crop_size//2, color = 'yellow', alpha=0.5)
            ax.axhline(y = crop_size//2, color = 'yellow', alpha=0.5)

        plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.show()
    

def plt_colorbar(im, pos='right', size='5%', pad=0.05):
    '''
    create an Axes on the right side of ax by default. The width of cax will be 5%
    of ax and the padding between cax and ax will be fixed at 0.05 inch by default.
    '''

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes(pos, size=size, pad=pad)
    plt.colorbar(im, cax=cax)


def imshow_multiple(imgs=None, title='', c_num=3, crop_size=100, loc=None, cmap='gray'):
    imgs = imgs.cpu().numpy() if (torch.is_tensor(imgs) and (imgs.device != 'cpu')) else imgs

    if len(np.shape(imgs))>2:
        Nz, Nx, Ny = np.shape(imgs)
        Nyc, Nxc = (Ny // 2), (Nx // 2)
        if Nz <= c_num:
            img_col_n = c_num
        else:
            img_col_n = int(np.ceil(np.sqrt(Nz)))

        img_row_n = int(np.ceil(Nz / img_col_n))
        image_height = 3
        fig = plt.figure(figsize=(img_col_n * image_height, image_height * img_row_n + 0.5))
        fig.suptitle(title, y=0.97)

        img_n = 0
        for iz in range(Nz):
            img_n = img_n + 1
            ax = fig.add_subplot(img_row_n, img_col_n, img_n)
            ax.set_title("" + str(img_n))

            single_img = imgs[iz, :, :]
            if crop_size is not None:
                single_img = single_img[Nyc - crop_size//2:Nyc + crop_size//2, Nxc - crop_size//2:Nxc + crop_size//2]


            im = ax.imshow(single_img, aspect='equal', cmap=cmap)
            if loc is not None:
                if crop_size is not None:
                    plt.plot(crop_size//2+loc[iz,1], crop_size//2+loc[iz,0], '+r', markersize=8, alpha=1, label='')
                else:
                    plt.plot(Nyc+loc[iz,1], Nxc+loc[iz,0], '+r', markersize=8, alpha=1, label='')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

        fig.tight_layout()
        plt.show()
        
    else:
        Nx, Ny = np.shape(imgs)
        Nyc, Nxc = (Ny // 2), (Nx // 2)
        
        if crop_size is not None:
            single_img = imgs[Nyc - crop_size//2:Nyc + crop_size//2, Nxc - crop_size//2:Nxc + crop_size//2]
            
            
        fig = plt.figure(figsize=(5, 5))
        fig.suptitle(title, y=0.88)
                         
        ax = fig.add_subplot(1, 1, 1)
                         
        ax=plt.imshow(single_img, cmap=cmap)

        plt.colorbar(ax, fraction=0.046, pad=0.04)
        
        fig.tight_layout()
        plt.show()