# -*- coding: utf-8 -*-
"""
Ptychographic Wavefront Camera

Created in March, 2024
Author: Ni Chen (https://ni-chen.github.io/) 

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import random

import torch
import torch.fft

import torch.nn.functional as F
import torchvision.transforms as transforms 
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from optimizer import AdamP
from util import *

torch.set_default_dtype(torch.float64)

seed = 2024
np_rng = np.random.default_rng(seed)
torch.manual_seed(seed) 


##############################################################################
class FourierPtychography:
    
    def __init__(self, N_hr=256, N_lr=128, 
                 ft_pad=1, padding=1, 
                 aperture_shape='rect', aperture_num=None,
                 is_energy_constraint=True, is_band_limit=False, photons=1e6, 
                 is_exp=False, im_meas=None, cam_loc=None, overlapping_ratio=None,
                 kappa_tv=[0.0,0.0], tv_tau=1e-3,
                 tol=1e-3, is_verbose=False, device='cpu', *args, **kwargs):
            
        self.device = device
        
        self.is_exp = is_exp 
        self.im_meas = im_meas    
        self.cam_loc = cam_loc
        self.overlapping_ratio = overlapping_ratio
        
        self.N_hr = int(N_hr)           # Original image dimensional: N_hr x N_hr
        self.N_lr = int(N_lr)           # Measured image dimensional: N_lr x N_lr
        
        self.padding = padding          # Adding padding to the image fourier spectrum
        
        self.ft_pad = ft_pad            # padding ratio of the sub FT spectrum to perform oversampling
        self.ft_pad_size = self.ft_pad*self.N_lr//2            
        
        self.aperture_shape = aperture_shape    # Aperture shape: rect, circ
        
        self.photons = photons      
                
        self.is_energy_constraint = is_energy_constraint 
        
        self.is_band_limit = is_band_limit
                
        self.aperture_num = aperture_num        
        self.meas_num = 0 if self.aperture_num is None else self.aperture_num**2 
        
        self.aperture_loc = None               
        self.pattern = None
        self.pattern_mask = None
        
        self.lp_filter = None
        
        self.is_verbose = is_verbose
        self.tol=tol
        
        self.kappa_tv = kappa_tv
        self.tv_tau = tv_tau
        
        self.ang_mse_hist = []
        self.amp_mse_list = []        
        self.loss_hist = []
        
        if self.is_band_limit:
            self.get_band_pass_filter()
            
        if self.padding==1:
            self.padding = self.N_hr//2
        
        self._check_params_()


    def _check_params_(self,):
        self.is_energy_constraint = True            
        print('Check simulation parameters passed!')


    def sampling(self, overlapping_ratio=0.75):        
        side_overlapping_ratio = overlapping_ratio

        
        if self.padding is None:        
            if self.N_hr%self.N_lr==0 and side_overlapping_ratio==0:
                self.padding = 0
            else:
                ft_left_spacing = int((self.N_hr-self.N_lr) % np.ceil(self.N_lr*(1-side_overlapping_ratio)))
                self.padding = int(np.round(np.ceil(self.N_lr*(1-side_overlapping_ratio)) - ft_left_spacing)/2) + 1
         
        self.N_hr_pad = self.N_hr + 2 * self.padding
        
        aperture_spacing = int(np.ceil(self.N_lr * (1 - side_overlapping_ratio)))
        
        xy = torch.arange(0, self.N_hr + 2*self.padding - self.N_lr + 1, aperture_spacing)
        
        if self.aperture_num is not None and self.aperture_num<=len(xy):
            xy = xy[:self.aperture_num]
        
        self.meas_num = len(xy)**2
        
        xloc, yloc = torch.meshgrid(xy, xy, indexing='xy')
        xloc, yloc = xloc.flatten().int(), yloc.flatten().int()

        if self.padding is not None:  
            shift_x = np.round(((xloc.max() + self.N_lr)/2 - (self.N_hr_pad/2))).int()
            shfit_y = np.round(((yloc.max() + self.N_lr)/2 - (self.N_hr_pad/2))).int()

            xloc -= shift_x
            yloc -= shfit_y
        
        self.aperture_loc = torch.stack([xloc.reshape([-1,1]), yloc.reshape([-1,1])], dim=1)
        
        self.set_sampling_pattern(aperture_loc=self.aperture_loc)


    def get_pupil(self, ):
        r = self.N_lr / 2
        
        y, x = torch.meshgrid(torch.arange(-r + 0.5, r, step=torch.tensor(1)),
                              torch.arange(-r + 0.5, r, step=torch.tensor(1)), indexing='ij')

        if self.aperture_shape=='rect':
            pupil = rect1d(x / (2 * r)) * rect1d(y / (2 * r))

        elif self.aperture_shape=='circ':
            circ = ((x ** 2 + y ** 2) < r ** 2).type(torch.float32)
            idx = ((x ** 2 + y ** 2) == r ** 2)
            circ[idx] = 0.5

            pupil = torch.sigmoid(r ** 2 - (x ** 2 + y ** 2))
            
        else:
            raise Exception("Undefined aperture type (accepted: rect, circ)")


        pupil = torch.as_tensor(pupil.float(), device=self.device)
        
        return pupil
        
    
    def set_sampling_pattern(self, aperture_loc=[]):
        self.pattern = torch.zeros([self.N_hr_pad, self.N_hr_pad]).to(self.device)
        
        for idx in range(self.meas_num):
            self.pattern[aperture_loc[idx, 0]:aperture_loc[idx, 0] + self.N_lr,
                         aperture_loc[idx, 1]:aperture_loc[idx, 1] + self.N_lr] += self.get_pupil()
        
        self.pattern_mask = (self.pattern==0.0)
        # self.pattern = torch.maximum(self.pattern, torch.as_tensor(0.5))
        self.pattern = self.pattern.to(self.device)
        


    def get_band_pass_filter(self,):
        lp_filter = np.zeros([self.N_hr, self.N_hr])
        x, y = np.meshgrid(np.arange(self.N_hr), np.arange(self.N_hr))
        radius = np.sqrt((x - self.N_hr/2)**2 + (y - self.N_hr/2)**2)
        lp_filter[radius < self.N_hr/2] = 1
        
        self.lp_filter = torch.as_tensor(lp_filter).to(self.device)
        
        
    def show_sampling(self, pattern=[], is_black=True, out_dir=''): 
        if is_black:
            fig=plt.figure(figsize=(5, 5))
            plt.imshow(self.pattern.cpu().numpy(), cmap='gray')  
            cb = plt.colorbar(fraction=0.046, pad=0.04)
            plt.clim([0, None])
            
        else:
            N_img = self.N_hr + 2*self.padding

            if self.aperture_shape=='rect':
                fig, ax = plt.subplots(figsize=(5,5), facecolor='white')
                for i in range(self.meas_num):
                    rect = Rectangle((self.aperture_loc[i,0].numpy(), self.aperture_loc[i,1].numpy()), 
                                     self.N_lr, self.N_lr,  
                                     alpha=0.1, fill=True, facecolor='gray', edgecolor='black', linestyle='-')

                    ax.add_patch(rect)

                ax.set_xlim(0, N_img)
                ax.set_ylim(0, N_img)
                ax.set_aspect('equal', adjustable='box')
                plt.gca().invert_yaxis()


            elif self.aperture_shape=='circ':
                fig,ax = plt.subplots(figsize=(5,5), facecolor='white')
                for i in range(self.meas_num):
                    circle = plt.Circle(self.aperture_loc[i]+self.N_lr/2, radius=self.N_lr/2, 
                                        fill=True, facecolor='gray', edgecolor='black', alpha=0.1, linestyle='-')
                    ax.add_artist(circle)

                ax.set_xlim(0, N_img)
                ax.set_ylim(0, N_img)
                ax.set_aspect('equal', adjustable='box')
                plt.gca().invert_yaxis()
                cb = plt.colorbar() 

            else:
                raise Exception("Undefined aperture type (accepted: rect, circ)")

        plt.axis('off')

        plt.show()


    def fp_forward(self, x=[], is_measurement=True, is_noisy=True):
        
        if is_measurement:
            self.x = torch.sqrt(torch.tensor(self.photons))/torch.sqrt(torch.tensor(2.0)) * x.to(torch.complex128).to(self.device)
        
            if self.is_band_limit:
                self.x_band = iFT(FT(self.x)*self.lp_filter)
            else:
                self.x_band = self.x
                
            x = self.x_band


        x_ft = FT(x) / self.N_hr        
        x_ft = F.pad(x_ft, (self.padding,) * 4)
        
        if self.is_energy_constraint == True:
            # x_ft = x_ft / torch.sqrt(self.pattern)
            x_ft[~self.pattern_mask] = x_ft[~self.pattern_mask] / torch.sqrt(self.pattern[~self.pattern_mask])

                
        if self.ft_pad > 0:
            measure_size = self.N_lr + 2*self.ft_pad_size
            im_measure = torch.zeros([self.meas_num, measure_size, measure_size], device=self.device)
        else:
            im_measure = torch.zeros([self.meas_num, self.N_lr, self.N_lr], device=self.device)

        for idx in range(self.meas_num):
            x_ft_sub = x_ft[self.aperture_loc[idx, 0]:self.aperture_loc[idx, 0] + self.N_lr,
                            self.aperture_loc[idx, 1]:self.aperture_loc[idx, 1] + self.N_lr] * self.get_pupil()
            
            if self.ft_pad > 0:
                x_ft_sub = F.pad(x_ft_sub, (self.ft_pad_size,) * 4)
            
            im_measure[idx, :, :] = (iFT(x_ft_sub) * (self.N_lr + 2*self.ft_pad_size)).abs()**2

        if is_noisy:
            im_measure = (torch.poisson(im_measure))

        if is_measurement:
            self.im_meas = im_measure

        return im_measure
    
    
    def fp_forward_gd(self, x=[]):  
        x_ft = FT(x) / self.N_hr 
        x_ft = F.pad(x_ft, (self.padding,) *4)
        
        if self.is_energy_constraint == True:
            x_ft[~self.pattern_mask] = x_ft[~self.pattern_mask] / torch.sqrt(self.pattern[~self.pattern_mask])

            
        if self.ft_pad > 0:
            measure_size = self.N_lr + 2*self.ft_pad_size
            im_measure = torch.zeros([self.meas_num, measure_size, measure_size], device=self.device)
        else:
            im_measure = torch.zeros([self.meas_num, self.N_lr, self.N_lr], device=self.device)

        for idx in range(self.meas_num):
            x_ft_sub = x_ft[self.aperture_loc[idx, 0]:self.aperture_loc[idx, 0] + self.N_lr,
                            self.aperture_loc[idx, 1]:self.aperture_loc[idx, 1] + self.N_lr] * self.get_pupil()
            
            if self.ft_pad > 0:
                x_ft_sub = F.pad(x_ft_sub, (self.ft_pad_size,) *4)
            
            im_measure[idx, :, :] = (iFT(x_ft_sub) * (self.N_lr + 2*self.ft_pad_size)).abs()**2
            
            c = self.im_meas[idx].mean()/im_measure[idx].mean()
            
            if self.is_exp==True:
                im_measure[idx] = c*im_measure[idx].mean()
            

        return im_measure, c

    
    
    def fp_gs(self, x_est=None, max_it_gs=500, is_continue=False, tol=1e-3, is_verbose=True):
                
        if is_continue == False:
            print(f'GS iteration number is {max_it_gs}')
            
        self.loss_hist = []
        self.ang_mse_hist = []
        self.amp_mse_hist = []

        if x_est == None:       
            x_est = torch.randn([self.N_hr, self.N_hr]) + 1j*torch.randn([self.N_hr, self.N_hr])
            x_est = x_est.to(self.device)
        else:
            x_est = torch.mean(torch.sqrt(self.im_meas), dim=(0))
            x_est = F.interpolate(x_est[(None,) * 2], size=(self.N_hr, self.N_hr), mode='nearest-exact').squeeze().to(torch.complex64)
        
        measure = torch.sqrt(self.im_meas)
        for idx in tqdm(range(max_it_gs)): 

            x_est_ft = FT(x_est)/self.N_hr            
            x_est_ft = F.pad(x_est_ft, (self.padding,) * 4)            
            if self.is_energy_constraint == True:
                x_est_ft[~self.pattern_mask] = x_est_ft[~self.pattern_mask] / torch.sqrt(self.pattern[~self.pattern_mask])

            
            order = list(range(self.meas_num))
            random.shuffle(order)
            for im_idx in order:
                
                pupil = self.get_pupil()                

                idx_left = self.aperture_loc[im_idx, 1]
                idx_right = self.aperture_loc[im_idx, 1] + self.N_lr
                idx_up = self.aperture_loc[im_idx, 0]
                idx_down = self.aperture_loc[im_idx, 0] + self.N_lr
                
                x_est_ft_sub = x_est_ft[idx_up:idx_down, idx_left:idx_right] * pupil

                if self.ft_pad > 0:
                    x_est_ft_sub = F.pad(x_est_ft_sub, (self.ft_pad_size,) * 4) 
                
                im_sub_prime = iFT(x_est_ft_sub)
                
                im_sub_update = measure[im_idx] * torch.exp(1j * torch.angle(im_sub_prime))
            
                im_sub_ft_update = FT(im_sub_update) / (self.N_lr + 2*self.ft_pad_size)
                if self.ft_pad > 0:
                    im_sub_ft_update = transforms.CenterCrop((self.N_lr, self.N_lr))(im_sub_ft_update) 
                
                x_est_ft[idx_up:idx_down, idx_left:idx_right] = x_est_ft[idx_up:idx_down, idx_left:idx_right] * (1 - pupil) + im_sub_ft_update * pupil

            
            update_region = slice(self.padding, self.padding + self.N_hr)     
            x_est = iFT(x_est_ft[update_region, update_region] * torch.sqrt(self.pattern[update_region, update_region])) * self.N_hr
    
            if self.is_exp==False:
                self.ang_mse_hist.append(minAngMSE(self.x_band, x_est))
                self.amp_mse_hist.append(minAmpMSE(self.x_band, x_est))
                
            # calculate loss
            if is_verbose:
                img_pred = self.fp_forward(x=x_est, is_measurement=False, is_noisy=False)

                
                loss = torch.mean((torch.sqrt(img_pred) - torch.sqrt(self.im_meas)) ** 2, dim=(1,2)).mean()
                self.loss_hist.append(loss.cpu().data.numpy())     

                # criterion = (img_pred.flatten()-self.im_meas.flatten()).norm()/self.im_meas.flatten().norm()

            criterion = abs(self.ang_mse_hist[idx]-self.ang_mse_hist[idx-1]) if idx>0 else self.tol
            # print(f'criterion {criterion}')
            if criterion < self.tol:
                break     
           
        if self.is_exp==False:
            print(f'After GS, Ang MSE is {self.ang_mse_hist[-1]}; Amp MSE is {self.amp_mse_hist[-1]}.')
        
        return x_est.cpu()
    
    

    def fp_gd(self, x_est=None, max_it_gd=1000, lr=0.5, tol=1e-3, is_continue=False, is_verbose=True):
            
        if not is_continue:
            if is_verbose:
                print(f'Gradient Descent iteration is {max_it_gd}; lr is {lr}')
                
            self.loss_hist = []
            self.ang_mse_hist = []
            self.amp_mse_hist = []
            
        if x_est is None:
            x_mean = torch.mean(self.im_meas, dim=(0))
            x_est = F.interpolate(torch.sqrt(x_mean)[(None,) *2],
                                  size=(self.N_hr, self.N_hr), mode='nearest').squeeze().to(torch.complex64)
        else:
            x_est = torch.as_tensor(x_est, device=self.device)

        x_est = nn.Parameter(x_est, requires_grad=True)
        vars = [{'params': x_est, 'lr': lr}]

        optimizer = AdamP(vars)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max_it_gd//2, gamma=0.1)
        for idx in tqdm(range(max_it_gd)):
            optimizer.zero_grad()
            
            img_pred, c = self.fp_forward_gd(x=x_est)

            loss = torch.sum((torch.sqrt(img_pred) - torch.sqrt(self.im_meas))**2, dim=(1,2)).sum()
            
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()            

            if self.is_exp==False:
                self.ang_mse_hist.append(minAngMSE(self.x_band, x_est))
                self.amp_mse_hist.append(minAmpMSE(self.x_band, x_est))
                criterion = abs(self.ang_mse_hist[idx]-self.ang_mse_hist[idx-1]) if idx>0 else self.tol


            # criterion = (img_pred.flatten()-self.im_meas.flatten()).norm()/self.im_meas.flatten().norm()
            # print(f'criterion {criterion}')
            if criterion < self.tol:
                break    


        return (x_est).cpu().detach()

    
    def fp_gs_gd(self, x_est=None, max_it_gs=500, max_it_gd=500, lr=0.5, is_verbose=True):        
        print(f'GS-GD method: GS iteration number is {max_it_gs}; GD iteration is {max_it_gd}')
        
        self.loss_hist = []        
        self.ang_mse_hist = []
        self.amp_mse_hist = []
        
        x_est = self.fp_gs(x_est=x_est, max_it_gs=max_it_gs, is_continue=True, is_verbose=is_verbose)
        x_est = self.fp_gd(x_est=x_est, max_it_gd=max_it_gd, lr=lr, is_continue=True, is_verbose=is_verbose)
        
        return x_est
    

##############################################################################

def exclude_exception(x=[], x_min=None, x_max=None): 
    
    if x_max is not None:
        x = [e for e in x if e <= x_max]
        
    if x_min is not None:
        x = [e for e in x if e >= x_min]
    
    return np.array(x)



def group_test_exclude(try_num=5, N_hr=256, N_lr=128, overlapping_ratio=0.5625, padding=1, 
                aperture_num=None, photon_num=1e6, ft_pad=1, is_band_limit = False,
                method='gs_gd', max_it_gs=1000, max_it_gd=500,
                gt=None, device='cpu'):
    
    mse_ang = []
    mse_amp = []
    mse_cpx = []
    
    psnr_ang = []
    psnr_amp = []
    
    for idx in range(try_num):
        if gt == None:
            gt = torch.randn([N_hr, N_hr]) + 1j*torch.randn([N_hr, N_hr])
        
        fp = FourierPtychography(N_hr=N_hr, N_lr=N_lr, 
                                 padding=padding, aperture_num=aperture_num, 
                                 is_band_limit=is_band_limit, photons=photon_num, 
                                 ft_pad=ft_pad, device=device)
        
        fp.sampling(overlapping_ratio=overlapping_ratio)
        
        print(f'measurement number is: {fp.meas_num}')

        fp.fp_forward(gt)

        x_est = fp.fp_gs_gd(max_it_gs=max_it_gs, max_it_gd=max_it_gd)
        
        mse_ang.append(minAngMSE(fp.x_band, x_est))
        # mse_amp.append(minAmpMSE(fp.x_band, x_est))
        # mse_cpx.append(minCpxMSE(fp.x_band, x_est))
        # psnr_ang.append(PSNR(fp.x_band.angle().cpu().numpy(), x_est.angle().cpu().numpy()))
        # psnr_amp.append(PSNR(fp.x_band.abs().cpu().numpy(), x_est.abs().cpu().numpy()))        
        
        print(f'Test {idx+1}, cpx mse is {minCpxMSE(fp.x_band, x_est)}, angle mse is {minAngMSE(fp.x_band, x_est)}, amp mse is {minAmpMSE(fp.x_band, x_est)}')
        
    
    mse_cpx = exclude_exception(x=mse_cpx, x_max=20).mean()
    
    print(mse_ang)
    mse_ang = exclude_exception(x=mse_ang, x_max=20).mean()
    # mse_amp = exclude_exception(x=mse_amp, x_max=20).mean()
    
    # psnr_ang = exclude_exception(x=psnr_ang, x_min=20).mean()
    # psnr_amp = exclude_exception(x=psnr_amp, x_min=20).mean()
    
    metrics = [mse_cpx, mse_ang, mse_amp, psnr_ang, psnr_amp]
        
    return metrics

