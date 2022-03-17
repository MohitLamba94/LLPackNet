import os
import shutil
from turtle import clear
import torch
from torch.utils.data import DataLoader

from network import Net

import time
import imageio

## dataloader
import numpy as np
from torch.utils.data import Dataset
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import glob
import rawpy

class load_data(Dataset):
    """Loads the Data."""
    
    def __init__(self, mode, dry_run=False):

        assert mode in ["test"], "Only test mode allowed."

        root = '/path/to/sid_dataset/Sony/'

        self.paths_low = sorted(glob.glob(root+'short/1*_00_*.ARW'))
        self.paths_high = []
        for low_path in self.paths_low:
            low_path = low_path.split('short/')[-1].split('_00_')[0]
            self.paths_high.append(glob.glob(root+'long/*'+low_path+'*.ARW')[0])
        
        self.mode = mode
        self.dry_run = dry_run

    def __len__(self):
        if self.dry_run:
            print('mode=',self.mode,' number of images = 4')
            return 4
        else:
            training_images = len(self.paths_low)
            print('mode = ',self.mode,' number of images = ',training_images)
            return training_images

    def __getitem__(self, idx):

        img_list=[]
        flag = torch.ones(1)
        flag_01 = torch.ones(1)
        
        low_exp = float(self.paths_low[idx].split('_00_')[-1].split('s.ARW')[0])
        high_exp = float(self.paths_high[idx].split('_00_')[-1].split('s.ARW')[0])
        amp = min(high_exp/low_exp,300)

        for restrict in ['10034', '10045', '10172']:
            if restrict in self.paths_low[idx]:
                flag = 0
            else:
                flag = 1
       

        print(idx, self.paths_low[idx], self.paths_high[idx], amp, flag)

        raw = rawpy.imread(self.paths_low[idx])
        low = raw.raw_image_visible.astype(np.float32).copy()
        raw.close()
        low = (np.maximum(low - 512,0)/ (16383 - 512))
        img_list.append(torch.from_numpy(low*amp).float().unsqueeze(0))

        raw = rawpy.imread(self.paths_high[idx])
        high = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16).astype(np.float32).copy()
        raw.close()
        high = high/65535.0
        img_list.append(torch.from_numpy(high.transpose(2,0,1)))
        img_list.append(flag)
        
        return img_list

def tensor2image(tensor):
    return ((tensor[0,:,:,:].detach().cpu().numpy().transpose(1,2,0))*255.0).astype(np.uint8)

def get_basic_meterics(img,img_gt):
    img = tensor2image(img)
    img_gt = tensor2image(img_gt)

    psnr = PSNR(img,img_gt)
    ssim = SSIM(img,img_gt,multichannel=True)
    return img,img_gt,psnr,ssim
###


if __name__ ==  '__main__':
    
    save_inference_images = 'inference_images'
    inference_file = 'inference.txt'
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    shutil.rmtree(save_inference_images, ignore_errors = True)
    os.makedirs(save_inference_images)
    shutil.rmtree(inference_file, ignore_errors = True)

    
    dataloader_test = DataLoader(load_data(mode="test", dry_run=False), batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    device = torch.device("cuda")
    model = Net()
    print('\nTrainable parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model = model.to(device)
    checkpoint = torch.load('weights')
    model.load_state_dict(checkpoint['model'])
    print('Device on cuda: {}'.format(next(model.parameters()).is_cuda))

    psnr_all = 0
    ssim_all = 0
    all_count = 0

    psnr_restricted = 0
    ssim_restricted = 0
    restriction_count = 0

    with torch.no_grad():
        for imgg_numm, imgg in enumerate(dataloader_test):
            low = imgg[0].to(device)
            high = imgg[1].to(device)
            flag = imgg[2]

            model.eval()
            pred = model(low)

            pred,high,psnr,ssim = get_basic_meterics(pred,high)

            psnr_all+=psnr
            ssim_all+=ssim
            all_count+=1
            
            if flag==0:
                print('Restriction')
                psnr_restricted+=psnr
                ssim_restricted+=ssim
                restriction_count+=1
            
            imageio.imwrite(save_inference_images+'/pred_{}.jpg'.format(imgg_numm),pred)

    f = open(inference_file,'a')

    psnr_avg = psnr_all/all_count
    ssim_avg = ssim_all/all_count
    psnr_avg_restricted = (psnr_all-psnr_restricted)/(all_count-restriction_count)
    ssim_avg_restricted = (ssim_all-ssim_restricted)/(all_count-restriction_count)

    print('psnr_avg = {0:.4f}, ssim_avg = {1:.4f}, psnr_avg_restricted = {2:.4f}, ssim_avg_restricted = {3:.4f}'.format(psnr_avg,ssim_avg,psnr_avg_restricted,ssim_avg_restricted))

    print('psnr_avg = {0:.4f}, ssim_avg = {1:.4f}, psnr_avg_restricted = {2:.4f}, ssim_avg_restricted = {3:.4f}'.format(psnr_avg,ssim_avg,psnr_avg_restricted,ssim_avg_restricted), file = f)
    
    f.close()
                
                
        
    
