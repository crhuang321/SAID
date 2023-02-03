import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.utils as utils
from models import SAID_Bicubic, SAID_Lanczos
from utils.bicubic_pytorch import imresize as imresize_bicubic
from utils.lanczos_pytorch import imresize as imresize_lanczos

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='SAID_Bicubic', help='Name of the model to run')
parser.add_argument('-in_dir', type=str, help='Path to the HR images to be downscaled')
parser.add_argument('-out_dir', type=str, default='results/', help='Path to store test results')
parser.add_argument('-scale', type=float, default=2.0, help='Scale factor of downscaling')
parser.add_argument('-gpu', type=int, default=0, help='ID of the GPU to be used')
args = parser.parse_args()

print('\nModel:\t\t{}'.format(args.model))
print('Scale:\t\t{}'.format(args.scale))
print('Input dir:\t{}'.format(os.path.abspath(args.in_dir)))
print('Ouput dir:\t{}'.format(os.path.abspath(args.out_dir)))

device = torch.device('cuda:' + str(args.gpu))
dataloader = DataLoader(utils.Test(args.in_dir), batch_size=1, shuffle=False)
model_file = torch.load(os.path.join('pretrained_models', args.model + '.pth'))
model = {'SAID_Bicubic': SAID_Bicubic, 'SAID_Lanczos': SAID_Lanczos}[args.model](**model_file['args']).to(device)
model.load_state_dict(model_file['sd'])
model.eval()

psnr_rec, ssim_rec, lpips_rec = utils.Averager(), utils.Averager(), utils.Averager()
psnr_ref, ssim_ref, lpips_ref = utils.Averager(), utils.Averager(), utils.Averager()
lpips = utils.LPIPS(device)

for (gt, gt_path) in tqdm(dataloader):
    img_name = os.path.basename(gt_path[0])

    gt = gt.to(device)
    gt_size = [int(i) for i in gt.shape[-2:]]
    lr_size = [int(gt_size[0] / args.scale), int(gt_size[1] / args.scale)]
    
    with torch.no_grad():
        lr, sr = model(gt, lr_size, gt_size)
    lr = utils.quantize(lr, 255).cpu()
    sr = utils.quantize(sr, 255).cpu()
    if 'Bicubic' in args.model:
        lr_ref = imresize_bicubic(gt, sides=lr_size, antialiasing=True)
    if 'Lanczos' in args.model:
        lr_ref = imresize_lanczos(gt, lr_size)
    lr_ref = utils.quantize(lr_ref, 255).cpu()
    gt = utils.quantize(gt, 255).cpu()

    psnr_rec.add(utils.calc_psnr(sr, gt, args.scale, 255, True))
    ssim_rec.add(utils.calc_ssim(sr, gt, args.scale, True))
    psnr_ref.add(utils.calc_psnr(lr, lr_ref, args.scale, 255, True))
    ssim_ref.add(utils.calc_ssim(lr, lr_ref, args.scale, True))
    
    utils.save_img(utils.tensor_to_img(gt), os.path.join(args.out_dir, 'gt/'), img_name)
    utils.save_img(utils.tensor_to_img(lr), os.path.join(args.out_dir, 'lr/'), img_name)
    utils.save_img(utils.tensor_to_img(sr), os.path.join(args.out_dir, 'sr/'), img_name)
    utils.save_img(utils.tensor_to_img(lr_ref), os.path.join(args.out_dir, 'lr_ref/'), img_name)

    lpips_rec.add(lpips.calc(os.path.join(args.out_dir, 'gt/', img_name), os.path.join(args.out_dir, 'sr/', img_name)))
    lpips_ref.add(lpips.calc(os.path.join(args.out_dir, 'lr_ref/', img_name), os.path.join(args.out_dir, 'lr/', img_name)))

print('[Rec: SR ~   GT  ] PSNR / SSIM / LPIPS: {:.2f} / {:.4f} / {:.4f}'.format(psnr_rec.item(), ssim_rec.item(), lpips_rec.item()))
print('[Ref: LR ~ LR_ref] PSNR / SSIM / LPIPS: {:.2f} / {:.4f} / {:.4f}\n'.format(psnr_ref.item(), ssim_ref.item(), lpips_ref.item()))