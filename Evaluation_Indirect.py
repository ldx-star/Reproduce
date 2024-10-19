import numpy as np
import torch
import cv2
import os
import torchvision.transforms as transforms
import math
from skimage.metrics import structural_similarity
from PSTS.PSTSNet import PSTS_Pure

if __name__ == '__main__':
    device = 'cuda:0'
    model = PSTS_Pure(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load('weights/PurePhase_PSTS_Indirect.pdparams', map_location='cpu'))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.Normalize(0.5, 0.5)  # 将像素值归一化到[-1,1]，提高训练的稳定性
    ])
    transform1= transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224),
    ])
    datasetPath = "/home/ldx/temp/PurePhase"
    with open(datasetPath + "/val_total", "r") as f:
        lines = f.readlines()

    total_ssim = 0
    total_mae = 0
    total_mse = 0

    count = 0
    SSIM, MAE, MSE = [], [], []
    for line in lines:
        line = line.strip()
        wrapped_img = cv2.imread(os.path.join(datasetPath, "val", line, "wrapped", "wrapped.tiff"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        shift_img = cv2.imread(os.path.join(datasetPath, "val", line, "shift", "shift1.bmp"), 0) / 255
        cos_sum = cv2.imread(os.path.join(datasetPath, "val", line, "cos_sum", "cos_sum.tiff"), cv2.IMREAD_UNCHANGED)
        sin_sum = cv2.imread(os.path.join(datasetPath, "val", line, "sin_sum", "sin_sum.tiff"), cv2.IMREAD_UNCHANGED)

        cos_sum = transform1(cos_sum).squeeze().numpy()
        sin_sum = transform1(sin_sum).squeeze().numpy()

        inputs = transform(shift_img.astype(np.float32)).unsqueeze(0)
        _, outputs = model(inputs.to(device))
        sin_img = outputs[:, 0:1, :, :]
        cos_img = outputs[:, 1:2, :, :]
        sin_img = ((sin_img * 0.5) + 0.5) * 255
        cos_img = ((cos_img * 0.5) + 0.5) * 255

        sin_img = sin_img.cpu().detach().numpy()[0][0]
        cos_img = cos_img.cpu().detach().numpy()[0][0]
        pha = -np.arctan2(sin_img, cos_img)
        pha2 = -np.arctan2(sin_sum, cos_sum)
        pha[pha < 0] += 2 * math.pi
        pha2[pha2 < 0] += 2 * math.pi

        indirect_phaPre1 = pha.copy()
        indirect_phaRef1 = pha2.copy()

        B = np.sqrt(sin_img ** 2 + cos_img ** 2) * 2 / 8
        B2 = np.sqrt(sin_sum ** 2 + cos_sum ** 2) * 2 / 8

        indirect_phaPre1[B < 10] = 0
        indirect_phaRef1[B2 < 10] = 0

        ssim = structural_similarity(indirect_phaPre1, indirect_phaRef1, data_range=2 * math.pi)
        mae = abs(indirect_phaPre1 - indirect_phaRef1).mean()
        mse = ((indirect_phaPre1 - indirect_phaRef1) ** 2).mean()

        total_mse += mse
        total_mae += mae
        total_ssim += ssim

        SSIM.append(ssim)
        MSE.append(mse)
        MAE.append(mae)
        count += 1
        print(f"{count}:SSIM: {ssim}\t MAE: {mae}\t MSE: {mse}")
    print(f"SSIM: {total_ssim / count}\t MAE: {total_mae / count}\t MSE: {total_mse / count}")
    np.savetxt(f"SSIM{total_ssim / count}.txt", np.array(SSIM))
    np.savetxt(f"MAE{total_mae / count}.txt", np.array(MAE))
    np.savetxt(f"MSE{total_mse / count}.txt", np.array(MSE))
