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
    model.load_state_dict(torch.load('weights/PurePhase_PSTS_End-to-end.pdparams', map_location='cpu'))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.Normalize(0.5, 0.5)  # 将像素值归一化到[-1,1]，提高训练的稳定性
    ])
    transform1 = transforms.Compose([
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
        line = line.strip()
        wrapped_img = cv2.imread(os.path.join(datasetPath, "val", line, "wrapped", "wrapped.tiff"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        shift_img = cv2.imread(os.path.join(datasetPath, "val", line, "shift", "shift1.bmp"), 0) / 255

        wrapped_img = transform1(wrapped_img).squeeze().numpy()
        inputs = transform(shift_img.astype(np.float32)).unsqueeze(0)
        _, outputs = model(inputs.to(device))
        result = outputs.cpu().detach().numpy()[0][0]
        result = ((result * 0.5) + 0.5) * 2 * math.pi
        result[result < 0] = 0

        mae = abs(result - wrapped_img).mean()
        mse = ((result - wrapped_img) ** 2).mean()
        ssim = structural_similarity(result, wrapped_img, data_range=2 * np.pi)
        total_mse += mse
        total_mae += mae
        total_ssim += ssim
        MSE.append(mse)
        MAE.append(mae)
        SSIM.append(ssim)

        count += 1
        print(f"{count}:MAE: {mae}\t MSE: {mse}\t SSIM: {ssim}")
    print(f"SSIM: {total_ssim / count}\t MAE: {total_mae / count}\t MSE: {total_mse / count}")
    np.savetxt(f"MAE{total_mae / count}.txt", np.array(MAE))
    np.savetxt(f"MSE{total_mse / count}.txt", np.array(MSE))
    np.savetxt(f"SSIM{total_ssim / count}.txt", np.array(SSIM))
