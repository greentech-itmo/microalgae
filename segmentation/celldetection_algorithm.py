import torch
import celldetection as cd
from timm import create_model
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

def segment_cells(image_path, save_path=None):

    backbone = create_model("resnet34", features_only=True, pretrained=True)
    out_channels = [ch['num_chs'] for ch in backbone.feature_info.info]
    model = cd.models.UNet(backbone=backbone, out_channels=out_channels).eval()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    img = imread(image_path)


    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)


    masks = output.squeeze().cpu().numpy()

    num_cells = np.count_nonzero(masks > 0)

    if save_path:
        plt.imsave(save_path, masks, cmap="gray")

    return masks, num_cells


masks, num_cells = segment_cells("data/raw/BG-11-1-25-02-25-20-07-27.png")
print(f"Найдено клеток: {num_cells}")
