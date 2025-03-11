import torch
import celldetection as cd
from timm import create_model
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

def segment_cells(image_path, save_path=None):
    """
    Сегментирует клетки на изображении с помощью celldetection (UNet) и возвращает маску и количество клеток.

    :param image_path: путь к изображению
    :param save_path: путь для сохранения маски (если None, не сохранять)
    :return: (маска сегментации, количество клеток)
    """
    # Загружаем backbone ResNet34
    backbone = create_model("resnet34", features_only=True, pretrained=True)
    # Определяем out_channels (количество карт признаков на каждом уровне)
    out_channels = [ch['num_chs'] for ch in backbone.feature_info.info]
    # Создаём U-Net
    model = cd.models.UNet(backbone=backbone, out_channels=out_channels).eval()

    # Определяем, доступен ли GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Загружаем изображение
    img = imread(image_path)

    # Преобразуем в тензор
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # Запускаем модель
    with torch.no_grad():
        output = model(img_tensor)

    # Преобразуем в numpy
    masks = output.squeeze().cpu().numpy()

    # Подсчёт клеток (упрощённо)
    num_cells = np.count_nonzero(masks > 0)

    # Сохраняем, если указано
    if save_path:
        plt.imsave(save_path, masks, cmap="gray")

    return masks, num_cells

# Пример вызова:
masks, num_cells = segment_cells("C:/Projects/microalgae/segmentation/data/raw/BG-11-1-25-02-25-20-07-27.png")
print(f"Найдено клеток: {num_cells}")
