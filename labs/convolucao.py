import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


def load_image(path, size=(256, 256)):
    try:
        img = Image.open(path).convert('L')
    except Exception as e:
        raise ValueError(f"Inexistente: {e}")
    return T.Compose([
        T.Resize(size),
        T.ToTensor()
    ])(img).unsqueeze(0)


def apply_filter(img, kernel):
    return F.conv2d(img, kernel, padding=1)


img = load_image("lenna.png")

# Realce de bordas horizontais.
sobel_x = torch.tensor([[[-1., 0., 1.],
                         [-2., 0., 2.],
                         [-1., 0., 1.]]]).unsqueeze(0)

# Realce de bordas verticais.
sobel_y = torch.tensor([[[-1., -2., -1.],
                         [0., 0., 0.],
                         [1., 2., 1.]]]).unsqueeze(0)

# Realce de bordas abruptas em todas as direções (filtro de segunda ordem).
laplacian = torch.tensor([[[0., 1., 0.],
                           [1., -4., 1.],
                           [0., 1., 0.]]]).unsqueeze(0)

sx = apply_filter(img, sobel_x)
sy = apply_filter(img, sobel_y)
lap = apply_filter(img, laplacian)
sobel_mag = torch.sqrt(sx ** 2 + sy ** 2)

# Primeira linha de figuras
plt.figure(figsize=(12, 8))
for i, (image, title) in enumerate(zip([img, sx, sy], ['Original', 'Sobel X', 'Sobel Y'])):
    plt.subplot(2, 3, i + 1)
    plt.imshow(image.squeeze().detach().numpy(), cmap='gray')
    plt.title(title)
    plt.axis('off')
# Segunda linha de figuras
for i, (image, title) in enumerate(zip([sobel_mag, lap], ['Sobel Magnitude', 'Laplacian'])):
    plt.subplot(2, 3, 4 + i)
    plt.imshow(image.squeeze().detach().numpy(), cmap='gray')
    plt.title(title)
    plt.axis('off')
plt.subplots_adjust(left=0, right=1, wspace=0, hspace=0.1)
plt.show()

