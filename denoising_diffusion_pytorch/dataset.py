import os
import glob

from PIL import Image
import torch
from torch.utils.data import Dataset
from torch import Tensor
from torchvision.transforms import AutoAugment, Compose, Grayscale, Resize, ToTensor, CenterCrop
from torch import fft


class KSpaceDataset(Dataset):
    def __init__(self, img_dir: str, target_resolution: int = 128) -> None:
        super().__init__()

        self.img_dir = img_dir
        self.img_file_paths = glob.glob(img_dir + '/**/*.JPEG', recursive=True)
        self.target_resolution = target_resolution

        self.transforms = Compose([
            Resize(target_resolution),
            CenterCrop(target_resolution),
            AutoAugment(),
            Grayscale(),
            ToTensor()
        ])

    @staticmethod
    def _generate_complex_img(amp: Tensor) -> Tensor:
        height, width = amp.shape

        # Zeros should be distributed symmetrically around the center
        # Fill rectangle in center with noise

        h_start, h_end = height // 2 - 1, height // 2 + 2
        w_start, w_end = width // 2 - 1, width // 2 + 2

        if height % 2 == 0:
            h_start -= 1

        if width % 2 == 0:
            w_start -= 1

        rand_shape = (h_end - h_start, w_end - w_start)

        noise = torch.zeros(height, width, dtype=torch.cfloat)
        noise[h_start:h_end, w_start:w_end] = torch.rand(rand_shape, dtype=torch.cfloat)

        # Imag leads to phase images which are more asymmetrical
        # than real phase images (might be coincidence)
        phase_fft = torch.imag(fft.ifft2(fft.fftshift(noise)))

        # Normalize between -pi and pi
        phase_fft = (phase_fft - phase_fft.min()) / (
                    phase_fft.max() - phase_fft.min()) * 2 * torch.pi - torch.pi

        # Apply high pass filter on amplitude
        # We recycle the boundaries from above but make them a bit wider
        amp_fft = fft.fftshift(fft.fft2(amp))
        amp_fft[h_start - 6:h_end + 6, w_start - 6:w_end + 6] = 0

        amp_hp = torch.real(fft.ifft2(fft.ifftshift(amp_fft)))
        # Normalize between [-0.5, 0.5]
        amp_hp = (amp_hp - amp_hp.min()) / (amp_hp.max() - amp_hp.min()) - 0.5

        phase_total = phase_fft + 0.5 * amp_hp

        img_complex = amp * torch.exp(1j * phase_total)
        # Normalize complex image
        img_complex /= torch.max(torch.abs(img_complex))

        return img_complex

    def __len__(self) -> int:
        return len(self.img_file_paths)

    def __getitem__(self, idx: int) -> Tensor:
        # Load image to memory as PIL Image in Grayscale
        img_path = self.img_file_paths[idx]
        img_pil = Image.open(img_path).convert('RGB')

        # Data transforms include resizing, augmentation,
        # grayscaling and [0, 1] range conversion.
        img = self.transforms(img_pil)

        # Adding a small constant for numerical stability with phase image.
        img = torch.clamp(img + 1e-4, 0, 1)

        # Generate complex image from real image.
        img_complex = KSpaceDataset._generate_complex_img(img[0])

        # Convert image to cartesian K-Space
        img_fft = fft.fftshift(fft.fft2(fft.ifftshift(img_complex)))

        # Split real and imag into channels cfloat --> float with 2 channels
        img_fft = torch.stack((img_fft.real, img_fft.imag))

        img_fft /= torch.max(torch.abs(img_fft))

        return img_fft
