import torch, cv2
import torch.fft
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Load an example image
img = Image.open('./asset/x2.png')  # Replace 'path_to_your_image.jpg' with the actual path to your image
img = np.array(img)


def get_fft(img):
    # Perform Fourier transform
    img_fft = torch.fft.fft2(img, dim=(-2, -1))

    amplitude = torch.log(torch.abs(img_fft) + 1)  # Log scaling the amplitude for better visualization
    phase = torch.angle(img_fft)
    amplitude_centered = torch.fft.fftshift(amplitude)


    return amplitude_centered, phase


def get_ifft(amplitude_centered, phase):
    amplitude = torch.fft.ifftshift(amplitude_centered)
    combined_fft = torch.exp(amplitude) * torch.exp(1j * phase)  # Reconstruct the complex Fourier coefficients
    img_restored = torch.fft.ifft2(combined_fft, dim=(-2, -1))
    return img_restored.real


def get_low_filter( filter_rate = 0.95, h=128, w=128):
    cy, cx = int(h / 2), int(w / 2)  # centerness
    rh, rw = int(filter_rate * cy), int(filter_rate * cx)  # filter_size
    array = torch.zeros((h, w))
    array[cy - rh:cy + rh, cx - rw:cx + rw] = 1

    return array
    # the value of center pixel is zero.
    # fft_shift_img[cy - rh:cy + rh, cx - rw:cx + rw] = 0


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Convert the image to a PyTorch tensor
img_tensor = torch.tensor(img, dtype=torch.float32)
amplitude, phase = get_fft(img_tensor)

img_restored = get_ifft(amplitude, phase)



print("img_fft shape:", amplitude.shape, phase.shape, amplitude.max(), phase.max())  # [720, 1280, 3])



# Convert the tensor back to a numpy array
restored_img = img_restored.numpy().real.astype(np.uint8)

if __name__ == "__main__":
    # Display the original and restored images
    fig, axs = plt.subplots(1, 5, figsize=(10, 5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(restored_img, cmap='gray')
    axs[1].set_title('Restored Image')
    axs[1].axis('off')



    axs[2].imshow(amplitude, cmap='gray')
    axs[2].set_title('amplitude')
    axs[2].axis('off')


    axs[3].imshow(phase, cmap='gray')  # , cmap='rgb'
    axs[3].set_title('phase')
    axs[3].axis('off')



    plt.show()

