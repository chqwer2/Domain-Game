from PIL import Image
from biasfield_interpolate_cchen.adv_bias import AdvBias, rescale_intensity
import torch.fft
import matplotlib.pyplot as plt
import numpy as np
from fft import get_fft, get_ifft, get_low_filter


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


blend_grid_size = 24   #  24*2=48, 1/4 of image size
blend_epsilon   = 0.3  # 0.3

img = Image.open('./asset/x2.png')  # Replace 'path_to_your_image.jpg' with the actual path to your image
# img = img[]



img_o = np.asarray(img)[:200, :200, :3]
img = np.transpose(img_o, (2, 0, 1))

print("img:", img.shape)
img = np.expand_dims(np.array(img), 0)
img = np.mean(img, axis=1, keepdims=True, dtype=np.double)


blender_cofig = {
    'epsilon': blend_epsilon,
    'xi': 1e-6,
    'control_point_spacing': [blend_grid_size, blend_grid_size],
    'downscale': 2,  #
    'data_size': [1, 1, img.shape[-2], img.shape[-1]],
    'interpolation_order': 3,   # 2
    'init_mode': 'gaussian',
    'space': 'log'
}
blender_node = AdvBias(blender_cofig, use_gpu=False)
blender_node.init_parameters()

nb_gin = 20
gin_out_nc = 1       # fit into the network
gin_n_interm_ch = 2
gin_nlayer = 4 # 4
gin_norm = "frob"

from imagefilter import GINGroupConv
img_transform_node = GINGroupConv(in_channel= gin_out_nc, out_channel=gin_out_nc, n_layer=gin_nlayer,
                                  interm_channel=gin_n_interm_ch,
                                  out_norm=gin_norm, use_gpu=False)

img_transform_dwt = GINGroupConv(in_channel= gin_out_nc, out_channel=gin_out_nc,
                                 n_layer=gin_nlayer,
                                  interm_channel=gin_n_interm_ch,
                                  out_norm=gin_norm, use_gpu=False)


img_tensor = torch.from_numpy(img).to(torch.double)


print("img_tensor = ", img_tensor.dtype)

revert = np.random.random() > 0.1

if revert:
    print("revert!!")

    input_buffer0 = 1 - img_transform_node(1 - img_tensor)
    input_buffer1 = img_transform_node(img_tensor)
    input_buffer = torch.cat([input_buffer0, input_buffer1], dim=0)
else:
    input_buffer = torch.cat([img_transform_node(img_tensor) for ii in range(2)], dim=0)

# amp, phase = get_fft(img_tensor)
amp, phase = get_fft(img_tensor)

amp_buffer = torch.cat([img_transform_node(amp) for ii in range(2)], dim=0)



from pytorch_wavelets import DWTForward, DWTInverse
xfm = DWTForward(J=3, wave='db3', mode='zero')
ifm = DWTInverse(wave='db3', mode='zero')



img_tensor = img_tensor


# Wavelet Transform
Yl, Yh = xfm(img_tensor.float())
# Yh[0 ~ 2]

Yl = img_transform_dwt(Yl.double())

# h, c1, c2, c3, c4 = Yh[0].shape
# print(Yl.shape, Yh[0].shape, (Yh[0].view(h*c2, c1, c3, c4)).shape)
# Yh[0] = img_transform_dwt(Yh[0].view(h*c2, c1, c3, c4).double()).view(h, c1, c2, c3, c4).float()
# h, c1, c2, c3, c4 = Yh[1].shape
# Yh[1] = img_transform_dwt(Yh[1].view(h*c2, c1, c3, c4).double()).view(h, c1, c2, c3, c4).float()
# h, c1, c2, c3, c4 = Yh[2].shape
# Yh[2] = img_transform_dwt(Yh[2].view(h*c2, c1, c3, c4).double()).view(h, c1, c2, c3, c4).float()

Y = ifm((Yl.float(), Yh))




blender_node.init_parameters()
blend_mask = rescale_intensity(blender_node.bias_field).repeat(1, gin_out_nc, 1, 1)
b, c, h, w = input_buffer.shape
b = b //2

amp_buffer0 = amp_buffer[0].clone().detach() * blend_mask + \
              amp_buffer[1].clone().detach() * (1.0 - blend_mask)

amp_buffer1 = amp_buffer[1].clone().detach() * blend_mask + \
              amp_buffer[0].clone().detach() * (1.0 - blend_mask)


array = get_low_filter(0.97, amp.shape[-2], amp.shape[-1])
amp_buffer[0] = array * amp_buffer0 + (1 - array) * amp
amp_buffer[1] = array * amp_buffer1 + (1 - array) * amp



# array = 1 - get_low_filter(0.9, amp.shape[-2], amp.shape[-1])
# amp_buffer[0] = array * amp_buffer0 + (1 - array) * amp
# amp_buffer[1] = array * amp_buffer1 + (1 - array) * amp

# amp, phase = get_fft(img_tensor)
fft1 = get_ifft(amp_buffer[0], phase).numpy().astype(np.uint8)[0, 0]
fft2 = get_ifft(amp_buffer[1], phase).numpy().astype(np.uint8)[0, 0]



blender_node.init_parameters()
blend_mask = rescale_intensity(blender_node.bias_field).repeat(1, gin_out_nc, 1, 1)
b, c, h, w = input_buffer.shape
b = b //2

print("input_buffer cp1 = ", input_buffer.shape, blend_mask.shape, b)


blend_mask = blend_mask[..., :h, :w]
# spatially-variable blending
input_cp1 = input_buffer[: b].clone().detach() * blend_mask + \
            input_buffer[b: b * 2].clone().detach() * (1.0 - blend_mask)
        
input_cp2 = input_buffer[: b].clone().detach() * (1.0 - blend_mask) + \
            input_buffer[b: b * 2].clone().detach() * blend_mask

a = input_cp1[0, 0]
input_cp1 = input_cp1[0, 0].detach().numpy()
input_cp2 = input_cp2[0, 0].detach().numpy()



# Display the original and restored images
fig, axs = plt.subplots(1, 9, figsize=(10, 5))
axs[0].imshow(1 - img_o, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')


axs[1].imshow(input_cp1, cmap='gray')
axs[1].set_title('Restored Image')
axs[1].axis('off')


axs[2].imshow(input_cp2, cmap='gray')
axs[2].set_title('amplitude')
axs[2].axis('off')
#
#
axs[3].imshow(fft1, cmap='gray')  # , cmap='rgb'
axs[3].set_title('phase')
axs[3].axis('off')
#
#
axs[4].imshow(fft2, cmap='gray')  # , cmap='rgb'
axs[4].set_title('amplitude centered')
axs[4].axis('off')


axs[5].imshow(input_buffer[0, 0].numpy(), cmap='gray')  # , cmap='rgb'
axs[5].set_title('amplitude centered')
axs[5].axis('off')




amp, phase = get_fft(img_tensor)
amp_cp1, phase_cp1 = get_fft(a)

fft3 = get_ifft(amp, phase_cp1).numpy().astype(np.uint8)[0, 0]
fft4 = get_ifft(amp_cp1, phase).numpy().astype(np.uint8)[0, 0]



axs[6].imshow(fft3, cmap='gray')  # , cmap='rgb'
axs[6].set_title('amplitude centered')
axs[6].axis('off')

axs[7].imshow(fft4, cmap='gray')  # , cmap='rgb'
axs[7].set_title('amplitude centered')
axs[7].axis('off')

axs[8].imshow(Y[0,0], cmap='gray')  # , cmap='rgb'
axs[8].set_title('DWT')
axs[8].axis('off')

plt.show()




