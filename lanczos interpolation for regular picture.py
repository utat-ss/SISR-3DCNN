import cv2
import numpy as np


def lanczos_kernel(delta, a):
  return np.where(np.abs(delta) < a, np.sinc(delta) * np.sinc(delta / a), 0)


def lanczos_interpolate(image, a, sf):
  height, width, _ = image.shape
  new_height, new_width = height * sf, width * sf

  #Getting coordinates for the new image
  row_cord, col_cord = np.meshgrid(np.arange(new_height),
                                   np.arange(new_width),
                                   indexing='ij')
  #Mapping new coordinates to old image
  row_cord_ori, col_cord_ori = (row_cord + 0.5) / sf - 0.5, (col_cord +
                                                             0.5) / sf - 0.5
  #By definition getting the nearest pixel with integer coordinates
  int_row_cord_ori, int_col_cord_ori = np.floor(row_cord_ori).astype(
      int), np.floor(col_cord_ori).astype(int)
  #lanczos kernel window
  window = np.arange(-a + 1, a + 1)
  #Getting the neighbohood of the mapped old image pixel
  row_neigh = int_row_cord_ori[..., None] + window
  col_neigh = int_col_cord_ori[..., None] + window
  #Dealing with the boundary conditions
  row_neigh_clipped = np.clip(row_neigh, 0, height - 1)
  col_neigh_clipped = np.clip(col_neigh, 0, width - 1)
  #Calculating delata by definition of Lanczos Interpolation
  delta_row = row_neigh_clipped - row_cord_ori[..., None]
  delta_col = col_neigh_clipped - col_cord_ori[..., None]
  #Calculating the Lanczos kernel
  wr = lanczos_kernel(delta_row, a)
  wc = lanczos_kernel(delta_col, a)
  #Noramlizing the kernel
  weight = wr[..., None] * wc[:, :, None, :]
  weight_sum = np.sum(weight, axis=(-2, -1))
  normalized_weight = weight / weight_sum[:, :, None, None]
  #Creating a new image
  interpolated_image = np.zeros((new_height, new_width, 3), dtype=np.float32)
  for channel in range(3):
    #Getting the neighborhood channel values from the old image
    neighbor_value = image[row_neigh_clipped[..., None],
                           col_neigh_clipped[:, :, None, :], channel]
    #Interpolation
    interpolated_image[:, :,
                       channel] = np.sum(normalized_weight * neighbor_value,
                                         axis=(-2, -1))

  return np.clip(interpolated_image, 0, 255).astype(np.uint8)


image_1 = cv2.imread('test.jpg')#you can put in whatever picture you want here by changing the input name
scaled_image_1 = lanczos_interpolate(image_1, a=3, sf=2)
success = cv2.imwrite('scaled_test.jpg', scaled_image_1)#you can change the corresponding output name here
if success:
  print("Image saved successfully.")
else:
  print("Fail to save image.")
