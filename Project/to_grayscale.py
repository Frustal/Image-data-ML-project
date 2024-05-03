import numpy as np

def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if len(pil_image.shape) == 2:
        grayscale_im = pil_image[np.newaxis, :, :].copy()
    elif len(pil_image.shape) == 3 and pil_image.shape[2] == 3:
        norm_im = pil_image / 255.0

        R_lin = np.where(norm_im[:, :, 0] <= 0.04045,
                            norm_im[:, :, 0] / 12.92,
                            ((norm_im[:, :, 0] + 0.055) / 1.055) ** 2.4)
        G_lin = np.where(norm_im[:, :, 1] <= 0.04045,
                            norm_im[:, :, 1] / 12.92,
                            ((norm_im[:, :, 1] + 0.055) / 1.055) ** 2.4)
        B_lin = np.where(norm_im[:, :, 2] <= 0.04045,
                            norm_im[:, :, 2] / 12.92,
                            ((norm_im[:, :, 2] + 0.055) / 1.055) ** 2.4)
        Y_lin = 0.2126 * R_lin + 0.7152 * G_lin + 0.0722 * B_lin

        Y = np.where(Y_lin <= 0.0031308,
                     12.92 * Y_lin,
                     1.055 * Y_lin ** (1 / 2.4) - 0.055)

        grayscale_im = (Y * 255)

        if np.issubdtype(pil_image.dtype, np.integer):
            grayscale_im = np.round(grayscale_im).astype(pil_image.dtype)
        else:
            grayscale_im = (grayscale_im).astype(pil_image.dtype)

        grayscale_im = grayscale_im[np.newaxis, :, :]

        return grayscale_im
    else:
        raise ValueError("Invalid image shape")
