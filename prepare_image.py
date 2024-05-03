import numpy as np

def prepare_image(image: np.ndarray, width: int, height: int, x: int, y: int, size: int) -> tuple:
    if len(image.shape) != 3 or image.shape[0] != 1 or image.shape[1] <= 0 or image.shape[2] <= 0:
        raise ValueError("Invalid image shape (1, H, W)")

    if width < 32 or height < 32 or size < 32:
        raise ValueError("Width, height or size are less than 32")

    original_height, original_width = image.shape[1], image.shape[2]

    #resize image
    if original_height < height:
        pad_height = (height - original_height) // 2
        pad_top = pad_height
        pad_bottom = height - original_height - pad_top
        resized_im = np.pad(image, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode='edge')
    else:
        crop_top = (original_height - height) // 2
        crop_bottom = crop_top + height
        resized_im = image[:, crop_top:crop_bottom, :]

    if original_width < width:
        pad_width = (width - original_width) // 2
        pad_left = pad_width
        pad_right = width - original_width - pad_left
        resized_im = np.pad(resized_im, ((0, 0), (0, 0), (pad_left, pad_right)), mode='edge')
    else:
        crop_left = (original_width - width) // 2
        crop_right = crop_left + width
        resized_im = resized_im[:, :, crop_left:crop_right]

    #subarea
    if x < 0 or x + size > width:
        raise ValueError("Subarea exceeds the resized image width")
    if y < 0 or y + size > height:
        raise ValueError("Subarea exceeds the resized image height")
    subarea = resized_im[:, y:y + size, x:x + size]

    return (resized_im, subarea)
