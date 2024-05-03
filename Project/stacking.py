import torch

def stacking(batch_as_list: list):
    images, class_ids, class_names, image_filepaths = zip(*batch_as_list)

    stacked_images = torch.stack([torch.from_numpy(img) for img in images])
    stacked_class_ids = torch.tensor(class_ids).unsqueeze(1)

    return stacked_images, stacked_class_ids, list(class_names), list(image_filepaths)