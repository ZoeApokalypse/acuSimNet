import os
import json
import random
import numpy as np
import torch
from check_device import check_device
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

length = 512
batch_size = 64
target_size = (length, length)
max_keypoints = 174
test_length = 174

base_dir = check_device("dataloader").dataset_path
map_dir = os.path.join(base_dir, "map.txt")
train_image_dir = os.path.join(base_dir, f"train/image/img_{length}")
train_json_dir = os.path.join(base_dir, "train/label/label")
meridian_order = [
    'LI', 'ST', 'SI', 'BL', 'SJ', 'GB', 'EX', 'RN', 'DU'
]

def create_category_encoding():
    return {meridian: idx for idx, meridian in enumerate(meridian_order)}

def get_meridian(name):
    parts = name.split('_')
    code_part = parts[0]
    if code_part.startswith('EX'):
        return 'EX'
    for meridian in meridian_order:
        if code_part.startswith(meridian):
            return meridian
    if code_part.startswith('RN'):
        return 'RN'
    if code_part.startswith('DU'):
        return 'DU'
    raise ValueError(f"Unrecognized acupoint code: {name}")

def create_mask(keypoints):
    mask = np.all(np.isclose(keypoints, np.array([0.0, 0.0, 0.5], dtype=np.float32)), axis=-1)
    mask = np.where(mask, 0, 1)
    return mask.astype(np.int32)

def parse_json_and_map_weights(json_path, category_encoding, unique_acupoint_names, global_to_local, name_to_index):
    with open(json_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    keypoint_info = data["label"]
    keypoints = np.full((max_keypoints, 3), [0.0, 0.0, 0.5], dtype=np.float32)
    keypoints_2d = np.full((max_keypoints, 2), [0.0, 0.0], dtype=np.float32)
    weights = np.full(max_keypoints, 1.0, dtype=np.float32)
    local_indices = np.full(max_keypoints, -1, dtype=np.int32)

    categories = np.array([
        category_encoding[get_meridian(name)]
        for name in unique_acupoint_names
    ], dtype=np.int32)

    for global_idx in range(max_keypoints):
        if global_idx in global_to_local:
            local_info = global_to_local[global_idx]
            local_indices[global_idx] = local_info['local_idx']

    for point in keypoint_info:
        name = point["name"]
        if name not in name_to_index:
            continue
        k = name_to_index[name]
        meridian = get_meridian(name)

        if meridian not in category_encoding:
            raise ValueError(f"Unknown meridian type: {meridian}")

        keypoints[k] = [
            float(point["coordinate"]["x"]),
            float(point["coordinate"]["y"]),
            float(point["coordinate"]["h"])
        ]
        keypoints_2d[k] = keypoints[k][:2]
        weights[k] = 0.9 + 0.1 * (-float(point["coordinate"]["n"]))
    mask = create_mask(keypoints)

    return (
        keypoints.flatten(),
        keypoints_2d.flatten(),
        weights,
        categories,
        mask,
        local_indices
    )


class AcuPointsDataset(Dataset):
    def __init__(self, image_dir, json_dir, target_size, category_encoding, unique_acupoint_names, global_to_local, name_to_index, transform=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
        self.json_files = sorted(os.listdir(json_dir))
        self.target_size = target_size
        self.category_encoding = category_encoding
        self.unique_acupoint_names = unique_acupoint_names
        self.global_to_local = global_to_local
        self.name_to_index = name_to_index
        self.transform = transform

        self.exclude_frames = list(range(12, 30)) + list(range(46, 108)) + list(range(119, 138))
        self.image_paths, self.json_files = self.filter_frames(self.image_paths, self.json_files)

    def filter_frames(self, image_paths, json_files):
        filtered_image_paths = []
        filtered_json_files = []

        for path in image_paths:
            frame_number = int(path.split('_')[-1].split('.')[0])
            if frame_number not in self.exclude_frames:
                filtered_image_paths.append(path)

        for json_file in json_files:
            frame_number = int(json_file.split('_')[-1].split('.')[0])
            if frame_number not in self.exclude_frames:
                filtered_json_files.append(json_file)

        return filtered_image_paths, filtered_json_files

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        json_path = os.path.join(self.json_dir, self.json_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        (keypoints, keypoints_2d, weights, categories, mask, local_indices) = parse_json_and_map_weights(
            json_path, self.category_encoding, self.unique_acupoint_names, self.global_to_local, self.name_to_index
        )

        return {
            'images': image,
            'keypoints': keypoints,
            'keypoints_2d': keypoints_2d,
            'weights': weights,
            'categories': categories,
            'mask': mask,
            'local_index': local_indices
        }


if __name__ == '__main__':

    with open(map_dir, 'r', encoding='utf-8') as file:
        unique_acupoint_names = [line.strip() for line in file if line.strip()]
    name_to_index = {name: idx for idx, name in enumerate(unique_acupoint_names)}
    meridian_to_indices = {m: [] for m in meridian_order}
    global_to_local = {}

    for global_idx, name in enumerate(unique_acupoint_names):
        meridian = get_meridian(name)
        meridian_to_indices[meridian].append(global_idx)
        local_idx = len(meridian_to_indices[meridian]) - 1
        global_to_local[global_idx] = {
            'meridian': meridian,
            'local_idx': local_idx}

    meridian_sizes = {m: len(indices) for m, indices in meridian_to_indices.items()}
    print("Meridian distribution:", meridian_sizes)
    category_encoding = create_category_encoding()
    num_categories = len(meridian_order)

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    train_dataset = AcuPointsDataset(
        train_image_dir,
        train_json_dir,
        target_size,
        category_encoding,
        unique_acupoint_names,
        global_to_local,
        name_to_index,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    pos_count = 0
    total_count = 0
    for batch in train_loader:
        pos_count += batch['mask'].sum().item()
        total_count += batch['mask'].numel()
    print(pos_count)
    print(total_count)
    print(f"Global visibility positive ratio: {pos_count / total_count:.3f}")
    for batch in train_loader:
        images = batch['images']
        keypoints = batch['keypoints']
        keypoints_2d = batch['keypoints_2d']
        weights = batch['weights']
        categories = batch['categories']
        mask = batch['mask']
        local_indices = batch['local_index']

        print(f"images: {images.shape}, dtype={images.dtype}")
        print(f"keypoints: {keypoints.shape}, dtype={keypoints.dtype}")
        print(f"keypoints_2d: {keypoints_2d.shape}, dtype={keypoints_2d.dtype}")
        print(f"weights: {weights.shape}, dtype={weights.dtype}")
        print(f"categories: {categories.shape}, dtype={categories.dtype}")
        print(f"local_index: {local_indices.shape}, dtype={local_indices.dtype}")
        print(f"mask: {mask.shape}, dtype={mask.dtype}")
        break

    for batch in train_loader:
        images = batch['images']
        keypoints = batch['keypoints']
        keypoints_2d = batch['keypoints_2d']
        weights = batch['weights']
        categories = batch['categories']
        mask = batch['mask']
        local_indices = batch['local_index']

        random_idx = random.randint(0, images.size(0) - 1)
        print(f"Sample {random_idx}:")
        print(f"Image shape: {images[random_idx].shape}")
        print(f"Keypoints: {keypoints[random_idx].reshape(-1, 3)[:6]}")
        print(f"Keypoints 2D: {keypoints_2d[random_idx].reshape(-1, 2)[:6]}")
        print(f"Weights: {weights[random_idx][:test_length]}")
        print(f"Categories: {categories[random_idx][:test_length]}")
        print(f"Local Index: {local_indices[random_idx][:test_length]}")
        print(f"Mask: {mask[random_idx][:test_length]}")
        break