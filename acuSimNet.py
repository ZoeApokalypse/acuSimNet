import os
import torch
import datetime
import csv
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (resnet50,
                                resnet101,
                                resnet152,
                                densenet169,
                                densenet201,
                                convnext_base,
                                convnext_large,
                                vgg16,
                                vgg16_bn,
                                vgg19,
                                vgg19_bn)
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader_clean import AcuPointsDataset, create_category_encoding, get_meridian
from check_device import check_device
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else NotImplemented)
meridian_shape = {0: 10, 1: 20, 2: 12, 3: 22, 4: 18, 5: 42, 6: 31, 7: 2, 8: 17}
meridian_indices = {}
start_idx = 0
for mid in sorted(meridian_shape.keys()):
    size = meridian_shape[mid]
    meridian_indices[mid] = (start_idx, start_idx + size - 1)
    start_idx += size


def soft_wing_loss(delta, omega=1.0, epsilon=0.01, theta=1.0):
    delta = torch.clamp(delta, min=0.0, max=1e3)
    A = omega * (1/(epsilon + 1e-8) - theta + 1)
    case1_mask = (delta < theta).float()
    loss = case1_mask * omega * torch.log(1 + delta/(epsilon + 1e-8)) + \
           (1 - case1_mask) * (A * delta - omega * theta)
    return loss

def binary_focal_loss_with_logits(
        logits: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1e-8
) -> torch.Tensor:
    if not torch.all((target == 0) | (target == 1)):
        raise ValueError("Target values must be 0 or 1.")

    logits = logits.clamp(min=-100, max=100)
    p = torch.sigmoid(logits)
    p = torch.clamp(p, eps, 1.0 - eps)

    y = target.float()
    p_t = y * p + (1 - y) * (1 - p)
    alpha_t = y * alpha + (1 - y) * (1 - alpha)

    loss = -alpha_t * torch.pow(1 - p_t, gamma) * torch.log(p_t)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


class Config:
    IMAGE_SHAPE = 512
    KEYPOINTS = 174
    MAX_EPOCH = 300
    BATCH_SIZE = 36
    TAU_LOW = 0.075
    TAU_HIGH = 0.925
    MIN_WEIGHT = 0.075
    W_MIN = 0.03
    A_EPOCH = int(MAX_EPOCH / 6) # ini 10
    B_EPOCH = int(MAX_EPOCH / 1.2) # ini 2.5
    INIT_LR = 5e-5
    WEIGHT_DECAY = 1e-5

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base = densenet201(pretrained=True)
        self.features = base.features.to(device)
        self.feature_dim = base.classifier.in_features
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ).to(device)

    def forward(self, x):
        x = self.features(x)  # [B, 1920, 16, 16]
        return self.global_pool(x)  # [B, 1920]

class MeridianLayer(nn.Module):
    def __init__(self, max_landmarks, input_dim=1920):
        super().__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(1024),

            nn.Linear(1024, 768),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.LayerNorm(768),

            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(512)
        )
        self.max_landmarks = max_landmarks
        self.vis_head = nn.Linear(512, max_landmarks)
        self.coord_head = nn.Linear(512, max_landmarks * 2)
        self.cls_head = nn.Linear(512, max_landmarks * max_landmarks)
        initial_log_vars = torch.tensor([0.0, -5.0, 1.0])
        self.log_vars = nn.Parameter(initial_log_vars)

    def forward(self, x):
        shared = self.shared_fc(x)
        cls_output = self.cls_head(shared).view(-1, self.max_landmarks, self.max_landmarks)
        return {
            'vis': self.vis_head(shared),
            'coord': self.coord_head(shared),
            'cls': cls_output
        }

class VisibilityMaskGenerator:
    def __init__(self, tau_low=Config.TAU_LOW,
                 tau_high=Config.TAU_HIGH,
                 min_weight=Config.MIN_WEIGHT):
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.min_weight = min_weight

    def __call__(self, vis_pred):
        weights = (vis_pred.sigmoid() - self.tau_low) / (self.tau_high - self.tau_low)
        weights = torch.clamp(weights, 0, 1)
        mask = (vis_pred.sigmoid() >= self.tau_low) & (vis_pred.sigmoid() < self.tau_high)
        weights = torch.where(mask, torch.clamp(weights, min=self.min_weight), weights)
        return weights

class acuSimNetLoss(nn.Module):
    def __init__(self, meridian_sizes,
                 w_min=Config.W_MIN,
                 a=Config.A_EPOCH,
                 b=Config.B_EPOCH):
        super().__init__()
        self.meridian_sizes = meridian_sizes
        self.vis_mask_gen = VisibilityMaskGenerator()
        self.w_min = w_min
        self.a, self.b = a, b

    def decaying_weight(self, epoch):
        epoch_tensor = torch.tensor(float(epoch), dtype=torch.float32)
        scale = 4.0 / (self.b - self.a)
        shifted_epoch = epoch_tensor - (self.a + self.b) / 2.0
        tanh_arg = scale * shifted_epoch
        weight = self.w_min + (1 - self.w_min) * (1 - torch.tanh(tanh_arg)) / 2.0
        return weight

    def forward(self, preds, targets, current_epoch):
        total_loss = 0.0
        for meridian_id in range(9):
            meridian_pred = preds[meridian_id]
            meridian_target = self._get_meridian_targets(targets, meridian_id)
            vis_loss = self._calc_visibility_loss(meridian_pred, meridian_target)
            coord_loss = self._calc_coordinate_loss(meridian_pred, meridian_target, current_epoch)
            cls_loss = self._calc_classification_loss(meridian_pred, meridian_target)
            log_vars = preds[meridian_id]['log_vars']
            sigma = torch.exp(log_vars)
            weighted_loss = (
                    vis_loss / (2 * sigma[0] ** 2) +
                    cls_loss / (2 * sigma[1] ** 2) +
                    coord_loss / (2 * sigma[2] ** 2) +
                    torch.sum(log_vars)
            )
            total_loss += weighted_loss
        return total_loss

    def _get_meridian_targets(self, targets, meridian_id):
        start_idx, end_idx = meridian_indices[meridian_id]
        max_landmarks = self.meridian_sizes[meridian_id]
        vis_target = targets['mask'][:, start_idx:end_idx + 1]
        coord_cols = []
        for i in range(start_idx, end_idx + 1):
            coord_cols.extend([2 * i, 2 * i + 1])
        coord_target = targets['keypoints_2d'][:, coord_cols]
        cls_target = targets['local_index'][:, start_idx:end_idx + 1].contiguous()

        return {
            'vis': vis_target,
            'coord': coord_target,
            'cls': cls_target
        }

    def _calc_visibility_loss(self, pred, target):
        return binary_focal_loss_with_logits(
            logits=pred['vis'],
            target=target['vis'],
            alpha=0.75,
            gamma=2.0,
            reduction='sum'
        )

    def _calc_coordinate_loss(self, pred, target, epoch):
        pred_coord = pred['coord'].view(-1, 2)
        gt_coord = target['coord'].view(-1, 2)
        delta = torch.abs(pred_coord - gt_coord)
        loss = soft_wing_loss(delta)
        epoch_tensor = torch.tensor(epoch, dtype=torch.float32, device=delta.device)
        zero_threshold = 1e-6
        is_zero = (gt_coord.abs() < zero_threshold).all(dim=-1)
        weight_condition = (epoch_tensor > 1) & is_zero
        weight = torch.where(
            weight_condition,
            self.decaying_weight(epoch_tensor),
            torch.ones_like(gt_coord[:, 0])
        ).unsqueeze(-1)
        return (loss * weight).sum()

    def _calc_classification_loss(self, pred, target):
        return F.cross_entropy(
            pred['cls'].reshape(-1, pred['cls'].size(-1)),
            target['cls'].reshape(-1).long(),
            reduction='sum'
        )

class AcuSimNet(nn.Module):
    def __init__(self, meridian_sizes, feature_dim=1920):
        super().__init__()
        assert meridian_sizes == {0: 10, 1: 20, 2: 12, 3: 22, 4: 18, 5: 42, 6: 31, 7: 2, 8: 17}
        self.backbone = Backbone()
        self.feature_dim = feature_dim
        self.meridian_layers = nn.ModuleList([
            MeridianLayer(
                max_landmarks=size,
                input_dim=self.backbone.feature_dim
            ) for size in meridian_sizes.values()
        ])
        self.meridian_sizes = meridian_sizes
        assert len(self.meridian_layers) == 9

    def forward(self, x):
        features = self.backbone(x)
        assert features.shape[1] == self.feature_dim
        outputs = []
        for i, layer in enumerate(self.meridian_layers):
            output = layer(features)
            output['log_vars'] = layer.log_vars
            outputs.append(output)
        return outputs

class TrainingMetrics:
    @staticmethod
    def compute_visibility_accuracy(vis_pred, vis_gt, meridian_mask):
        with torch.no_grad():
            valid_mask = meridian_mask.bool()
            expanded_pred = torch.zeros_like(vis_gt).float()
            expanded_pred[valid_mask] = vis_pred.view(-1)[:valid_mask.sum()]
            pred_labels = (expanded_pred.sigmoid() > 0.5).float()
            correct = (pred_labels[valid_mask] == vis_gt[valid_mask]).sum()
            total = valid_mask.sum().clamp(min=1e-6)
            return (correct / total).item()

    @staticmethod
    def compute_cls_accuracy(cls_pred, cls_gt, vis_mask, meridian_sizes):
        batch_acc = []
        for b in range(cls_gt.size(0)):
            correct = 0
            total = 0
            # 遍历每个样本的所有穴位
            for global_idx in range(cls_gt.size(1)):
                if vis_mask[b, global_idx] == 0:
                    continue
                meridian_id = targets['categories'][b, global_idx].item()
                local_gt = cls_gt[b, global_idx].item()
                pred = cls_pred[meridian_id][b]
                _, pred_label = torch.max(pred, dim=0)
                if pred_label.item() == local_gt:
                    correct += 1
                total += 1
            if total > 0:
                batch_acc.append(correct / total)
        return sum(batch_acc) / len(batch_acc) if batch_acc else 0.0

    @staticmethod
    def compute_coord_loss(pred_coords, gt_coords, vis_mask):
        visible_coords = gt_coords[vis_mask == 1]
        pred_visible = pred_coords[vis_mask == 1]
        if len(visible_coords) == 0:
            return 0.0
        return F.l1_loss(pred_visible, visible_coords).item()

    @staticmethod
    def compute_cls_accuracy_for_meridian(pred, target):
        with torch.no_grad():
            pred_labels = torch.argmax(pred, dim=-1)
            correct = (pred_labels == target).sum().float()
            total = target.numel()
            return (correct / total).item() if total > 0 else 0.0

    @staticmethod
    def compute_coord_loss_for_meridian(pred, target):
        with torch.no_grad():
            delta = torch.abs(pred - target)
            return soft_wing_loss(delta).mean().item()


def validate_model(model, val_loader, loss_fn, device):
    model.eval()
    total_val_loss = 0.0
    total_vis_acc = 0.0
    total_cls_acc = 0.0
    total_coord_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].to(device, non_blocking=True)
            targets = {
                'mask': batch['mask'].to(device),
                'keypoints_2d': batch['keypoints_2d'].to(device),
                'categories': batch['categories'].to(device),
                'local_index': batch['local_index'].to(device)
            }
            
            preds = model(images)
            loss = loss_fn(preds, targets, current_epoch=0)
            
            batch_vis_acc = 0.0
            batch_cls_acc = 0.0
            batch_coord_loss = 0.0
            
            for meridian_id in range(9):
                pred = preds[meridian_id]
                meridian_target = loss_fn._get_meridian_targets(targets, meridian_id)
                meridian_mask = (targets['categories'] == meridian_id)
                
                vis_acc = TrainingMetrics.compute_visibility_accuracy(
                    pred['vis'], targets['mask'], meridian_mask
                )
                cls_acc = TrainingMetrics.compute_cls_accuracy_for_meridian(
                    pred['cls'], meridian_target['cls']
                )
                coord_loss = TrainingMetrics.compute_coord_loss_for_meridian(
                    pred['coord'], meridian_target['coord']
                )
                
                batch_vis_acc += vis_acc
                batch_cls_acc += cls_acc
                batch_coord_loss += coord_loss
            
            batch_vis_acc /= 9
            batch_cls_acc /= 9
            batch_coord_loss /= 9
            
            total_val_loss += loss.item()
            total_vis_acc += batch_vis_acc
            total_cls_acc += batch_cls_acc
            total_coord_loss += batch_coord_loss
            batch_count += 1
    
    return (
        total_val_loss / batch_count,
        total_vis_acc / batch_count,
        total_cls_acc / batch_count,
        total_coord_loss / batch_count
    )


if __name__ == '__main__':

    cfg = Config()
    metrics = {'vis_acc': [], 'cls_acc': [], 'coord_loss': []}

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    base_filename = f"{script_name}_{current_time}_{cfg.A_EPOCH}_{cfg.B_EPOCH}_{cfg.TAU_LOW}"
    csv_filename = f"training_log_{base_filename}.csv"
    
    best_vis_acc = 0.0
    checkpoint_start_epoch = int(cfg.MAX_EPOCH * 3 // 4)
    best_checkpoint_path = None
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Vis_Acc', 'Train_Cls_Acc', 'Train_Coord_Loss',
                         'Val_Loss', 'Val_Vis_Acc', 'Val_Cls_Acc', 'Val_Coord_Loss',
                         'Vis_Unc', 'Cls_Unc', 'Coord_Unc'])

    length = cfg.IMAGE_SHAPE
    batch_size = cfg.BATCH_SIZE
    target_size = (length, length)
    max_keypoints = cfg.KEYPOINTS
    test_length = cfg.KEYPOINTS

    base_dir = check_device("train").dataset_path
    map_dir = os.path.join(base_dir, "map.txt")
    train_image_dir = os.path.join(base_dir, f"train/image/img_{length}")
    train_json_dir = os.path.join(base_dir, "train/label/label")
    
    val_image_dir = os.path.join(base_dir, f"temp/image/img_{length}")
    val_json_dir = os.path.join(base_dir, "temp/label/label")
    meridian_order = [
        'LI', 'ST', 'SI', 'BL', 'SJ', 'GB', 'EX', 'RN', 'DU'
    ]

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
    print(f"Using dataloader: dataloader_clean")
    category_encoding = create_category_encoding()
    num_categories = len(meridian_order)

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])

    sample_ratio = 1.0
    train_dataset = AcuPointsDataset(
        train_image_dir,
        train_json_dir,
        target_size,
        category_encoding,
        unique_acupoint_names,
        global_to_local,
        name_to_index,
        transform=transform,
        sample_ratio=sample_ratio
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_dataset = AcuPointsDataset(
        val_image_dir,
        val_json_dir,
        target_size,
        category_encoding,
        unique_acupoint_names,
        global_to_local,
        name_to_index,
        transform=transform,
        sample_ratio=1.0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Data sampling ratio: {sample_ratio} ({'Full dataset' if sample_ratio == 1.0 else f'{sample_ratio*100:.1f}% of data'})")
    print(f"Training dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")
    meridian_sizes = {0: 10, 1: 20, 2: 12, 3: 22, 4: 18, 5: 42, 6: 31, 7: 2, 8: 17}
    model = AcuSimNet(meridian_sizes).to(device)
    loss_fn = acuSimNetLoss(meridian_sizes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.INIT_LR,
        weight_decay=cfg.WEIGHT_DECAY
    )

    for epoch in range(cfg.MAX_EPOCH):
        model.train()
        epoch_loss = 0.0
        epoch_vis_acc = 0.0
        epoch_cls_acc = 0.0
        epoch_coord_loss = 0.0
        batch_count = 0

        dummy_loader = iter(train_loader)
        next(dummy_loader)
        del dummy_loader

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{cfg.MAX_EPOCH}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
            initial=0,
            mininterval=0.5
        )

        for batch_idx, batch in progress_bar:
            images = batch['images'].to(device, non_blocking=True)
            targets = {
                'mask': batch['mask'].to(device),
                'keypoints_2d': batch['keypoints_2d'].to(device),
                'categories': batch['categories'].to(device),
                'local_index': batch['local_index'].to(device)
            }
            preds = model(images)
            loss = loss_fn(preds, targets, current_epoch=epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                batch_vis_acc = 0.0
                batch_cls_acc = 0.0
                batch_coord_loss = 0.0

                for meridian_id in range(9):
                    pred = preds[meridian_id]
                    meridian_target = loss_fn._get_meridian_targets(targets, meridian_id)
                    meridian_mask = (targets['categories'] == meridian_id)

                    vis_acc = TrainingMetrics.compute_visibility_accuracy(
                        pred['vis'], targets['mask'], meridian_mask
                    )
                    cls_acc = TrainingMetrics.compute_cls_accuracy_for_meridian(
                        pred['cls'], meridian_target['cls']
                    )
                    coord_loss = TrainingMetrics.compute_coord_loss_for_meridian(
                        pred['coord'], meridian_target['coord']
                    )

                    batch_vis_acc += vis_acc
                    batch_cls_acc += cls_acc
                    batch_coord_loss += coord_loss

                batch_vis_acc /= 9
                batch_cls_acc /= 9
                batch_coord_loss /= 9

                epoch_loss += loss.item()
                epoch_vis_acc += batch_vis_acc
                epoch_cls_acc += batch_cls_acc
                epoch_coord_loss += batch_coord_loss
                batch_count += 1

            progress_bar.set_postfix({
                'Loss': f"{loss.item():.2f}",
                'Vis': f"{batch_vis_acc:.2f}",
                'Cls': f"{batch_cls_acc:.2f}",
                'Coord': f"{batch_coord_loss:.2f}"
            })

        avg_loss = epoch_loss / batch_count
        epoch_vis_acc /= batch_count
        epoch_cls_acc /= batch_count
        epoch_coord_loss /= batch_count

        vis_unc_total = 0.0
        cls_unc_total = 0.0
        coord_unc_total = 0.0

        for meridian_layer in model.meridian_layers:
            log_vars = meridian_layer.log_vars.detach().cpu().numpy()
            vis_unc_total += log_vars[0]
            cls_unc_total += log_vars[1]
            coord_unc_total += log_vars[2]

        num_meridians = len(model.meridian_layers)
        avg_vis_unc = vis_unc_total / num_meridians
        avg_cls_unc = cls_unc_total / num_meridians
        avg_coord_unc = coord_unc_total / num_meridians

        val_loss, val_vis_acc, val_cls_acc, val_coord_loss = validate_model(
            model, val_loader, loss_fn, device
        )

        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{avg_loss:.4f}",
                f"{epoch_vis_acc:.4f}",
                f"{epoch_cls_acc:.4f}",
                f"{epoch_coord_loss:.4f}",
                f"{val_loss:.4f}",
                f"{val_vis_acc:.4f}",
                f"{val_cls_acc:.4f}",
                f"{val_coord_loss:.4f}",
                f"{avg_vis_unc:.4f}",
                f"{avg_cls_unc:.4f}",
                f"{avg_coord_unc:.4f}"
            ])

        if epoch >= checkpoint_start_epoch and val_vis_acc > best_vis_acc:
            best_vis_acc = val_vis_acc
            if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                os.remove(best_checkpoint_path)
            
            best_checkpoint_path = f"best_checkpoint_{base_filename}_epoch{epoch+1}_val_vis{val_vis_acc:.4f}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_vis_acc': best_vis_acc,
                'val_metrics': {
                    'val_loss': val_loss,
                    'val_vis_acc': val_vis_acc,
                    'val_cls_acc': val_cls_acc,
                    'val_coord_loss': val_coord_loss
                },
                'config': cfg.__dict__
            }, best_checkpoint_path)
            print(f"Saved best checkpoint: {best_checkpoint_path} (Val Vis Acc: {val_vis_acc:.4f})")

        print(f"EPOCH {epoch + 1} SUMMARY: "
              f"Train - Vis: {epoch_vis_acc:.3f}, Cls: {epoch_cls_acc:.3f}, Coord: {epoch_coord_loss:.4f} | "
              f"Val - Vis: {val_vis_acc:.3f}, Cls: {val_cls_acc:.3f}, Coord: {val_coord_loss:.4f}")

    final_model_path = f"final_model_{base_filename}_epoch{cfg.MAX_EPOCH}.pth"
    torch.save({
        'epoch': cfg.MAX_EPOCH,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_vis_acc': epoch_vis_acc,
        'config': cfg.__dict__
    }, final_model_path)
    print(f"\nTraining completed! Final model saved: {final_model_path}")
    
    if best_checkpoint_path:
        print(f"Best checkpoint: {best_checkpoint_path} (Best val vis accuracy: {best_vis_acc:.4f})")
    else:
        print("No best checkpoint saved (no improvement found during last 1/4 training period)")