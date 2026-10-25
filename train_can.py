"""Small-scale YOLOX-Nano finetuning script driven by config.yaml.
Run from the YOLOX repository root. For larger-scale official YOLOX GPU training,
use the official trainer with an appropriately configured Exp instead.
"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from can_nano import Exp
from config_loader import load_config
from yolox.data.datasets import COCODataset
from yolox.data.data_augment import TrainTransform


def main():
    cfg = load_config()
    mc, dc, tc = cfg["model"], cfg["data"], cfg["training"]
    if tc["optimizer"].upper() != "SGD":
        raise ValueError("Current train_can.py supports SGD only")
    device = torch.device(tc["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but unavailable")
    torch.set_num_threads(int(tc["num_threads"]))

    exp = Exp()
    exp.num_classes = int(mc["num_classes"])
    exp.input_size = tuple(mc["input_size"])
    exp.test_size = tuple(mc["input_size"])
    model = exp.get_model().to(device)

    pretrained_path = Path(mc["pretrained_weights"])
    if pretrained_path.is_file():
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        pretrained = checkpoint["model"]
        current = model.state_dict()
        compatible = {name: value for name, value in pretrained.items()
                      if name in current and value.shape == current[name].shape}
        result = model.load_state_dict(compatible, strict=False)
        print(f"Pretrained loaded: {len(compatible)} tensors; missing: {len(result.missing_keys)}")
    else:
        raise FileNotFoundError(f"Pretrained checkpoint missing: {pretrained_path}")
    model.train()

    transform = TrainTransform(max_labels=int(dc["max_labels"]),
                               flip_prob=float(tc["flip_prob"]),
                               hsv_prob=float(tc["hsv_prob"]))
    dataset = COCODataset(data_dir=dc["directory"], json_file=dc["train_annotations"],
                          name=dc["train_images"], img_size=exp.input_size, preproc=transform)
    loader = DataLoader(dataset, batch_size=int(tc["batch_size"]), shuffle=True,
                        num_workers=int(tc["num_workers"]), drop_last=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(tc["learning_rate"]),
                                momentum=float(tc["momentum"]), weight_decay=float(tc["weight_decay"]))
    epochs = int(tc["epochs"])
    max_steps = int(tc["max_steps"])
    if epochs <= 0 or max_steps < 0:
        raise ValueError("epochs must be > 0 and max_steps >= 0")
    print(f"Train {len(dataset)} images | batch={tc['batch_size']} | epochs={epochs} | lr={tc['learning_rate']} | device={device}")
    global_step = 0
    for epoch in range(epochs):
        total_loss, n_steps = 0.0, 0
        for images, targets, _, _ in loader:
            images, targets = images.float().to(device), targets.float().to(device)
            outputs = model(images, targets)
            loss = outputs["total_loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_steps += 1
            global_step += 1
            print(f"Epoch {epoch+1}/{epochs} | Step {global_step} | Loss {loss.item():.4f}", flush=True)
            if max_steps and global_step >= max_steps:
                break
        print(f"Epoch {epoch+1} mean loss: {total_loss / n_steps:.4f}")
        if max_steps and global_step >= max_steps:
            break
    path = Path(tc["output_checkpoint"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch+1, "global_step": global_step,
                "config": cfg}, path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
