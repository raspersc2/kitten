import torch
from filelock import FileLock


def save_checkpoint(checkpoint_path, epoch, model, optimizer):
    lock_file = f"{str(checkpoint_path)}.lock"
    lock = FileLock(lock_file)
    with lock:
        state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        f_path = checkpoint_path
        torch.save(state, f_path)


def load_checkpoint(checkpoint_path, model, optimizer, device):
    lock_file = f"{str(checkpoint_path)}.lock"
    lock = FileLock(lock_file)
    with lock:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return model, optimizer, checkpoint["epoch"]
