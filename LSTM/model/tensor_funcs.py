import torch


def tensor_window_slide(old_tensor: torch.Tensor, new_tensor: torch.Tensor) -> torch.Tensor:
    """
    slides the sliding window one unit over for a batched tensor
    :param old_tensor: the old sliding window of shape (batch_size, 1, window_size)
    :param new_tensor: the new tensor to add of shape (batch_size, 1, 1)
    :return: tensor of shape (batch_size, 1, window_size)
    """
    return torch.cat((old_tensor[:, :, 1:], new_tensor.unsqueeze(1).unsqueeze(1)), dim=2)
