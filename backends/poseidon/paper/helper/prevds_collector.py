from torch.utils.data import Dataset

class PrevTimeDataset(Dataset):
    """
    一个与 base_dataset 完全平行的 Dataset。
    当请求索引 idx (对应 t1 -> t2) 时，返回 idx - 1 (对应 t1 -> t2 - 1) 的数据。
    返回的数据结构与 base_dataset 100% 相同。
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        # 获取时间映射信息
        traj_idx, t_diff, t1, t2 = self.base_dataset._idx_map(idx)
        
        # 如果是有效的跨度 (t2 > t1)，返回同一初始时间下的前一步 (idx - 1)
        if t2 > t1:
            return self.base_dataset[idx - 1]
        else:
            # 如果 t1 == t2，没有“前一步”，直接返回当前步数据
            return self.base_dataset[idx]
            
    def __getattr__(self, attr):
        # 魔法方法：如果外部代码试图访问这个类没有的属性（比如 .output_dim）
        # 自动去底层的 base_dataset 里拿，保证行为完全一致
        return getattr(self.base_dataset, attr)