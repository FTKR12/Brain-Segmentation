from collections.abc import Callable, Sequence
from monai.utils import MAX_SEED, convert_to_tensor, get_seed, look_up_option, min_version, optional_import
from monai.data import Dataset, ZipDataset
from typing import IO, TYPE_CHECKING, Any, cast

from monai.transforms import (
    Compose,
    Randomizable,
    RandomizableTrait,
    Transform,
    apply_transform,
    convert_to_contiguous,
    reset_ops_id,
)

from torch.utils.data import Dataset as _TorchDataset

class OriginalArrayDataset(Randomizable, _TorchDataset):
    def __init__(
        self,
        img1: Sequence,
        img2: Sequence,
        img1_transform=None,
        img2_transform=None,
        seg=None,
        seg_transform=None,
        labels=None,
        label_transform=None,
    ) -> None:
        
        items = [(img1, img1_transform), (img2, img2_transform), (seg, seg_transform), (labels, label_transform)]
        self.set_random_state(seed=get_seed())
        datasets = [Dataset(x[0], x[1]) for x in items if x[0] is not None]
        self.dataset = datasets[0] if len(datasets) == 1 else ZipDataset(datasets)

        self._seed = 0  # transform synchronization seed

    def __len__(self) -> int:
        return len(self.dataset)

    def randomize(self, data=None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int):
        self.randomize()
        if isinstance(self.dataset, ZipDataset):
            # set transforms of each zip component
            for dataset in self.dataset.data:
                transform = getattr(dataset, "transform", None)
                if isinstance(transform, Randomizable):
                    transform.set_random_state(seed=self._seed)
        transform = getattr(self.dataset, "transform", None)
        if isinstance(transform, Randomizable):
            transform.set_random_state(seed=self._seed)
        return self.dataset[index]