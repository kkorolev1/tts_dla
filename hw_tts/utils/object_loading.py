from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

import hw_tts.datasets
from hw_tts import batch_sampler as batch_sampler_module
from hw_tts.utils.parse_config import ConfigParser
from hw_tts.collate_fn import collate


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        if split == 'train':
            drop_last = True
        else:
            drop_last = False

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(configs.init_obj(
                ds, hw_tts.datasets, config_parser=configs, batch_expand_size=configs["trainer"]["batch_expand_size"]))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        elif "batch_sampler" in params:
            batch_sampler = configs.init_obj(params["batch_sampler"], batch_sampler_module,
                                             data_source=dataset)
            bs, shuffle = 1, False
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs * configs["trainer"]["batch_expand_size"], collate_fn=collate,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler, drop_last=drop_last
        )
        dataloaders[split] = dataloader
    return dataloaders
