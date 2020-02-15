from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets.voc import VOCSegmentation
from torchvision.datasets.cityscapes import Cityscapes

from .transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, RandomFlip, ConvertMaskID


def get_voc(C, split="train"):
    if split == "train":
        transforms = Compose([
            ToTensor(),
            RandomCrop((256, 256)),
            Resize((256, 256)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transforms = Compose([
            ToTensor(),
            Resize((256, 256)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return VOCSegmentation(C['root'], download=True, image_set=split, transforms=transforms)


def get_cityscapes(C, split="train"):
    if split == "train":
        # Appendix B. Semantic Segmentation Details
        transforms = Compose([
            ToTensor(),
            RandomCrop(768),
            ConvertMaskID(Cityscapes.classes),
            RandomFlip(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transforms = Compose([
            ToTensor(),
            Resize(768),
            ConvertMaskID(Cityscapes.classes),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return Cityscapes(**C, split=split, transforms=transforms)


def get_loader(C, split, distributed):
    """
    Args:
        C (Config): C.data
        split (str): args of dataset,
                    The image split to use, ``train``, ``test`` or ``val`` if split="gtFine"
                    otherwise ``train``, ``train_extra`` or ``val`
    """
    print(C.name, C.dataset, split)
    if C.name == "cityscapes":
        dset = get_cityscapes(C.dataset, split)
    elif C.name == "pascalvoc":
        dset = get_voc(C.dataset, split)

    if split == "train":
        shuffle = True
        drop_last = False
    else:
        shuffle = False
        drop_last = False

    sampler = None
    if distributed:
        sampler = DistributedSampler(dset, shuffle=shuffle)
        shuffle = None

    return DataLoader(dset, **C.loader, sampler=sampler,
                      shuffle=shuffle, drop_last=drop_last,
                      pin_memory=True)
