from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets.voc import VOCSegmentation
from torchvision.datasets.cityscapes import Cityscapes


def get_voc(C, split="train"):
    if split == "train":
        T.Compose([
            T.Resize((256, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return VOCSegmentation(
        **C,
        image_set=split,
        transform=T.Compose([
            T.Resize((256, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        target_transform=T.Compose([
            T.Resize((256, 512)),
            T.ToTensor()
        ])
    )


def get_cityscapes(C, split="train"):
    if split == "train":
        # Appendix B. Semantic Segmentation Details
        transforms = T.Compose([
            T.RandomCrop(768),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    print(C, split)
    return Cityscapes(**C, split=split, transforms=transforms)


def get_loader(C, split):
    """
    Args:
        C (Config): C.data
        split (str): args of dataset,
                    The image split to use, ``train``, ``test`` or ``val`` if split="gtFine"
                    otherwise ``train``, ``train_extra`` or ``val`
    """

    if C.name == "cityscapes":
        dset = get_cityscapes(C.dataset, split)
    elif C.name == "pascalvoc":
        dset = get_voc(C.dataset, split)

    return DataLoader(dset, **C.loader, pin_memory=True)


if __name__ == "__main__":
    for x, gt in dset:
        print(x.shape, gt.shape)
        break
