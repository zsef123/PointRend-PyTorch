from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets.voc import VOCSegmentation


def get_loader(path, batch=1):
    dset = VOCSegmentation(
        path,
        image_set="train",
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

    loader = DataLoader(
        dset, batch,
        num_workers=8, pin_memory=True
    )
    return loader


if __name__ == "__main__":
    for x, gt in dset:
        print(x.shape, gt.shape)
        break
