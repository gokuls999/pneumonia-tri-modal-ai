import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader
from src.dataloaders.ct_dataset import CTCovidDataset


def main():
    root = Path("data/ct_raw/sarscov2")
    ds = CTCovidDataset(root_dir=root)
    print("Dataset size:", len(ds))

    dl = DataLoader(ds, batch_size=4, shuffle=True)

    batch = next(iter(dl))
    images, labels = batch
    print("Batch images shape:", images.shape)  # expect [4, 1, 224, 224]
    print("Batch labels:", labels)


if __name__ == "__main__":
    main()
