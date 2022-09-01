from data import MVTecDataset
from models import SPADE

import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    method = 'spade'
    dataset = 'hazelnut_reduced'
    model = SPADE(k=50,backbone_name="wide_resnet50_2",)
    print(f"\n█│ Running {method} on {dataset} dataset.")
    print(  f" ╰{'─'*(len(method)+len(dataset)+23)}\n")
    train_ds, test_ds = MVTecDataset(dataset).get_dataloaders()
    print("   Training ...")
    model.fit(train_ds)
    while (True):
        print("   Testing ...")
        image_rocauc, pixel_rocauc = model.evaluate(test_ds)
