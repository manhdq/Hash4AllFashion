"""Configuration file."""
NumCate = 7

CateName = [
    "full-body",
    "top",
    "bottom",
    "outerwear",
    "bag",
    "footwear",
    "accessory",
]

CateIdx = {
    cate: i for i, cate in enumerate(CateName)
}

NumPhase = 3
Phase = ["train", "val", "test"]
PhaseIdx = {"train": 0, "val": 1, "test": 2}
