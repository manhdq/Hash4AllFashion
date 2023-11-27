"""Configuration file."""
NumCate = 5

# CateName = [
#     "full-body",
#     "top",
#     "bottom",
#     "outerwear",
#     "bag",
#     "footwear",
#     "accessory",
# ]

CateName = [
    "top",
    "bottom",
    "outerwear",
    "bag",
    "footwear",
]

CateIdx = {
    CateName[i]: i for i in range(len(CateName))
}

NumPhase = 3
Phase = ["train", "val", "test"]
PhaseIdx = {"train": 0, "val": 1, "test": 2}
