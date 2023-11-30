"""Configuration file."""
NumCate = 7

AllCate = [
    "full-body",
    "top",
    "bottom",
    "outerwear",
    "bag",
    "footwear",
    "accessory",
]

# SelectCate = [
#     "top",
#     "bottom",
#     "outerwear",
#     "bag",
#     "footwear",
# ]

SelectCate = AllCate

CateIdx = {
    SelectCate[i]: i for i in range(len(SelectCate))
}

NumPhase = 3
Phase = ["train", "val", "test"]
PhaseIdx = {"train": 0, "val": 1, "test": 2}
