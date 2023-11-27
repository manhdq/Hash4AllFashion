"""Configuration file."""
NumCate = 7
CateName = [
#    "full-body",
    "top",
    "bottom",
    "outerwear",
    "bag",
    "footwear",
#    "accessory",
]
# CateIdx = {
#     "full-body": 0,
#     "bottom": 1,
#     "top": 2,
#     "outerwear": 3,
#     "bag": 4,
#     "footwear": 5,
#     "accessory": 6,
# }
CateIdx = {
    CateName[i]: i for i in range(len(CateName))
}
print(CateIdx)

NumPhase = 3
Phase = ["train", "val", "test"]
PhaseIdx = {"train": 0, "val": 1, "test": 2}
