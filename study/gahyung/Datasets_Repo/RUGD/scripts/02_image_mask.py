from pathlib import Path

image_dir = Path(
    r"C:\Users\gahyu\RUGD\raw\RUGD\3.after join creek\image"
)
mask_dir = Path(
    r"C:\Users\gahyu\RUGD\raw\RUGD\3.after join creek\indexLabel"
)

image_paths = list(image_dir.rglob("*.png"))
mask_paths = list(mask_dir.rglob("*.png"))

print("Image folder exists:", image_dir.exists())
print("Mask folder exists :", mask_dir.exists())
print("Images:", len(image_paths))
print("Masks :", len(mask_paths))

print("\nImage example:")
for path in image_paths[:3]:
    print(path)

print("\nMask example:")
for path in mask_paths[:3]:
    print(path)