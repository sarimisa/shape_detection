import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(".")
DATASET_DIR = BASE_DIR / "dataset"

# Class names (must match your XML labels)
CLASSES = ["cylinder", "cone", "sphere"]

VOC_SPLITS = {
    "train": DATASET_DIR / "train",
    "val": DATASET_DIR / "eval",   # Change to "val" if your folder is named val
}

YOLO_DIR = BASE_DIR / "yolo_dataset"


def convert_box(size, box):
    img_w, img_h = size
    xmin, ymin, xmax, ymax = box

    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    return x_center, y_center, width, height


def convert_split(split_name, split_path):
    images_dir = split_path / "Images"
    annotations_dir = split_path / "Annotations"

    out_images = YOLO_DIR / "images" / split_name
    out_labels = YOLO_DIR / "labels" / split_name

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    xml_files = list(annotations_dir.glob("*.xml"))

    if not xml_files:
        print(f"Warning: No XML files found in {annotations_dir}")
        return

    print(f"Converting {len(xml_files)} annotations for {split_name}...")

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.findtext("filename")
        if filename is None:
            filename = xml_file.stem + ".jpg"

        img_path = images_dir / filename

        if not img_path.exists():
            found = False
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                possible = images_dir / f"{xml_file.stem}{ext}"
                if possible.exists():
                    img_path = possible
                    filename = possible.name
                    found = True
                    break

            if not found:
                print(f"Image not found for {xml_file.name}, skipping...")
                continue

        try:
            width = int(root.find("size/width").text)
            height = int(root.find("size/height").text)
        except AttributeError:
            print(f"Missing image size in {xml_file.name}, skipping...")
            continue

        shutil.copy2(img_path, out_images / filename)

        label_path = out_labels / f"{Path(filename).stem}.txt"

        with open(label_path, "w") as f:
            for obj in root.findall("object"):

                class_name = obj.findtext("name").strip().lower()

                if class_name not in CLASSES:
                    print(f"Skipping unknown class '{class_name}' in {xml_file.name}")
                    continue

                class_id = CLASSES.index(class_name)

                box = obj.find("bndbox")

                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))

                x_center, y_center, box_width, box_height = convert_box(
                    (width, height),
                    (xmin, ymin, xmax, ymax)
                )

                f.write(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{box_width:.6f} "
                    f"{box_height:.6f}\n"
                )


def create_data_yaml():
    yaml_path = YOLO_DIR / "data.yaml"

    with open(yaml_path, "w") as f:
        f.write(f"path: {YOLO_DIR.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write("names:\n")

        for cls in CLASSES:
            f.write(f"  - {cls}\n")

    return yaml_path


def main():
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)

    print("Converting Pascal VOC dataset to YOLO format...")

    convert_split("train", VOC_SPLITS["train"])
    convert_split("val", VOC_SPLITS["val"])

    data_yaml = create_data_yaml()

    print("Dataset conversion complete.")
    print("Starting YOLOv8 training...")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=10,
        imgsz=640,
        batch=8,
        device="cpu",      # Change to device=0 if using an NVIDIA GPU
        workers=2,
        name="curvilinear_shape_detector",
    )

    print("Training complete!")
    print("Best model saved in:")
    print("runs/detect/curvilinear_shape_detector/weights/best.pt")


if __name__ == "__main__":
    main()