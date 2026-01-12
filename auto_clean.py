import os
from pathlib import Path
from typing import Tuple

# ----------------------------
# CONFIG (edit if needed)
# ----------------------------
DATASET_DIR = Path("dataset")

# Which class to keep (single-class project)
TARGET_CLASS_ID = 0
TARGET_CLASS_NAME = "bicycle"   # will match case-insensitively
ALIAS_NAMES = {"bicycle", "Bicycle"}  # accepted string labels

# If True, delete images with missing labels
REMOVE_ORPHAN_IMAGES = True

# Dry run: if True, it will NOT delete/overwrite anything, only report
DRY_RUN = False

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ----------------------------
# Helpers
# ----------------------------
def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def parse_label_line(line: str) -> Tuple[bool, str]:
    """
    Returns (keep, new_line_or_empty).
    Supports:
      - '0 x y w h' numeric YOLO
      - 'Bicycle x y w h' name-based
    Keeps only bicycle, converts to numeric '0 ...'
    """
    line = line.strip()
    if not line:
        return (False, "")

    parts = line.split()
    if len(parts) < 5:
        return (False, "")

    first = parts[0]

    # Case 1: numeric YOLO class id
    try:
        cid = int(float(first))
        if cid != TARGET_CLASS_ID:
            return (False, "")
        # Validate 4 floats
        _ = list(map(float, parts[1:5]))
        return (True, " ".join([str(TARGET_CLASS_ID)] + parts[1:5]))
    except ValueError:
        pass

    # Case 2: name-based class
    name = first.strip()
    if name.lower() not in {n.lower() for n in ALIAS_NAMES} and name.lower() != TARGET_CLASS_NAME.lower():
        return (False, "")

    # Validate 4 floats and convert to numeric class id
    try:
        _ = list(map(float, parts[1:5]))
        return (True, " ".join([str(TARGET_CLASS_ID)] + parts[1:5]))
    except ValueError:
        return (False, "")

def clean_split(split_name: str) -> dict:
    img_dir = DATASET_DIR / "images" / split_name
    lbl_dir = DATASET_DIR / "labels" / split_name

    stats = {
        "split": split_name,
        "images_total": 0,
        "labels_total": 0,
        "labels_cleaned": 0,
        "labels_deleted_empty": 0,
        "images_deleted_no_bicycle": 0,
        "orphan_labels_deleted": 0,
        "orphan_images_deleted": 0,
        "lines_removed_non_target": 0,
        "lines_kept_target": 0,
    }

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"⚠️ Split '{split_name}' missing folders. Skipping.")
        return stats

    # Count files
    images = [p for p in img_dir.iterdir() if p.is_file() and is_image_file(p)]
    labels = [p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]

    stats["images_total"] = len(images)
    stats["labels_total"] = len(labels)

    # Build quick lookup sets (stems)
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    # 1) Remove orphan labels (no matching image)
    orphan_labels = [p for p in labels if p.stem not in image_stems]
    for lp in orphan_labels:
        stats["orphan_labels_deleted"] += 1
        if not DRY_RUN:
            lp.unlink(missing_ok=True)

    # 2) Optionally remove orphan images (no matching label)
    if REMOVE_ORPHAN_IMAGES:
        orphan_images = [p for p in images if p.stem not in label_stems]
        for ip in orphan_images:
            stats["orphan_images_deleted"] += 1
            if not DRY_RUN:
                ip.unlink(missing_ok=True)

    # Refresh lists after removals (for accuracy)
    images = [p for p in img_dir.iterdir() if p.is_file() and is_image_file(p)]
    labels = [p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]

    # 3) Clean labels: keep only bicycle lines; if none remain, delete both image and label
    for lp in labels:
        try:
            raw = lp.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            raw = []

        kept_lines = []
        for line in raw:
            keep, new_line = parse_label_line(line)
            if keep:
                kept_lines.append(new_line)
                stats["lines_kept_target"] += 1
            else:
                # Count as removed if it looked like a label line
                if line.strip():
                    stats["lines_removed_non_target"] += 1

        if len(kept_lines) == 0:
            # Delete label + image (no bicycle boxes)
            stats["labels_deleted_empty"] += 1
            img_path = None
            # Find any image with same stem
            for ext in IMG_EXTS:
                candidate = img_dir / f"{lp.stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is not None:
                stats["images_deleted_no_bicycle"] += 1
                if not DRY_RUN:
                    img_path.unlink(missing_ok=True)

            if not DRY_RUN:
                lp.unlink(missing_ok=True)
        else:
            # Overwrite cleaned label with only bicycle class=0 lines
            new_content = "\n".join(kept_lines) + "\n"
            stats["labels_cleaned"] += 1
            if not DRY_RUN:
                lp.write_text(new_content, encoding="utf-8")

    return stats

def main():
    splits = []
    # Detect available splits automatically
    for s in ["train", "val", "test"]:
        if (DATASET_DIR / "images" / s).exists():
            splits.append(s)

    if not splits:
        print("❌ No dataset splits found under dataset/images/(train|val|test). Check your folder structure.")
        return

    print("=== YOLO Auto Clean ===")
    print(f"Dataset dir: {DATASET_DIR.resolve()}")
    print(f"Target: bicycle only (class_id={TARGET_CLASS_ID})")
    print(f"DRY_RUN: {DRY_RUN}  (set DRY_RUN=False to apply changes)")
    print(f"REMOVE_ORPHAN_IMAGES: {REMOVE_ORPHAN_IMAGES}")
    print("-----------------------")

    all_stats = []
    for s in splits:
        st = clean_split(s)
        all_stats.append(st)

    print("\n=== Summary ===")
    for st in all_stats:
        print(
            f"[{st['split']}] "
            f"images_total={st['images_total']} labels_total={st['labels_total']} | "
            f"labels_cleaned={st['labels_cleaned']} | "
            f"deleted_empty_labels={st['labels_deleted_empty']} "
            f"(images_deleted_no_bicycle={st['images_deleted_no_bicycle']}) | "
            f"orphan_labels_deleted={st['orphan_labels_deleted']} "
            f"orphan_images_deleted={st['orphan_images_deleted']} | "
            f"lines_kept={st['lines_kept_target']} lines_removed={st['lines_removed_non_target']}"
        )

    print("\n✅ Done.")
    if DRY_RUN:
        print("⚠️ This was a DRY RUN. Set DRY_RUN=False at the top to actually delete/overwrite files.")

if __name__ == "__main__":
    main()
