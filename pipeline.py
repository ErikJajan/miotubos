"""Construye manifiestos reproducibles para segmentación de miotubos."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 2026
SERIES = re.compile(r"^[0-9a-f]{8}-([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})_")
UNKNOWN = "origen-no-confirmado"
FIELDS = ("image_id", "image", "mask", "width", "height", "instances", "group_id")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def coco_to_csv(root, destination):
    """Exporta una fila por imagen anotada, incluidas las de máscara vacía."""
    data = json.loads((root / "coco/annotations.json").read_text(encoding="utf-8"))
    ids = {item["id"] for item in data["images"]}
    if len(ids) != len(data["images"]):
        raise ValueError("COCO contiene identificadores de imagen repetidos")
    if any(item["image_id"] not in ids for item in data["annotations"]):
        raise ValueError("COCO contiene anotaciones sin imagen")
    if len(data["categories"]) != 1:
        raise ValueError("La tarea requiere una sola categoría COCO de primer plano")
    if any(item["category_id"] != data["categories"][0]["id"] for item in data["annotations"]):
        raise ValueError("La anotación contiene una categoría distinta a la declarada")
    instances = Counter(item["image_id"] for item in data["annotations"])
    rows = []
    for item in sorted(data["images"], key=lambda image: image["file_name"]):
        path = Path(item["file_name"])
        if path.name != item["file_name"]:
            raise ValueError(f"COCO contiene una ruta no esperada: {path}")
        match = SERIES.match(path.stem)
        rows.append(dict(
            image_id=str(item["id"]), image=f"images/{path.name}",
            mask=f"masks_png/{path.stem}.png", width=item["width"], height=item["height"],
            instances=instances[item["id"]], group_id=match[1] if match else UNKNOWN,
        ))
    write_csv(destination, rows, FIELDS)


def read_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Columnas faltantes en {path}: {sorted(missing)}")
        rows = [{field: row[field] for field in FIELDS} for row in reader]
    if not rows or any(not row[field] for row in rows for field in FIELDS):
        raise ValueError("CSV vacío o con campos obligatorios faltantes")
    for field in ("image_id", "image", "mask"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"Valores repetidos en {field}")
    return sorted(rows, key=lambda row: row["image"])


def image_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Archivo inválido o ausente: {relative}")
    return path


def fingerprint(image):
    thumbnail = np.asarray(image.resize((33, 32), Image.Resampling.BILINEAR))
    dhash = np.packbits(thumbnail[:, 1:] > thumbnail[:, :-1]).tobytes()
    return hashlib.sha256(image.tobytes()).digest(), int.from_bytes(dhash, "big")


def audit_and_split(root, source_csv, out, seed=SEED, val_fraction=0.20):
    if not 0 < val_fraction < 1:
        raise ValueError("La fracción de validación debe estar entre 0 y 1")
    rows = read_manifest(source_csv)
    parents = {row["group_id"]: row["group_id"] for row in rows}

    def representative(group):
        while parents[group] != group:
            group = parents[group]
        return group

    fingerprints = []
    duplicates = []
    stats = []
    for row in rows:
        size = int(row["width"]), int(row["height"])
        with Image.open(image_path(root, row["mask"])) as mask:
            levels = mask.histogram()
            if mask.mode != "L" or mask.size != size or sum(levels[2:255]) or (levels[1] and levels[255]):
                raise ValueError(f"Máscara inválida: {row['mask']}")
        positive = levels[1] + levels[255]
        if bool(positive) != (int(row["instances"]) > 0) or int(row["instances"]) < 0:
            raise ValueError(f"Desacuerdo entre máscara y anotaciones: {row['image']}")

        with Image.open(image_path(root, row["image"])) as image:
            if image.mode != "L" or image.size != size:
                raise ValueError(f"Imagen incompatible: {row['image']}")
            pixel_hash, dhash = fingerprint(image)
            histogram = np.asarray(image.histogram())
        cdf = histogram.cumsum()
        p05, p95 = np.searchsorted(cdf, (0.05 * cdf[-1], 0.95 * cdf[-1]))
        stats.append(dict(image_id=row["image_id"], positive=positive, pixels=size[0] * size[1],
                          p05=int(p05), p95=int(p95)))

        for prior, prior_size, prior_hash, prior_dhash in fingerprints:
            if size != prior_size:
                continue
            distance = (dhash ^ prior_dhash).bit_count()
            exact = pixel_hash == prior_hash
            # Unir candidatos muy parecidos aunque sus nombres sugieran series distintas.
            if exact or distance <= 24:
                a, b = sorted((representative(row["group_id"]), representative(prior["group_id"])))
                parents[b] = a
                duplicates.append(dict(
                    image_a=prior["image"], image_b=row["image"], exact=exact, dhash_distance=distance,
                ))
        fingerprints.append((row, size, pixel_hash, dhash))

    original_groups = len(parents)
    train_only = representative(UNKNOWN) if UNKNOWN in parents else None
    for row in rows:
        row["group_id"] = representative(row["group_id"])
    counts = Counter(row["group_id"] for row in rows)
    groups = sorted(set(counts) - {train_only})
    if len(counts) < 2 or not groups:
        raise ValueError("Se necesitan grupos distintos para entrenamiento y validación")

    # Buscar una partición por grupos cercana al tamaño y desbalance globales.
    rng = np.random.default_rng(seed)
    target = len(rows) * val_fraction
    by_group = {group: np.zeros(3, dtype=np.int64) for group in counts}
    for row, info in zip(rows, stats, strict=True):
        by_group[row["group_id"]] += (info["positive"], info["pixels"], info["positive"] == 0)
    total_pos, total_pixels, total_empty = sum(by_group.values())
    best_cost, val_groups = float("inf"), set()
    for _ in range(2000):
        order = list(rng.permutation(groups))
        cumulative = np.cumsum([counts[group] for group in order])
        eligible = np.flatnonzero((cumulative < len(rows)) & (cumulative > 0))
        if not len(eligible):
            continue
        index = min(eligible, key=lambda i: abs(cumulative[i] - target))
        candidates = order[:index + 1]
        pos, pixels, empty = sum(by_group[group] for group in candidates)
        if pos == 0 or pos == pixels or pos == total_pos or pixels - pos == total_pixels - total_pos:
            continue
        cost = (abs(cumulative[index] - target) / len(rows)
                + abs(pos / pixels - total_pos / total_pixels)
                + 0.15 * abs(empty / cumulative[index] - total_empty / len(rows)))
        if cost < best_cost:
            best_cost, val_groups = cost, set(candidates)
    if not val_groups:
        raise ValueError("No se encontró una partición con las dos clases en ambos conjuntos")
    train = [row for row in rows if row["group_id"] not in val_groups]
    val = [row for row in rows if row["group_id"] in val_groups]
    membership = {row["image"]: split for split, partition in (("train", train), ("val", val))
                  for row in partition}
    cross_pairs = sum(membership[p["image_a"]] != membership[p["image_b"]] for p in duplicates)
    if cross_pairs or {row["group_id"] for row in train} & val_groups:
        raise RuntimeError("Se detectó un cruce de grupos o de candidatos duplicados")

    write_csv(out / "train.csv", train, FIELDS)
    write_csv(out / "val.csv", val, FIELDS)
    write_csv(out / "image_stats.csv", stats, ("image_id", "positive", "pixels", "p05", "p95"))
    write_csv(out / "similar_pairs.csv", duplicates, ("image_a", "image_b", "exact", "dhash_distance"))
    audit = dict(seed=seed, val_fraction=val_fraction, original_groups=original_groups,
                 train_groups=len(counts) - len(val_groups), val_groups=len(val_groups),
                 exact_pairs=sum(pair["exact"] for pair in duplicates),
                 near_pairs=sum(not pair["exact"] for pair in duplicates), cross_pairs=cross_pairs,
                 train_only_group=train_only)
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("vt/vt"))
    parser.add_argument("--out", type=Path, default=Path("resultados"))
    parser.add_argument("--csv", type=Path, help="Manifiesto de entrada; si se omite, exporta COCO")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    args = parser.parse_args()
    source = args.csv or args.out / "catalogo_coco.csv"
    if args.csv is None:
        coco_to_csv(args.root, source)
    print(json.dumps(audit_and_split(args.root, source, args.out, args.seed, args.val_fraction), indent=2))


if __name__ == "__main__":
    main()
