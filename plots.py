"""Gráficas reproducibles del notebook, con Pillow y NumPy."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

INK = "#183344"
MUTED = "#486476"
TEAL = "#087e83"
GRID = "#deeaed"


def font(size, bold=False):
    names = ("segoeuib.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "DejaVuSans.ttf")
    for name in names:
        for folder in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")):
            if (folder / name).exists():
                return ImageFont.truetype(str(folder / name), size)
    return ImageFont.load_default(size=size)


def histogram(values, edges, xlabel, path):
    counts, _ = np.histogram(values, bins=edges)
    chart = Image.new("RGB", (800, 340), "white")
    draw = ImageDraw.Draw(chart)
    left, top, right, bottom = 87, 21, 775, 264
    ymax = max(5, int(np.ceil(counts.max() / 5) * 5))
    for count in np.linspace(0, ymax, 6, dtype=int):
        y = bottom - count / ymax * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 13, y), str(count), font=font(18), fill=MUTED, anchor="rm")
    scale = (right - left) / (edges[-1] - edges[0])
    for first, last, count in zip(edges[:-1], edges[1:], counts, strict=True):
        if count:
            x0 = left + (first - edges[0]) * scale + 2
            x1 = left + (last - edges[0]) * scale - 2
            draw.rectangle((x0, bottom - count / ymax * (bottom - top), x1, bottom), fill=TEAL)
    for value in edges[::max(1, (len(edges) - 1) // 6)]:
        x = left + (value - edges[0]) * scale
        draw.text((x, bottom + 11), f"{value:g}", font=font(18), fill=MUTED, anchor="mt")
    draw.text(((left + right) // 2, 329), xlabel, font=font(19), fill=INK, anchor="ms")
    ylabel = Image.new("RGB", (120, 27), "white")
    ImageDraw.Draw(ylabel).text((60, 13), "Imágenes", font=font(18), fill=INK, anchor="mm")
    chart.paste(ylabel.rotate(90, expand=True), (6, 81))
    chart.save(path)


def overlay_panel(root, examples, path):
    width, height = 247, 185
    canvas = Image.new("RGB", (800, 451), "white")
    draw = ImageDraw.Draw(canvas)
    for i, example in enumerate(examples):
        with Image.open(root / example["image"]) as source:
            image = source.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")
        with Image.open(root / example["mask"]) as source:
            mask = np.asarray(source.resize((width, height), Image.Resampling.NEAREST)) > 0
        base = np.asarray(image).copy()
        base[mask] = 0.45 * base[mask] + 0.55 * np.asarray([45, 232, 205])
        x, y = 9 + (i % 3) * 267, 4 + (i // 3) * 224
        canvas.paste(Image.fromarray(base), (x, y))
        draw.text((x, y + height + 6), example["caption"], font=font(17, bold=True), fill=INK)
    canvas.save(path)


def precision_recall(curve, prevalence, path):
    chart = Image.new("RGB", (800, 415), "white")
    draw = ImageDraw.Draw(chart)
    left, top, right, bottom = 87, 28, 775, 345

    def point(recall, precision):
        return left + recall * (right - left), bottom - precision * (bottom - top)

    for value in np.linspace(0, 1, 6):
        x, y = point(value, value)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        draw.text((left - 12, y), f"{value:.1f}", font=font(18), fill=MUTED, anchor="rm")
        draw.text((x, bottom + 11), f"{value:.1f}", font=font(18), fill=MUTED, anchor="mt")
    for level, color in [(0.80, "#ac621a"), (prevalence, "#7796a6")]:
        y = point(0, level)[1]
        for x in range(left, right, 20):
            draw.line((x, y, min(x + 10, right), y), fill=color, width=2)
    valid = curve["predicted"] > 0
    points = [point(x, y) for x, y in zip(curve["recall"][valid], curve["precision"][valid], strict=True)]
    draw.line(points, fill=TEAL, width=4)
    draw.text((right - 8, point(0, 0.80)[1] - 8), "Meta de precisión", font=font(17),
              fill="#ac621a", anchor="rs")
    draw.text((right - 8, point(0, prevalence)[1] - 12), "Prevalencia", font=font(17),
              fill=MUTED, anchor="rs")
    draw.text(((left + right) // 2, 406), "Recall", font=font(19), fill=INK, anchor="ms")
    ylabel = Image.new("RGB", (120, 27), "white")
    ImageDraw.Draw(ylabel).text((60, 13), "Precisión", font=font(18), fill=INK, anchor="mm")
    chart.paste(ylabel.rotate(90, expand=True), (6, 125))
    chart.save(path)
