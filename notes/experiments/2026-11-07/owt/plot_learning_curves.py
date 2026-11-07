"""Rebuild the A1 validation curves from the two archived training CSVs."""

import argparse
import csv
from math import isfinite
from pathlib import Path

from reportlab.graphics import renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib.colors import HexColor, white


HERE = Path(__file__).resolve().parent
TINY_CSV = HERE.parents[1] / "2026-11-01" / "full-training-20k" / "metrics.csv"
OWT_CSV = HERE / "training-metrics.csv"
OUTPUT = HERE / "learning-curves.svg"
FORMAL = HERE.parent / "autodl-formal"


def read_curve(path, tokens_per_step):
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    steps = [int(row["step"]) for row in rows]
    expected_steps = [0, 1, *range(100, 20_001, 100)]
    if steps != expected_steps:
        raise ValueError(f"Missing, duplicate, or out-of-order steps in {path.name}")
    points = []
    for step, row in zip(steps, rows):
        tokens = int(row["processed_tokens"])
        loss = float(row["val_loss"])
        if tokens != step * tokens_per_step or not isfinite(loss) or not 0 <= loss <= 11:
            raise ValueError(f"Invalid point at step {step} in {path.name}")
        points.append((tokens / 1_000_000, loss))
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true", help="Plot the AutoDL formal-size runs.")
    args = parser.parse_args()
    if args.formal:
        paths = [FORMAL / domain / "metrics.csv" for domain in ("tiny", "owt")]
        tokens_per_step, x_max, tick_spacing = 16384, 336, 80
        output = FORMAL / "learning-curves.svg"
        subtitle = "20,000 updates per model at the CS336 formal dimensions"
    else:
        paths = [TINY_CSV, OWT_CSV]
        tokens_per_step, x_max, tick_spacing = 2048, 42, 10
        output = OUTPUT
        subtitle = "20,000 updates per model on the CS336 local configuration"
    curves = [read_curve(path, tokens_per_step) for path in paths]
    colors = [HexColor("#1261A0"), HexColor("#A34A12")]
    ink = HexColor("#162230")
    secondary = HexColor("#425266")
    axes = HexColor("#8291A3")
    drawing = Drawing(720, 405)
    drawing.add(Rect(0, 0, 720, 405, strokeColor=None, fillColor=white))
    drawing.add(String(31, 373, "Validation loss during training", fontName="Helvetica-Bold", fontSize=22, fillColor=ink))
    drawing.add(String(31, 352, subtitle, fontSize=11, fillColor=secondary))

    chart = LinePlot()
    chart.x, chart.y, chart.width, chart.height = 68, 83, 585, 228
    chart.data = curves
    chart.xValueAxis.valueMin, chart.xValueAxis.valueMax = 0, x_max
    chart.xValueAxis.valueSteps = list(range(0, 5 * tick_spacing, tick_spacing))
    chart.yValueAxis.valueMin, chart.yValueAxis.valueMax = 0, 11
    chart.yValueAxis.valueSteps = [0, 2, 4, 6, 8, 10]
    for axis in (chart.xValueAxis, chart.yValueAxis):
        axis.strokeColor = axes
        axis.tickStrokeColor = axes
        axis.labelTextFormat = "%d"
        axis.labels.fontSize = 10
        axis.labels.fillColor = secondary
    chart.yValueAxis.visibleGrid = True
    chart.yValueAxis.gridStrokeColor = HexColor("#DBE3ED")
    chart.yValueAxis.gridStrokeWidth = 0.6
    chart.yValueAxis.gridStart, chart.yValueAxis.gridEnd = chart.x, chart.x + chart.width
    chart.gridFirst = True
    for index, color in enumerate(colors):
        chart.lines[index].strokeColor = color
        chart.lines[index].strokeWidth = 2.2
    drawing.add(chart)

    drawing.add(String(68, 322, "Cross-entropy (nats per token)", fontSize=10, fillColor=secondary))
    drawing.add(String(360, 51, "Training target positions (millions)", textAnchor="middle", fontSize=11, fillColor=ink))
    labels = ["TinyStories (10K vocabulary)", "OpenWebText (32K vocabulary)"]
    for curve, color, label, legend_x in zip(curves, colors, labels, [68, 310]):
        final_tokens, final_loss = curve[-1]
        x = chart.x + chart.width * final_tokens / chart.xValueAxis.valueMax
        y = chart.y + chart.height * final_loss / chart.yValueAxis.valueMax
        drawing.add(Circle(x, y, 3.0, fillColor=color, strokeColor=white, strokeWidth=0.8))
        drawing.add(String(x + 8, y - 3, f"{final_loss:.3f}", fontName="Helvetica-Bold", fontSize=10, fillColor=color))
        drawing.add(Line(legend_x, 29, legend_x + 20, 29, strokeColor=color, strokeWidth=2.2))
        drawing.add(String(legend_x + 26, 26, label, fontSize=10, fillColor=ink))
    renderSVG.drawToFile(drawing, str(output))


if __name__ == "__main__":
    main()
