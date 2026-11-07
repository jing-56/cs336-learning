"""Plot the recorded validation losses of the high learning-rate runs."""

import csv
from math import log10
from pathlib import Path

from reportlab.graphics import renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib.colors import HexColor, white


HERE = Path(__file__).resolve().parent
EARLIER_SWEEP = HERE.parents[1] / "2026-10-27" / "lr"
RUNS = [
    ("0.003", EARLIER_SWEEP, "#1261A0", None),
    ("1", EARLIER_SWEEP, "#37724C", [6, 3]),
    ("3", HERE, "#A34A12", [2, 3]),
    ("10", HERE, "#9C2A56", None),
]


def read_curve(rate, folder):
    path = folder / f"lr_{rate}_loss.csv"
    with path.open(encoding="utf-8", newline="") as file:
        points = [(int(row["step"]), log10(float(row["val_loss"]))) for row in csv.DictReader(file)]
    if [step for step, _ in points] != [0, 1, *range(100, 1001, 100)]:
        raise ValueError(f"Unexpected recorded steps in {path}")
    if not all(0 <= loss <= 3.5 for _, loss in points):
        raise ValueError(f"Loss outside the chart range in {path}")
    return points


def main():
    ink = HexColor("#162230")
    secondary = HexColor("#425266")
    axes = HexColor("#8291A3")
    drawing = Drawing(720, 400)
    drawing.add(Rect(0, 0, 720, 400, strokeColor=None, fillColor=white))
    drawing.add(String(30, 370, "Training stability at high learning rates",
                       fontName="Helvetica-Bold", fontSize=21, fillColor=ink))
    drawing.add(String(30, 347, "Same model, initialization and batches across runs",
                       fontSize=11, fillColor=secondary))
    drawing.add(String(78, 321, "Validation cross-entropy (log scale)", fontSize=10, fillColor=secondary))

    chart = LinePlot()
    chart.x, chart.y, chart.width, chart.height = 78, 83, 575, 226
    chart.data = [read_curve(rate, folder) for rate, folder, _, _ in RUNS]
    chart.xValueAxis.valueMin, chart.xValueAxis.valueMax, chart.xValueAxis.valueStep = 0, 1000, 200
    chart.yValueAxis.valueMin, chart.yValueAxis.valueMax = 0, 3.5
    chart.yValueAxis.valueSteps = [0, 1, 2, 3]
    chart.xValueAxis.labelTextFormat = "%d"
    chart.yValueAxis.labelTextFormat = lambda value: f"{10 ** value:,.0f}"
    for axis in (chart.xValueAxis, chart.yValueAxis):
        axis.strokeColor = axes
        axis.tickStrokeColor = axes
        axis.labels.fontSize = 10
        axis.labels.fillColor = secondary
    chart.yValueAxis.visibleGrid = True
    chart.yValueAxis.gridStrokeColor = HexColor("#DBE3ED")
    chart.yValueAxis.gridStrokeWidth = 0.6
    chart.yValueAxis.gridStart, chart.yValueAxis.gridEnd = chart.x, chart.x + chart.width
    chart.gridFirst = True
    for index, (_, _, color, dash) in enumerate(RUNS):
        chart.lines[index].strokeColor = HexColor(color)
        chart.lines[index].strokeWidth = 2
        if dash is not None:
            chart.lines[index].strokeDashArray = dash
    drawing.add(chart)
    drawing.add(String(365, 49, "Optimizer updates", textAnchor="middle", fontSize=11, fillColor=ink))
    for index, (rate, _, color, dash) in enumerate(RUNS):
        x = 78 + 144 * index
        line = Line(x, 28, x + 24, 28, strokeColor=HexColor(color), strokeWidth=2)
        if dash is not None:
            line.strokeDashArray = dash
        drawing.add(line)
        drawing.add(String(x + 30, 25, f"Peak LR {rate}", fontSize=10, fillColor=ink))
    renderSVG.drawToFile(drawing, str(HERE / "stability-curves.svg"))


if __name__ == "__main__":
    main()
