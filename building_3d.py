"""
building_3d.py
==============
Simplified 3D block model of the building for visual presentation in the
dashboard and PDF report (Phase 3, Feature 3.3 — Option A from the project
plan: a scaled box with facades colour-coded by energy intensity).

This is purely illustrative — it does not feed back into any energy
calculation. Implemented with matplotlib (already a dependency) rather than
pyvista, to avoid adding a heavy new dependency for a cosmetic feature.
"""
from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import config as C

# Facade order matches azimuth convention used in solar_gain.py
_FACADES = [
    {"label": "North", "azimuth_deg": 0},
    {"label": "East",  "azimuth_deg": 90},
    {"label": "South", "azimuth_deg": 180},
    {"label": "West",  "azimuth_deg": 270},
]


def render(
    floor_area_m2: float,
    height_m: float = 12.0,
    facade_gains: dict[str, float] | None = None,
):
    """
    Render a 3D block model scaled to floor_area_m2 and height_m.

    facade_gains: optional dict mapping facade label ("North"/"East"/"South"
    /"West") to an annual solar gain value (kWh/yr) — used to colour each
    wall by relative intensity. If omitted, walls are coloured uniformly.

    Returns a matplotlib Figure (pass to st.pyplot()).
    """
    side = max(np.sqrt(floor_area_m2), 1.0)
    h = max(height_m, 1.0)

    x0, x1 = -side / 2, side / 2
    y0, y1 = -side / 2, side / 2

    # 8 corners of the box
    corners = {
        "0": (x0, y0, 0), "1": (x1, y0, 0), "2": (x1, y1, 0), "3": (x0, y1, 0),
        "4": (x0, y0, h), "5": (x1, y0, h), "6": (x1, y1, h), "7": (x0, y1, h),
    }

    faces = {
        "South": [corners["0"], corners["1"], corners["5"], corners["4"]],  # y = y0
        "East":  [corners["1"], corners["2"], corners["6"], corners["5"]],  # x = x1
        "North": [corners["2"], corners["3"], corners["7"], corners["6"]],  # y = y1
        "West":  [corners["3"], corners["0"], corners["4"], corners["7"]],  # x = x0
        "Roof":  [corners["4"], corners["5"], corners["6"], corners["7"]],
    }

    if facade_gains:
        vals = np.array([facade_gains.get(f["label"], 0.0) for f in _FACADES])
        vmax = max(vals.max(), 1.0)
        cmap = matplotlib.colormaps["YlOrRd"]
        colours = {f["label"]: cmap(facade_gains.get(f["label"], 0.0) / vmax) for f in _FACADES}
    else:
        colours = {f["label"]: (0.55, 0.65, 0.8, 1.0) for f in _FACADES}
    colours["Roof"] = (0.6, 0.6, 0.6, 1.0)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    for label, verts in faces.items():
        poly = Poly3DCollection([verts], alpha=0.9)
        poly.set_facecolor(colours[label])
        poly.set_edgecolor("black")
        ax.add_collection3d(poly)

    ax.set_xlim(x0 - side * 0.3, x1 + side * 0.3)
    ax.set_ylim(y0 - side * 0.3, y1 + side * 0.3)
    ax.set_zlim(0, h * 1.3)
    ax.set_box_aspect((1, 1, h / side if side > 0 else 1))
    ax.set_xlabel("m")
    ax.set_ylabel("m")
    ax.set_zlabel("Height (m)")
    ax.set_title(f"Building block model — {floor_area_m2:,.0f} m² footprint, {h:.0f} m high")

    if facade_gains:
        mappable = matplotlib.cm.ScalarMappable(cmap=matplotlib.colormaps["YlOrRd"])
        mappable.set_array([0, vmax])
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label="Annual solar gain (kWh/yr)")

    fig.tight_layout()
    return fig
