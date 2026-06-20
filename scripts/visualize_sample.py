from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.alignment import check_swc_in_volume
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.point_cloud import swc_to_skeleton_records, volume_to_point_cloud
from pointneuron.data.swc import parse_swc
from pointneuron.data.vaa3d_raw import read_header, read_volume


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local HTML point-cloud/SWC viewer for one Gold166 sample.")
    parser.add_argument("--root", default="data/gold166", help="Path to Gold166 root.")
    parser.add_argument("--sample-index", type=int, default=0, help="Index in the scanned Gold166 samples.")
    parser.add_argument("--threshold", type=int, default=0, help="Foreground threshold; voxels > threshold become points.")
    parser.add_argument("--max-points", type=int, default=8192, help="Maximum foreground points to render.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for point sampling.")
    parser.add_argument("--output", default="tmp/visualizations/sample.html", help="Output HTML file.")
    args = parser.parse_args()

    samples = scan_gold166(args.root)
    sample = samples[args.sample_index]
    if sample.volume_path is None:
        raise ValueError(f"Sample has no volume: {sample.sample_id}")

    header = read_header(sample.volume_path)
    swc = parse_swc(sample.swc_path)
    report = check_swc_in_volume(swc, header)
    if not report.is_aligned:
        print(f"sample_id: {sample.sample_id}")
        print(f"out_of_bounds_nodes: {len(report.out_of_bounds_node_ids)}")
        print("Refusing to visualize misaligned sample. Pick a clean sample or handle the label policy first.")
        return 2

    volume = read_volume(sample.volume_path)
    point_cloud = volume_to_point_cloud(
        volume,
        threshold=args.threshold,
        max_points=args.max_points,
        seed=args.seed,
    )
    skeleton = swc_to_skeleton_records(swc)
    html = render_html(
        sample_id=sample.sample_id,
        volume_dimensions=point_cloud.volume_dimensions,
        total_foreground_count=point_cloud.total_foreground_count,
        points=[
            [point.x, point.y, point.z, point.intensity]
            for point in point_cloud.points
        ],
        skeleton=[
            [node.node_id, node.x, node.y, node.z, node.radius, node.parent_id]
            for node in skeleton
        ],
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"sample_id: {sample.sample_id}")
    print(f"rendered_points: {len(point_cloud.points)}")
    print(f"skeleton_nodes: {len(skeleton)}")
    print(f"output: {output}")
    return 0


def render_html(
    sample_id: str,
    volume_dimensions: tuple[int, int, int, int],
    total_foreground_count: int,
    points: list[list[float]],
    skeleton: list[list[float]],
) -> str:
    payload = {
        "sampleId": sample_id,
        "volumeDimensions": volume_dimensions,
        "totalForegroundCount": total_foreground_count,
        "points": points,
        "skeleton": skeleton,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PointNeuron Sample Viewer</title>
  <style>
    html, body {{
      margin: 0;
      height: 100%;
      overflow: hidden;
      background: #101318;
      color: #e7eef7;
      font-family: Segoe UI, Arial, sans-serif;
    }}
    #viewer {{
      display: block;
      width: 100vw;
      height: 100vh;
      cursor: grab;
    }}
    #viewer:active {{
      cursor: grabbing;
    }}
    #hud {{
      position: fixed;
      top: 12px;
      left: 12px;
      max-width: min(520px, calc(100vw - 24px));
      padding: 10px 12px;
      background: rgba(16, 19, 24, 0.82);
      border: 1px solid rgba(231, 238, 247, 0.16);
      border-radius: 6px;
      font-size: 13px;
      line-height: 1.45;
      backdrop-filter: blur(6px);
    }}
    #hud strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 14px;
    }}
    #legend {{
      margin-top: 6px;
      color: #aab8c8;
    }}
  </style>
</head>
<body>
<canvas id="viewer"></canvas>
<div id="hud">
  <strong id="title"></strong>
  <div id="meta"></div>
  <div id="legend">Drag to rotate. Wheel to zoom. Gray: sampled foreground voxels. Red: selected SWC skeleton.</div>
</div>
<script>
const DATA = {payload_json};
const canvas = document.getElementById("viewer");
const ctx = canvas.getContext("2d");
const title = document.getElementById("title");
const meta = document.getElementById("meta");

title.textContent = DATA.sampleId;
meta.textContent = `volume ${{DATA.volumeDimensions.join(" x ")}}, rendered points ${{DATA.points.length.toLocaleString()}}, foreground voxels ${{DATA.totalForegroundCount.toLocaleString()}}, skeleton nodes ${{DATA.skeleton.length.toLocaleString()}}`;

let width = 0;
let height = 0;
let yaw = -0.65;
let pitch = 0.55;
let zoom = 1.0;
let dragging = false;
let lastX = 0;
let lastY = 0;

const dims = DATA.volumeDimensions;
const center = [dims[0] / 2, dims[1] / 2, dims[2] / 2];
const scaleBase = 0.82 / Math.max(dims[0], dims[1], dims[2] * 4);
const skeletonById = new Map(DATA.skeleton.map(node => [node[0], node]));

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  width = window.innerWidth;
  height = window.innerHeight;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = width + "px";
  canvas.style.height = height + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function project(point) {{
  const zStretch = 4.0;
  let x = point[0] - center[0];
  let y = point[1] - center[1];
  let z = (point[2] - center[2]) * zStretch;

  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);

  const x1 = x * cy - z * sy;
  const z1 = x * sy + z * cy;
  const y1 = y * cp - z1 * sp;
  const z2 = y * sp + z1 * cp;

  const scale = Math.min(width, height) * scaleBase * zoom;
  return [
    width / 2 + x1 * scale,
    height / 2 + y1 * scale,
    z2,
  ];
}}

function draw() {{
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101318";
  ctx.fillRect(0, 0, width, height);

  const projectedPoints = DATA.points.map(point => {{
    const screen = project(point);
    return [screen[0], screen[1], screen[2], point[3]];
  }}).sort((a, b) => a[2] - b[2]);

  for (const point of projectedPoints) {{
    const alpha = Math.max(0.20, Math.min(0.85, point[3] / 180));
    ctx.fillStyle = `rgba(210, 220, 232, ${{alpha}})`;
    ctx.fillRect(point[0], point[1], 1.4, 1.4);
  }}

  ctx.lineWidth = 1.15;
  ctx.strokeStyle = "rgba(255, 92, 92, 0.78)";
  ctx.beginPath();
  for (const node of DATA.skeleton) {{
    const parentId = node[5];
    if (parentId < 0 || !skeletonById.has(parentId)) continue;
    const parent = skeletonById.get(parentId);
    const a = project([node[1], node[2], node[3]]);
    const b = project([parent[1], parent[2], parent[3]]);
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
  }}
  ctx.stroke();

  ctx.fillStyle = "rgba(255, 116, 116, 0.95)";
  for (const node of DATA.skeleton) {{
    const p = project([node[1], node[2], node[3]]);
    ctx.beginPath();
    ctx.arc(p[0], p[1], 1.7, 0, Math.PI * 2);
    ctx.fill();
  }}
}}

canvas.addEventListener("mousedown", event => {{
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
}});
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", event => {{
  if (!dragging) return;
  const dx = event.clientX - lastX;
  const dy = event.clientY - lastY;
  lastX = event.clientX;
  lastY = event.clientY;
  yaw += dx * 0.008;
  pitch = Math.max(-1.35, Math.min(1.35, pitch + dy * 0.008));
  draw();
}});
canvas.addEventListener("wheel", event => {{
  event.preventDefault();
  zoom *= event.deltaY < 0 ? 1.08 : 0.92;
  zoom = Math.max(0.25, Math.min(8, zoom));
  draw();
}}, {{ passive: false }});
window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

