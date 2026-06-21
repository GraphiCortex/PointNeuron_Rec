from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.point_cloud import swc_to_skeleton_records, volume_to_point_cloud
from pointneuron.data.swc import parse_swc
from pointneuron.data.vaa3d_raw import read_volume


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize initialized proposal graph over a Gold166 sample.")
    parser.add_argument("--graph", required=True, help="Initialized graph .npz from scripts/initialize_proposal_graph.py.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--sample-index", type=int, help="Gold166 sample index override. Defaults to graph metadata.")
    parser.add_argument("--render-points", type=int, default=12000, help="Foreground points to render.")
    parser.add_argument("--seed", type=int, default=0, help="Foreground point sampling seed.")
    parser.add_argument("--output", default="tmp/visualizations/initialized_graph.html", help="Output HTML file.")
    args = parser.parse_args()

    graph_path = Path(args.graph)
    graph = np.load(graph_path, allow_pickle=False)
    metadata = json.loads(str(graph["metadata"])) if "metadata" in graph else {}
    proposal_metadata = metadata.get("proposal_metadata", {})
    sample_index = args.sample_index
    if sample_index is None:
        sample_index = int(proposal_metadata.get("sample_index", 0))

    samples = scan_gold166(args.root)
    sample = samples[sample_index]
    if sample.volume_path is None:
        raise ValueError(f"Sample has no volume: {sample.sample_id}")

    threshold = int(proposal_metadata.get("threshold", 0))
    volume = read_volume(sample.volume_path)
    point_cloud = volume_to_point_cloud(volume, threshold=threshold, max_points=args.render_points, seed=args.seed)
    gt_swc = parse_swc(sample.swc_path)
    init_swc_path = metadata.get("init_swc")
    init_swc = parse_swc(init_swc_path) if init_swc_path else None

    centers = graph["centers"].astype(float, copy=False)
    scores = graph["scores"].astype(float, copy=False) if "scores" in graph else np.zeros((centers.shape[0],), dtype=float)
    edges = graph["edges"].astype(int, copy=False)
    components = connected_components(centers.shape[0], edges)
    component_count = len(set(components)) if components else 0

    html = render_html(
        payload={
            "sampleId": sample.sample_id,
            "graphPath": str(graph_path),
            "initSwc": str(init_swc_path),
            "volumeDimensions": point_cloud.volume_dimensions,
            "totalForegroundCount": point_cloud.total_foreground_count,
            "points": [[point.x, point.y, point.z, point.intensity] for point in point_cloud.points],
            "gtSkeleton": skeleton_payload(gt_swc),
            "initSkeleton": skeleton_payload(init_swc) if init_swc is not None else [],
            "proposals": [
                [float(centers[index, 0]), float(centers[index, 1]), float(centers[index, 2]), float(scores[index]), int(components[index])]
                for index in range(centers.shape[0])
            ],
            "graphEdges": edges.tolist(),
            "metadata": {
                **metadata,
                "components": component_count,
            },
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"sample_id: {sample.sample_id}")
    print(f"proposal_nodes: {centers.shape[0]}")
    print(f"graph_edges: {edges.shape[0]}")
    print(f"components: {component_count}")
    print(f"init_swc: {init_swc_path}")
    print(f"output: {output_path}")
    return 0


def skeleton_payload(swc):
    if swc is None:
        return []
    return [
        [node.node_id, node.x, node.y, node.z, node.radius, node.parent_id]
        for node in swc_to_skeleton_records(swc)
    ]


def connected_components(node_count: int, edges: np.ndarray) -> list[int]:
    parent = list(range(node_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in edges.tolist():
        union(int(left), int(right))
    roots = {}
    labels = []
    for index in range(node_count):
        root = find(index)
        if root not in roots:
            roots[root] = len(roots)
        labels.append(roots[root])
    return labels


def render_html(payload: dict) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PointNeuron Initialized Graph Viewer</title>
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
    #viewer:active {{ cursor: grabbing; }}
    #hud {{
      position: fixed;
      top: 12px;
      left: 12px;
      max-width: min(760px, calc(100vw - 24px));
      padding: 10px 12px;
      background: rgba(16, 19, 24, 0.84);
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
  <div id="legend">Drag to rotate. Wheel to zoom. Gray: foreground. Red: GT SWC. Amber: init SWC. Cyan: proposal nodes. Blue: initialized graph edges.</div>
</div>
<script>
const DATA = {payload_json};
const canvas = document.getElementById("viewer");
const ctx = canvas.getContext("2d");
const title = document.getElementById("title");
const meta = document.getElementById("meta");

title.textContent = DATA.sampleId;
meta.textContent = `nodes ${{DATA.proposals.length.toLocaleString()}}, edges ${{DATA.graphEdges.length.toLocaleString()}}, components ${{DATA.metadata.components}}, foreground ${{DATA.totalForegroundCount.toLocaleString()}}, init ${{DATA.initSwc}}`;

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
const gtById = new Map(DATA.gtSkeleton.map(node => [node[0], node]));
const initById = new Map(DATA.initSkeleton.map(node => [node[0], node]));

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
  return [width / 2 + x1 * scale, height / 2 + y1 * scale, z2];
}}

function drawSkeleton(skeleton, byId, color, lineWidth) {{
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = color;
  ctx.beginPath();
  for (const node of skeleton) {{
    const parentId = node[5];
    if (parentId < 0 || !byId.has(parentId)) continue;
    const parent = byId.get(parentId);
    const a = project([node[1], node[2], node[3]]);
    const b = project([parent[1], parent[2], parent[3]]);
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
  }}
  ctx.stroke();
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
    const alpha = Math.max(0.14, Math.min(0.58, point[3] / 255));
    ctx.fillStyle = `rgba(210, 220, 232, ${{alpha}})`;
    ctx.fillRect(point[0], point[1], 1.25, 1.25);
  }}

  drawSkeleton(DATA.gtSkeleton, gtById, "rgba(255, 82, 82, 0.55)", 0.9);
  drawSkeleton(DATA.initSkeleton, initById, "rgba(255, 184, 76, 0.72)", 1.05);

  ctx.lineWidth = 1.2;
  ctx.strokeStyle = "rgba(86, 155, 255, 0.78)";
  ctx.beginPath();
  for (const edge of DATA.graphEdges) {{
    const source = DATA.proposals[edge[0]];
    const target = DATA.proposals[edge[1]];
    const a = project(source);
    const b = project(target);
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
  }}
  ctx.stroke();

  const projectedProposals = DATA.proposals.map(point => {{
    const screen = project(point);
    return [screen[0], screen[1], screen[2], point[3], point[4]];
  }}).sort((a, b) => a[2] - b[2]);
  for (const proposal of projectedProposals) {{
    const score = proposal[3];
    const radius = Math.max(2.3, Math.min(6.8, 2.3 + score * 4.5));
    ctx.fillStyle = `rgba(78, 221, 255, ${{Math.max(0.32, Math.min(0.96, score))}})`;
    ctx.beginPath();
    ctx.arc(proposal[0], proposal[1], radius, 0, Math.PI * 2);
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
