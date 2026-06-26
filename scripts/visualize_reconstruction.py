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
    parser = argparse.ArgumentParser(description="Compare a reconstructed SWC against the Gold166 GT SWC.")
    parser.add_argument("--root", default="data/gold166", help="Path to Gold166 root.")
    parser.add_argument("--sample-index", type=int, required=True, help="Index in the scanned Gold166 samples.")
    parser.add_argument("--reconstruction-swc", required=True, help="Generated SWC file to compare.")
    parser.add_argument("--threshold", type=int, default=0, help="Foreground threshold; voxels > threshold become points.")
    parser.add_argument("--render-points", type=int, default=12000, help="Maximum foreground points to render.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for point sampling.")
    parser.add_argument("--proposals", help="Optional aggregated proposal .npz to overlay before graph selection.")
    parser.add_argument("--graph", help="Optional initialized graph .npz to overlay selected graph nodes.")
    parser.add_argument("--proposal-score-threshold", type=float, default=0.85, help="Minimum proposal score to render in the diagnostic overlay.")
    parser.add_argument("--proposal-render-limit", type=int, default=4096, help="Maximum pre-selection proposals to render.")
    parser.add_argument("--output", default="tmp/reconstructions/reconstruction_compare.html", help="Output HTML file.")
    args = parser.parse_args()

    samples = scan_gold166(args.root)
    sample = samples[args.sample_index]
    if sample.volume_path is None:
        raise ValueError(f"Sample has no volume: {sample.sample_id}")

    reconstruction_path = Path(args.reconstruction_swc)
    gt_swc = parse_swc(sample.swc_path)
    reconstruction_swc = parse_swc(reconstruction_path)

    volume = read_volume(sample.volume_path)
    point_cloud = volume_to_point_cloud(
        volume,
        threshold=args.threshold,
        max_points=args.render_points,
        seed=args.seed,
    )
    proposals = proposal_payload(
        Path(args.proposals),
        score_threshold=args.proposal_score_threshold,
        render_limit=args.proposal_render_limit,
    ) if args.proposals else []
    graph_nodes = graph_node_payload(Path(args.graph)) if args.graph else []

    html = render_html(
        {
            "sampleId": sample.sample_id,
            "gtSwc": str(sample.swc_path),
            "reconstructionSwc": str(reconstruction_path),
            "proposals": str(args.proposals) if args.proposals else "",
            "graph": str(args.graph) if args.graph else "",
            "volumeDimensions": point_cloud.volume_dimensions,
            "totalForegroundCount": point_cloud.total_foreground_count,
            "points": [[point.x, point.y, point.z, point.intensity] for point in point_cloud.points],
            "gtSkeleton": skeleton_payload(gt_swc),
            "reconstructionSkeleton": skeleton_payload(reconstruction_swc),
            "proposalCenters": proposals,
            "graphNodes": graph_nodes,
        }
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"sample_id: {sample.sample_id}")
    print(f"rendered_points: {len(point_cloud.points)}")
    print(f"gt_nodes: {len(gt_swc.nodes)}")
    print(f"reconstruction_nodes: {len(reconstruction_swc.nodes)}")
    print(f"reconstruction_roots: {reconstruction_swc.root_count}")
    if proposals:
        print(f"proposal_overlay_nodes: {len(proposals)}")
    if graph_nodes:
        print(f"graph_overlay_nodes: {len(graph_nodes)}")
    print(f"output: {output}")
    return 0


def proposal_payload(path: Path, score_threshold: float, render_limit: int) -> list[list[float]]:
    data = np.load(path, allow_pickle=False)
    centers = data["centers"].astype(float, copy=False)
    scores = data["scores"].astype(float, copy=False) if "scores" in data else np.ones((centers.shape[0],), dtype=float)
    indices = np.flatnonzero(scores >= score_threshold)
    if indices.size > render_limit:
        order = np.argsort(scores[indices])[::-1][:render_limit]
        indices = indices[order]
    return [
        [float(centers[index, 0]), float(centers[index, 1]), float(centers[index, 2]), float(scores[index])]
        for index in indices.tolist()
    ]


def graph_node_payload(path: Path) -> list[list[float]]:
    data = np.load(path, allow_pickle=False)
    centers = data["centers"].astype(float, copy=False)
    scores = data["scores"].astype(float, copy=False) if "scores" in data else np.ones((centers.shape[0],), dtype=float)
    return [
        [float(centers[index, 0]), float(centers[index, 1]), float(centers[index, 2]), float(scores[index])]
        for index in range(centers.shape[0])
    ]


def skeleton_payload(swc):
    return [
        [node.node_id, node.x, node.y, node.z, node.radius, node.parent_id]
        for node in swc_to_skeleton_records(swc)
    ]


def render_html(payload: dict) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PointNeuron Reconstruction Compare</title>
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
      max-width: min(780px, calc(100vw - 24px));
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
  <div id="legend">Drag to rotate. Wheel to zoom. Gray: foreground. Red: GT SWC. Faint cyan: high-score proposals. White/cyan dots: selected graph nodes. Bright cyan: reconstructed SWC.</div>
</div>
<script>
const DATA = {payload_json};
const canvas = document.getElementById("viewer");
const ctx = canvas.getContext("2d");
const title = document.getElementById("title");
const meta = document.getElementById("meta");

title.textContent = DATA.sampleId;
meta.textContent = `GT nodes ${{DATA.gtSkeleton.length.toLocaleString()}}, reconstructed nodes ${{DATA.reconstructionSkeleton.length.toLocaleString()}}, proposal overlay ${{DATA.proposalCenters.length.toLocaleString()}}, graph nodes ${{DATA.graphNodes.length.toLocaleString()}}, foreground ${{DATA.totalForegroundCount.toLocaleString()}}`;

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
const reconstructionById = new Map(DATA.reconstructionSkeleton.map(node => [node[0], node]));

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

function drawSkeleton(skeleton, byId, lineColor, pointColor, lineWidth, radius) {{
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = lineColor;
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

  ctx.fillStyle = pointColor;
  for (const node of skeleton) {{
    const p = project([node[1], node[2], node[3]]);
    ctx.beginPath();
    ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
    ctx.fill();
  }}
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
    const alpha = Math.max(0.13, Math.min(0.54, point[3] / 255));
    ctx.fillStyle = `rgba(210, 220, 232, ${{alpha}})`;
    ctx.fillRect(point[0], point[1], 1.25, 1.25);
  }}

  drawSkeleton(DATA.gtSkeleton, gtById, "rgba(255, 82, 82, 0.70)", "rgba(255, 116, 116, 0.90)", 1.05, 1.8);

  const projectedProposals = DATA.proposalCenters.map(point => {{
    const screen = project(point);
    return [screen[0], screen[1], screen[2], point[3]];
  }}).sort((a, b) => a[2] - b[2]);
  for (const proposal of projectedProposals) {{
    const score = proposal[3];
    ctx.fillStyle = `rgba(76, 218, 255, ${{Math.max(0.10, Math.min(0.34, score * 0.30))}})`;
    ctx.beginPath();
    ctx.arc(proposal[0], proposal[1], 2.1, 0, Math.PI * 2);
    ctx.fill();
  }}

  const projectedGraphNodes = DATA.graphNodes.map(point => {{
    const screen = project(point);
    return [screen[0], screen[1], screen[2], point[3]];
  }}).sort((a, b) => a[2] - b[2]);
  for (const node of projectedGraphNodes) {{
    ctx.fillStyle = "rgba(230, 252, 255, 0.82)";
    ctx.beginPath();
    ctx.arc(node[0], node[1], 3.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(48, 212, 255, 0.90)";
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }}

  drawSkeleton(DATA.reconstructionSkeleton, reconstructionById, "rgba(66, 220, 255, 0.92)", "rgba(66, 220, 255, 0.78)", 1.35, 2.4);
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
