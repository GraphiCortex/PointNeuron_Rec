from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.torch_dataset import TrainingCacheDataset, collate_training_records, load_split_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize trained skeleton proposal predictions for one cached sample.")
    parser.add_argument("--split-file", required=True, help="Path to split JSON.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Split to draw from.")
    parser.add_argument("--record-index", type=int, default=0, help="Record index within the selected split.")
    parser.add_argument("--checkpoint", required=True, help="Path to proposal checkpoint.")
    parser.add_argument("--output", default="tmp/visualizations/proposals.html", help="Output HTML file.")
    parser.add_argument("--k", type=int, help="Override kNN neighbors. Defaults to checkpoint value or 20.")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Minimum objectness probability to consider.")
    parser.add_argument("--top-proposals", type=int, default=256, help="Maximum proposals to render after NMS.")
    parser.add_argument("--nms-radius", type=float, default=8.0, help="Spherical NMS radius in voxels.")
    parser.add_argument("--positive-distance", type=float, default=6.0, help="Distance used only for reporting close proposal rate.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on.")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is required. Install a CUDA-compatible torch build before visualizing proposals.")
        return 2
    from pointneuron.models.dgcnn import DGCNNEncoder
    from pointneuron.models.proposal import SkeletonProposalHead

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but torch.cuda.is_available() is false.")
        return 2

    paths = load_split_paths(args.split_file, args.split)
    if not paths:
        print(f"No records found for split {args.split!r}.")
        return 2
    if args.record_index < 0 or args.record_index >= len(paths):
        print(f"--record-index must be in [0, {len(paths) - 1}], got {args.record_index}")
        return 2

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    k = args.k if args.k is not None else int(checkpoint_args.get("k", 20))

    dataset = TrainingCacheDataset([paths[args.record_index]])
    batch = collate_training_records([dataset[0]])
    points = batch["points"].to(device)
    skeleton_nodes = batch["skeleton_nodes"].to(device)
    skeleton_mask = batch["skeleton_mask"].to(device)

    encoder = DGCNNEncoder(k=k).to(device)
    proposal = SkeletonProposalHead(in_channels=encoder.output_dim).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    proposal.load_state_dict(checkpoint["proposal"])
    encoder.eval()
    proposal.eval()

    with torch.no_grad():
        features = encoder(points)
        output = proposal(points, features)
        probabilities = output.objectness_logits.softmax(dim=-1)[0, :, 1]
        centers = output.center_proposals[0]
        radii = output.radius[0, :, 0]
        valid_skeleton = skeleton_nodes[0, skeleton_mask[0]]
        distances = torch.cdist(centers, valid_skeleton[:, 1:4]).min(dim=1).values

    selected_indices = select_proposals(
        centers=centers,
        scores=probabilities,
        score_threshold=args.score_threshold,
        top_proposals=args.top_proposals,
        nms_radius=args.nms_radius,
    )
    selected = [
        [
            float(centers[index, 0].item()),
            float(centers[index, 1].item()),
            float(centers[index, 2].item()),
            float(radii[index].item()),
            float(probabilities[index].item()),
            float(distances[index].item()),
        ]
        for index in selected_indices
    ]
    selected_distances = [proposal_record[5] for proposal_record in selected]
    selected_scores = [proposal_record[4] for proposal_record in selected]
    if selected:
        selected_centers = torch.tensor([proposal_record[:3] for proposal_record in selected], dtype=valid_skeleton.dtype, device=device)
        swc_to_selected = torch.cdist(valid_skeleton[:, 1:4], selected_centers).min(dim=1).values
        covered_nodes = int((swc_to_selected <= args.positive_distance).sum().item())
    else:
        covered_nodes = 0

    metadata = batch["metadata"][0]
    html = render_html(
        sample_id=str(metadata.get("sample_id", paths[args.record_index])),
        volume_dimensions=tuple(metadata["volume_dimensions"]),
        total_foreground_count=int(metadata["total_foreground_count"]),
        points=batch["points"][0].cpu().tolist(),
        skeleton=batch["skeleton_nodes"][0, batch["skeleton_mask"][0]].cpu().tolist(),
        proposals=selected,
        split=args.split,
        checkpoint=args.checkpoint,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    close_count = sum(1 for distance in selected_distances if distance <= args.positive_distance)
    print(f"device: {device}")
    print(f"sample_id: {metadata.get('sample_id', paths[args.record_index])}")
    print(f"record_path: {paths[args.record_index]}")
    print(f"candidate_points: {points.shape[1]}")
    print(f"selected_proposals: {len(selected)}")
    if selected:
        print(f"mean_score: {statistics.fmean(selected_scores):.4f}")
        print(f"mean_distance_to_swc: {statistics.fmean(selected_distances):.4f}")
        print(f"median_distance_to_swc: {statistics.median(selected_distances):.4f}")
        print(f"within_{args.positive_distance:g}_voxels: {close_count}/{len(selected)}")
        print(f"swc_nodes_covered_within_{args.positive_distance:g}_voxels: {covered_nodes}/{valid_skeleton.shape[0]}")
    print(f"output: {output_path}")
    return 0


def select_proposals(centers, scores, score_threshold: float, top_proposals: int, nms_radius: float) -> list[int]:
    import torch

    candidate_indices = torch.nonzero(scores >= score_threshold, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        candidate_indices = torch.arange(scores.shape[0], device=scores.device)

    candidate_scores = scores[candidate_indices]
    order = candidate_scores.argsort(descending=True)
    ordered_indices = candidate_indices[order[: max(top_proposals * 4, top_proposals)]]

    selected: list[int] = []
    radius_squared = float(nms_radius) ** 2
    for tensor_index in ordered_indices:
        index = int(tensor_index.item())
        if len(selected) >= top_proposals:
            break
        center = centers[index]
        keep = True
        for selected_index in selected:
            distance_squared = torch.sum((center - centers[selected_index]) ** 2).item()
            if distance_squared <= radius_squared:
                keep = False
                break
        if keep:
            selected.append(index)
    return selected


def render_html(
    sample_id: str,
    volume_dimensions: tuple[int, int, int, int],
    total_foreground_count: int,
    points: list[list[float]],
    skeleton: list[list[float]],
    proposals: list[list[float]],
    split: str,
    checkpoint: str,
) -> str:
    payload = {
        "sampleId": sample_id,
        "volumeDimensions": volume_dimensions,
        "totalForegroundCount": total_foreground_count,
        "points": points,
        "skeleton": skeleton,
        "proposals": proposals,
        "split": split,
        "checkpoint": checkpoint,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PointNeuron Proposal Viewer</title>
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
      max-width: min(620px, calc(100vw - 24px));
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
  <div id="legend">Drag to rotate. Wheel to zoom. Gray: sampled foreground. Red: SWC skeleton. Cyan: predicted proposal centers.</div>
</div>
<script>
const DATA = {payload_json};
const canvas = document.getElementById("viewer");
const ctx = canvas.getContext("2d");
const title = document.getElementById("title");
const meta = document.getElementById("meta");

title.textContent = DATA.sampleId;
meta.textContent = `split ${{DATA.split}}, volume ${{DATA.volumeDimensions.join(" x ")}}, points ${{DATA.points.length.toLocaleString()}}, foreground ${{DATA.totalForegroundCount.toLocaleString()}}, skeleton nodes ${{DATA.skeleton.length.toLocaleString()}}, proposals ${{DATA.proposals.length.toLocaleString()}}`;

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
  return [width / 2 + x1 * scale, height / 2 + y1 * scale, z2];
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
    const alpha = Math.max(0.16, Math.min(0.70, point[3] / 255));
    ctx.fillStyle = `rgba(210, 220, 232, ${{alpha}})`;
    ctx.fillRect(point[0], point[1], 1.35, 1.35);
  }}

  ctx.lineWidth = 1.05;
  ctx.strokeStyle = "rgba(255, 92, 92, 0.72)";
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

  const projectedProposals = DATA.proposals.map(point => {{
    const screen = project(point);
    return [screen[0], screen[1], screen[2], point[3], point[4], point[5]];
  }}).sort((a, b) => a[2] - b[2]);
  for (const proposal of projectedProposals) {{
    const score = proposal[4];
    const radius = Math.max(2.0, Math.min(7.0, 2.0 + score * 5.0));
    ctx.fillStyle = `rgba(78, 221, 255, ${{Math.max(0.22, Math.min(0.92, score))}})`;
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
