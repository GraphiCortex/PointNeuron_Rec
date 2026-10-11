# Manual skeleton verification: start here

**Direction updated:** the main pipeline will aim to run automatically.
This viewer is now an optional development diagnostic. The user's intended
semi-automated feature was painting/correcting the neurite mask, which this tool
does not implement. Manual review is not a required next action or a training
gate. See [automatic_preprocessing.md](automatic_preprocessing.md).

The proposal includes manual verification or parameter adjustment after image
skeleton extraction. Our earlier figures supported inspection, but there was no
workflow for adjusting parameters and recording a decision. The new local
reviewer supplies that workflow for a **two-patch pilot**. Stage 1 is still
incomplete: neither patch has been accepted by a human, and reviewed settings
have not been integrated into a new training cache.

## Open the prepared pilot

The viewer is currently available at <http://127.0.0.1:8765>. It runs on this
machine's CPU and does not start training. If the server has stopped, run this
from the original `Documents/GitHub/PointNeuron` folder:

```powershell
& .\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe .\pointneuron2\scripts\pointneuron2\review_sampling.py --serve .\pointneuron2\tmp\pointneuron2\manual_reviews\pilot_seed0
```

Leave the terminal running, open the address printed there, and use Ctrl+C to
stop the server. If port 8765 is occupied, add `--port 8766` and open that port.
The server listens on 127.0.0.1; the page uses no external services or assets.

## What you do next

1. Start with **train neuron 0, patch 1**. Its current revision 3 is a spacing
   comparison, still pending review. Inspect single slices in XY, XZ and YZ.
   Move the slice slider; turn the full skeleton overlay off/on to see whether
   branches follow the visible neurites through depth. Rotate the 3D view to
   check continuity. Maximum projections help orientation but can hide errors.
2. Note missing thin branches, guide paths through broad bright regions,
   disconnected components and questionable joins. Extra branches alone do not
   establish an error: annotations can omit visible structures.
3. Adjust one extraction setting at a time, then click **Recompute candidate**.
   Changing fields alone does not change the displayed revision. Brightness is
   a display control; it does not change extraction. Each recomputation saves
   a new revision and preserves the earlier files.
4. **Restore original settings** fills the original fields; recompute to view
   that extraction again. **Compare pool reduction only** uses the original
   extraction with spatial candidates and farthest-point guide reduction at
   the same 1,024/512 budgets. These are experimental alternatives, not defaults.
5. Enter your name and notes, then choose **Accept this revision**, **Needs
   adjustment**, or **Reject this revision**. Acceptance means you judged this
   particular image guide reliable for its stated use. It does not certify
   anatomical ground truth or launch training. Empty guides cannot be accepted;
   adjustment/rejection requires notes. Repeat for validation neuron 19, patch 0.

Use **Reference diagnostic** optionally after inspecting the image. It overlays
the clipped SWC and reveals proximity measurements. Exposure is recorded even
if the overlay is subsequently hidden. Reference-assisted validation review is
development evidence, not an independent image-only validation experiment.
SWC coordinates never select candidates or generate the skeleton.

## What the first comparisons tell us

Numbers below are percentages of the clipped reference curve within **3 voxel
units** of each point set. The diagnostic ruler stays [1,1,1] across settings so
the comparison does not change units. These are proximity checks, not final
reconstruction scores, precision measures or evidence of general improvement.

| Patch / revision | Full guide | Candidate pool | Bounded guide |
|---|---:|---:|---:|
| Train 0 / original (1) | 19.1% | 36.7% | 10.7% |
| Train 0 / pool-only (2) | 19.1% | 55.0% | 11.4% |
| Train 0 / source spacing (3) | 17.2% | 36.7% | 7.1% |
| Val 19 / original (1) | 96.1% | 72.3% | 59.4% |
| Val 19 / pool-only (2) | 96.1% | 63.9% | 67.1% |

The pool comparison is mixed. Better candidate coverage on the train example
does not repair its poor full guide, and the same candidate method worsens
coverage on the validation example. Guide reduction improves validation coverage
but still loses support. Neither method has been promoted.

Train neuron 0 has source spacing **0.103 x 0.103 x 0.3 micrometres**. Revision 3
uses relative XYZ spacing [1,1,2.912621359223301], keeping TEASAR distances and its
constant in X-voxel units. The foreground and uniform candidate pool stay the
same; full guide vertices change from 2,585 to 2,394. This comparison does not
fix the poor guide. Validation neuron 19 has no recognized spacing metadata;
the viewer asks for manual confirmation rather than guessing.

The working hypothesis remains that the train patch's broad thresholded
foreground produces an unsuitable medial axis. That can be a preprocessing
limitation even when skeletonization code executes correctly. Source-image/SWC
alignment and the suitability of the segmentation also need visual verification.
The proposal assumes a segmented neuronal volume; thresholding a raw image has
not yet been shown to meet that assumption for these cases.

## Files and reproducibility

- Entry point: `scripts/pointneuron2/review_sampling.py`.
- Viewer: `scripts/pointneuron2/review_sampling.html`.
- Shared algorithms and review records: `src/pointneuron/data/manual_review.py`.
- Extraction with preserved component edges: `src/pointneuron/data/medial_sampling.py`.
- Readable comparison snapshot: [manual_review_pilot.json](manual_review_pilot.json).
- Local generated bundle: `tmp/pointneuron2/manual_reviews/pilot_seed0`.

The bundle contains cropped source images, reference diagnostics, immutable
revision NPZ/JSON pairs and `session.json`. Your decisions create separate JSON
files in `reviews`, recording reviewer, notes, revision, exact artifact hash,
source hashes, parameters and reference exposure. All generated bundle data
stays in ignored `tmp`; scripts and this report are outside it. The report
snapshot predates your decisions and is not a live review log.

To prepare a separate fresh pilot from the existing cache, run from the PN 2.0
folder (this performs a small CPU extraction; it does not train):

```powershell
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe scripts/pointneuron2/review_sampling.py --prepare-only --output-dir tmp/pointneuron2/manual_reviews/pilot_new
```

Preparation verifies the original cached points against the raw crop. The pilot
deliberately selects one train and one validation patch with multiple local SWC
nodes; it is not representative dataset sampling. Dataset inventory scans SWCs
to choose valid files; only these train/validation patches enter this pilot.
No reserved test neuron is evaluated.

Validation: all **103 repository tests pass** with unrelated pytest plugins
disabled, including seven new checks for spacing, pool coverage selection,
image-only inputs, component edges and decision integrity. Browser checks covered
source-spacing entry, CPU recomputation, slice/projection controls and case
switching. No human review decision was fabricated; no training was launched.

The next proposed experiment compares automatic foreground extraction on a
broader training sample. Viewer decisions are optional diagnostic records and
will not become mandatory per-case cache approvals. Longer matched
training/evaluation follows agreement on the automatic preprocessing recipe;
PointNeXt remains a later step.
