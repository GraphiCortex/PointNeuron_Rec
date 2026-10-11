# Automatic preprocessing: current direction

The user clarified that semi-automation meant editing the neurite mask, for
example painting in missed branches. The parameter-adjustment viewer does not
provide that functionality. The development priority is now a main pipeline
that automatically processes each input image through SWC reconstruction.
Manual mask editing is deferred until the automatic version has been evaluated;
it may prove unnecessary, but that is not established yet.

The viewer stays available as an optional research diagnostic. Acceptance of
individual pilot revisions is not a condition for training or inference.
No existing sampler/model code depends on its review records, and the existing
training integration remains usable for execution checks. A review tool alone
does not solve automatic preparation of reliable training inputs.

## Current evidence and working hypothesis

In the informative train patch, the foreground covers 92.0% of the clipped
reference curve within 3 voxel units, whereas the full image guide covers 19.1%.
Using recorded voxel proportions changes full-guide coverage to 17.2%. Thus
neither guide compression alone nor spacing alone resolves that example.
Broad foreground regions may generate an unsuitable medial axis. This is a
working hypothesis, not proof that segmentation is the sole cause; image/trace
alignment and imaging differences still need investigation.

The validation example has stronger full-guide support (96.1%), but its bounded
pools lose coverage. Spatial candidate selection improves the train example
and worsens the validation example. These are two diagnostic patches, not a
dataset-wide result or a basis for promoting a preprocessing method.

## Next proposed experiment, before changing the main pipeline

**First mask comparison completed:** see [foreground_comparison.md](foreground_comparison.md).
It evaluated three automatic rules on 12 training patches. No method has been
promoted; the remaining sequence below requires further assessment of specificity,
guide reduction and broader input coverage before longer training.
The next [six-neuron full-guide mask/candidate experiment](preprocessing_comparison_expanded.md)
is now prepared for local user execution; the broader results remain pending.

1. Compare the current global intensity mask against an automatic mask using
   local background correction and an image-derived threshold. Use several
   training images spanning available imaging sources, including thin and weak
   branches. Preserve raw images and calibrated spacing. The exact method and
   settings are to be discussed before substantive implementation.
2. Hold skeleton extraction and point budgets fixed in this mask comparison.
   Record foreground size, branch/terminal reference support, guide continuity,
   extra structures, runtime and image overlays. Reference traces evaluate the
   result; they never construct input masks, guide vertices or candidates.
   Incomplete reference annotation limits claims about false positives.
3. Full guides are now retained by default, independently of the 1,024 candidate
   and 512 encoder-input budgets: see [full_guides.md](full_guides.md). Continue
   assessing candidate coverage and mask specificity separately. Empty guides
   already use a uniform sampling prior and zero guide loss; reliability of
   nonempty guides remains a research question.
4. Freeze image-only preprocessing choices from training development, prepare
   reproducible train/validation caches, then agree on matched random/learned
   training runs and evaluate full validation reconstructions. Existing inspected
   validation cases remain internal development evidence. Keep reserved test
   evaluation out of parameter selection.

Training can use fixed automatic preprocessing; it does not require the discrete
mask/skeleton extractor itself to be differentiable. The learnable sampler
already has its own tested gradient path. Reliable preprocessing and a matched
reconstruction comparison are still required before claiming stage 1 improves
PN 1.0 or moving to PointNeXt.

This note records the revised direction and proposed comparison. It introduces
no new extraction method, changes no model weights, and launches no training.
