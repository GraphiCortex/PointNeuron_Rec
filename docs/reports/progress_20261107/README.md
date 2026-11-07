# PointNeuron 2

## Progress and the next experimental direction

Jad Sawan · Report for Dr Arij · 7 November 2026

[Read the report](PointNeuron2_Progress_Report_Dr_Arij_20261107.pdf) · [Editable LaTeX](report.tex) · [All development results](development_results.csv)

Since the earlier proposal, I have been testing where the proposed changes to PointNeuron can improve the reconstructed neuron. This report follows that work from the original motivation for PointNeXt, Transformer proposals and GATv2 through the preparation audits, small fitting tests, curve supervision and downstream reconstruction comparisons.

The important development has been a clearer account of what survives through the pipeline. The Transformer could fit the small training task; the later dense head recovered more candidate geometry with broader training, but did not produce a useful compact reconstruction. That led to the graph-selector controls and to a more specific question about the semi-automatic idea: can a small amount of guidance improve the current image, and can it improve preparation on later uncorrected neurons?

The report separates those findings from the proposed next study. The experimental Transformer bypass still awaits Dr Arij's agreement. GATv2 remains intended and untrained. No new experiment or criterion is adopted by the report.

### Reading the report

The PDF contains the original architecture, the experimental paths actually tested and a possible adaptive design, together with the saved input-support chart, reconstruction comparison and microscopy review. Its main text explains why each test followed the previous one. The complete seven-neuron comparison, scorer definitions and linked source records are in the appendices. The five discussion questions close the main narrative.

### Numerical records

- [28 method–neuron rows](development_results.csv), including the Series017 archive protocol zero.
- [Unrounded seven-neuron summaries](development_summary.csv).
- [Saved source files and hashes](source_manifest.json), pinned to evidence commit `8c363d3`.
- [Document and numerical checks](validation.json).

The tables retain segment and resampled F1 at two and six native voxels, plus predicted/reference length. The seven development neurons have been inspected repeatedly; the eight reserved neurons remain untouched. Oracle measurements are reference-assisted, and the archive comparison retains its disclosed coordinate-check limitation.

### Editing and compilation

`report.tex` is the current report source. The author-supplied style reference is private and is not included. The document uses Latin Modern, standard article layout, TikZ diagrams and booktabs tables. The current PDF was compiled with pdfLaTeX, not ReportLab.

The generated table fragments in `tables/` are included, so the report can be compiled without running a model or a scorer. From this directory, compile twice:

```text
pdflatex -interaction=nonstopmode -halt-on-error -jobname=PointNeuron2_Progress_Report_Dr_Arij_20261107 report.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=PointNeuron2_Progress_Report_Dr_Arij_20261107 report.tex
```

For local development, direct compiler intermediates into an ignored folder under `PN2/tmp`. Tables can be regenerated from the saved CSVs with `scripts/prepare_progress_report_latex_20261107.py`. The historical ReportLab first-draft builder writes only to ignored `tmp` and does not overwrite this version.

The repository is public. All files needed to read the report are tracked; large experiment caches and checkpoints remain local under `tmp/`.
