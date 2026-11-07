"""Prepare LaTeX tables from the saved report CSVs; no experiment/scoring.

Run with the bundled Python, then compile report.tex from the report directory.
The attached NG report is a private style reference and is not published.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reports/progress_20261107'
GEN = OUT / 'tables'


def escape(s):
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_')]:
        s=s.replace(a,b)
    return s


def main():
    GEN.mkdir(exist_ok=True)
    manifest=json.loads((OUT/'source_manifest.json').read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for item in manifest['sources'].values():
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    assert sha(ROOT/'tmp/stage2_trees_20261105/STOP.json')==manifest['Stage2_STOP_sha256']
    with (OUT/'development_summary.csv').open(encoding='utf-8',newline='') as f:
        macro=list(csv.DictReader(f))
    with (OUT/'development_results.csv').open(encoding='utf-8',newline='') as f:
        detail=list(csv.DictReader(f))
    names={'classical':'Classical graph','archive':r'Archive leader\textsuperscript{*}',
           'broad_dense':'Broad dense head','sato':'Sato point forest'}
    fields=['segment_2','segment_6','resampled_2','resampled_6']
    def vals(row):
        return [f'{100*float(row[k]):.2f}' for k in fields]+[f'{float(row["length_ratio"]):.3f}']
    lines=[]
    for row in macro:lines.append(names[row['method']]+' & '+' & '.join(vals(row))+r' \\')
    (GEN/'development_macro.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    lines=[]
    for i,row in enumerate(detail):
        if i%4==0:
            if i:lines.append(r'\midrule')
            lines.append(r'\multicolumn{6}{l}{\textbf{'+escape(row['neuron'])+r'}}\\[2pt]')
        lines.append(names[row['method']]+' & '+' & '.join(vals(row))+r' \\')
    assert len(detail)==28
    (GEN/'development_neurons.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    # Keep source numbers in the headroom table tied to the existing full-precision manifest.
    lines=[]
    for key,label in [('stage0','Original union'),('compact','Compact union'),
                      ('prune_only','Classical prune only'),('prune_add','Classical prune plus add')]:
        row=manifest['oracle_values'][key]
        lines.append(label+' & '+' & '.join([f'{100*row[k]:.2f}' for k in fields]+[f'{row["length_ratio"]:.3f}'])+r' \\')
    (GEN/'oracle_results.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    macros=[]
    for name,file in [('DevelopmentMacroRows','development_macro.tex'),
                      ('DevelopmentNeuronRows','development_neurons.tex'),
                      ('OracleRows','oracle_results.tex')]:
        macros.append('\\newcommand{\\'+name+'}{%\n'+(GEN/file).read_text(encoding='utf-8')+'}\n')
    (GEN/'data_rows.tex').write_text(''.join(macros),encoding='utf-8',newline='\n')
    print('Prepared 4 macro rows, 28 neuron rows and 4 saved oracle rows; 17 source hashes and STOP unchanged.')


if __name__=='__main__':
    main()
