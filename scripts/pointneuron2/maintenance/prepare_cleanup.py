"""Inventory retired PN2 outputs and verify a compact archive; never delete files."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT.parent
GENERATED = ROOT / 'tmp/pointneuron2'
PRESERVED = GENERATED / 'preserved_reference'
TEXT = {'.json', '.jsonl', '.csv', '.txt', '.log', '.md', '.py', '.ps1', '.toml', '.yaml', '.yml', '.tex'}
RETIRED_SCRATCH = [
    'pointneuron2_audit', 'pointneuron2_layout', 'pointneuron2_sampling',
    'pointneuron2_staging', 'pointneuron2_training', 'pointneuron2_workspace_backup',
    'pointneuron2_review_ui_check.js', 'pointneuron2_workspace_transfer.json',
    'pointneuron2_workspace_transfer.ps1',
]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inventory(path):
    files, links = [], []
    stack = [path]
    while stack:
        item = stack.pop()
        stat = item.stat(follow_symlinks=False)
        if stat.st_file_attributes & 1024:
            links.append(str(item))
        elif item.is_dir():
            stack.extend(item.iterdir())
        else:
            files.append({'path': str(item), 'bytes': stat.st_size})
    return files, links


def main():
    assert ROOT.name == 'pointneuron2' and (ROOT / '.git').is_file()
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'pointneuron2'
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ORIGINAL, text=True).strip() == 'main'
    PRESERVED.mkdir(parents=True, exist_ok=True)
    archive = PRESERVED / 'research_before_reset_20261021.zip'
    plan_path = PRESERVED / 'cleanup_plan.json'
    if archive.exists() or plan_path.exists():
        raise FileExistsError('Preserve existing cleanup archive/plan')
    targets = [p for p in GENERATED.iterdir() if p.name not in {'sampling_env', 'pointnext_reference', 'preserved_reference'}]
    targets += [ROOT / 'tmp/pdfs', ROOT / '.pytest_cache']
    targets += [ORIGINAL / 'tmp' / name for name in RETIRED_SCRATCH]
    for folder in (ROOT / 'src', ROOT / 'scripts', ROOT / 'tests'):
        targets += sorted(folder.rglob('__pycache__'))
    targets += sorted(p for p in (ROOT / 'output/pdf').iterdir() if p.suffix in {'.aux', '.log', '.out'})
    targets = sorted({p.absolute() for p in targets if p.exists()}, key=str)
    details, all_files = [], []
    for target in targets:
        assert target.is_relative_to(ROOT) or target.parent == ORIGINAL / 'tmp' and target.name in RETIRED_SCRATCH
        assert not target.is_symlink()
        files, links = inventory(target)
        all_files.extend(files)
        details.append({'path': str(target), 'bytes': sum(f['bytes'] for f in files), 'files': len(files), 'reparse_entries': links})

    selected = set()
    for row in all_files:
        path = Path(row['path'])
        if path.suffix.lower() in TEXT and not any(part in {'sampling_env', '.venv', '__pycache__', '.pytest_cache'} for part in path.parts):
            selected.add(path)
    canonical = GENERATED / 'local_runs/canonical_joint_fit_20261021/model'
    stable = GENERATED / 'local_runs/stable_joint_learnability_20261020/model'
    selected.add(canonical / 'stage_4000.pt')
    for folder in (canonical, stable):
        for crop in range(4):
            selected.update(folder / f'step_3000_crop_{crop}{suffix}' for suffix in ('.npz', '.swc'))
    selected.update((GENERATED / 'local_runs/canonical_targets_incremental_20261021/four_crop_cache').rglob('*.npz'))
    selected.update((GENERATED / 'local_runs/field_attention_20261013').rglob('field_epoch_030.pt'))
    assert all(path.is_file() for path in selected)
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    tracked = [ROOT / p for p in tracked if p and (ROOT / p).is_file()]
    protected = {str(path): digest(path) for path in tracked if not any(path.is_relative_to(target) for target in targets)}
    for relative in ('tmp/checkpoints/proposal_conservative_guarded30_cw3_oreg005_nw3.pt',
                     'tmp/splits/mixed_gold_source_coverage_combined_filtered_min15_seed0.json',
                     'tmp/paper_style_eval/guarded30_stable_100_162.json'):
        path = ORIGINAL / relative
        protected[str(path)] = digest(path)
    retained = []
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(selected):
            base, label = (ROOT, 'pn2') if path.is_relative_to(ROOT) else (ORIGINAL, 'original')
            name = label + '/' + path.relative_to(base).as_posix()
            output.write(path, name)
            retained.append({'source': str(path), 'member': name, 'bytes': path.stat().st_size, 'sha256': digest(path)})
        output.writestr('deleted_file_inventory.json', json.dumps(all_files, indent=2))
        output.writestr('retained_files.json', json.dumps(retained, indent=2))
    with zipfile.ZipFile(archive) as saved:
        assert saved.testzip() is None
        for row in retained:
            assert hashlib.sha256(saved.read(row['member'])).hexdigest() == row['sha256']
    plan = {
        'date': '2026-10-21', 'status': 'archive_verified_pending_cleanup',
        'workspace': str(ROOT), 'original': str(ORIGINAL),
        'archive': str(archive), 'archive_sha256': digest(archive), 'archive_bytes': archive.stat().st_size,
        'retained_files': len(retained), 'retained_uncompressed_bytes': sum(r['bytes'] for r in retained),
        'targets': details, 'target_bytes': sum(r['bytes'] for r in details),
        'target_files': sum(r['files'] for r in details), 'protected_sha256': protected,
        'keep': [str(GENERATED / n) for n in ('sampling_env', 'pointnext_reference', 'preserved_reference')],
        'original_pn1_and_dataset_policy': 'Preserve PN1 artifacts and raw data; only explicitly named original/tmp PN2 scratch is retired.',
        'archive_policy': 'Historical diagnostic evidence only, never an automatic fresh-training initialization or exact-resume promise.',
        'inference_or_training_launched': False,
    }
    plan_path.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: plan[k] for k in ('status', 'archive_bytes', 'retained_files', 'target_bytes', 'target_files')}, indent=2))


if __name__ == '__main__':
    main()
