"""Prepare and serve a local manual skeleton-review pilot; never starts training."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import secrets
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.manual_review import (build_review_candidate, review_settings,
                                         spacing_metadata, write_review_decision)
from pointneuron.data.medial_sampling import MedialSamplingConfig
from pointneuron.data.point_cloud import swc_to_skeleton_records
from pointneuron.data.sampling_cache import load_sampling_manifest
from pointneuron.data.swc import parse_swc
from pointneuron.data.training_cache import skeleton_to_array, volume_data_array
from pointneuron.data.vaa3d_raw import read_volume
from scripts.pointneuron2.audit_sampling import clipped_reference, reference_support


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def source_hashes():
    paths = ('scripts/pointneuron2/review_sampling.py', 'scripts/pointneuron2/review_sampling.html',
             'scripts/pointneuron2/audit_sampling.py', 'src/pointneuron/data/manual_review.py',
             'src/pointneuron/data/medial_sampling.py')
    return {path: digest(REPO_ROOT / path) for path in paths}


def initial_settings(manifest):
    return {'threshold_fraction': manifest['config']['threshold_fraction'],
            **{key: manifest['medial_config'][key] for key in
               ('spacing_xyz', 'teasar_scale', 'teasar_const', 'dust_threshold')},
            'candidate_method': 'uniform', 'guide_method': 'uniform', 'cell_size': 8}


def save_candidate(bundle, case, candidate, revision, curve):
    path = Path(bundle) / 'revisions' / case['case_id'] / f'revision_{revision:04d}.npz'
    if path.exists():
        raise FileExistsError('Review revision already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: candidate[key] for key in
                        ('points', 'full_guide_xyz', 'guide_edges', 'image_guide_xyz')},
                        metadata=np.array(json.dumps(candidate['settings'])))
    ruler = np.ones(3)  # Keep this diagnostic ruler fixed when extraction spacing changes.
    metrics = {name: reference_support(curve, xyz, ruler) for name, xyz in
               (('all_foreground', candidate['foreground_xyz']), ('full_guide', candidate['full_guide_xyz']),
                ('candidate_pool', candidate['points'][:, :3]), ('bounded_guide', candidate['image_guide_xyz']))}
    state = {'revision': revision, 'settings': candidate['settings'], 'medial_config': candidate['medial_config'],
             'artifact': path.relative_to(bundle).as_posix(), 'artifact_sha256': digest(path),
             'foreground_count': candidate['foreground_count'],
             'full_guide_vertices': len(candidate['full_guide_xyz']), 'guide_edges': len(candidate['guide_edges']),
             'candidate_rows': len(candidate['points']), 'bounded_guide_vertices': len(candidate['image_guide_xyz']),
             'threshold': candidate['threshold'], 'reference_curve_points': len(curve),
             'reference_support': metrics, 'metric_spacing_xyz': ruler.tolist(),
             'metric_scope': 'reference proximity within voxel units; not reconstruction accuracy',
             'review_status': 'pending', 'training_ready': False,
             'source_sha256': source_hashes(), 'kimimaro_version': version('kimimaro')}
    dump(path.with_suffix('.json'), state)
    return state


def prepare(args):
    if args.output_dir.exists():
        raise FileExistsError('Choose a fresh output directory for the review pilot')
    manifest = load_sampling_manifest(args.manifest, args.partition)
    if manifest.get('status') != 'completed':
        raise ValueError('Review preparation requires a completed cache')
    # Deliberately small pilot: one informative image patch per partition.
    selected = []
    for role in ('train', 'val'):
        candidates = [row for row in manifest['records'] if row['partition'] == role and row['label_nodes'] > 1]
        if candidates:
            selected.append(candidates[0])
    if not selected:
        raise ValueError('No informative reference patches for this diagnostic pilot')
    samples = scan_gold166(args.root)
    args.output_dir.mkdir(parents=True)
    result = {'schema_version': 1, 'status': 'preparing', 'scope': 'two-patch internal-development manual-review pilot',
              'training_ready': False, 'manifest_sha256': digest(args.manifest),
              'partition_sha256': digest(args.partition), 'base_medial_config': manifest['medial_config'],
              'pool_config': {key: manifest['config'][key] for key in ('candidate_points', 'guide_points')},
              'source_sha256': source_hashes(),
              'kimimaro_version': version('kimimaro'), 'cases': [],
              'inventory_note': 'Dataset inventory scans SWCs for valid file selection; only the selected train/val sources enter this pilot.'}
    dump(args.output_dir / 'session.json', result)
    try:
        for row in selected:
            sample = samples[row['sample_index']]
            if sample.sample_id != row['sample_id']:
                raise ValueError('Dataset identity differs from frozen cache')
            with np.load(row['resolved_path'], allow_pickle=False) as cache:
                metadata = json.loads(cache['metadata'].item())
                cached_points, cached_guide = cache['points'].copy(), cache['image_guide_xyz'].copy()
            if digest(sample.volume_path) != metadata['volume_sha256'] or digest(sample.swc_path) != metadata['swc_sha256']:
                raise ValueError('Image/reference source changed since cache construction')
            volume = read_volume(sample.volume_path)
            width, height, depth, channels = volume.dimensions
            if channels != 1:
                raise ValueError('Expected a single-channel source')
            center, radius = np.asarray(metadata['patch_center']), manifest['config']['patch_radius']
            lower = np.maximum(0, np.floor(center - radius)).astype(int)
            upper = np.minimum(np.array([width, height, depth]) - 1, np.ceil(center + radius)).astype(int)
            x0, y0, z0 = lower; x1, y1, z1 = upper + 1
            image = volume_data_array(volume).reshape(depth, height, width)[z0:z1, y0:y1, x0:x1].copy()
            nodes = skeleton_to_array(swc_to_skeleton_records(parse_swc(sample.swc_path)))
            curve, segments, crossings = clipped_reference(nodes, lower, upper)
            case_id = f"sample_{row['sample_index']:04d}_patch_{row['patch_index']:03d}"
            asset = args.output_dir / 'cases' / f'{case_id}.npz'
            asset.parent.mkdir(exist_ok=True)
            np.savez_compressed(asset, image=image, reference_curve=curve, reference_segments=np.asarray(segments).reshape(-1, 2, 3))
            case = {key: row[key] for key in ('sample_index', 'sample_id', 'partition', 'patch_index')}
            case.update(case_id=case_id, lower_xyz=lower.tolist(), upper_xyz=upper.tolist(),
                        image_shape_zyx=list(image.shape), datatype=volume.header.datatype, seed=metadata['seed'],
                        source_volume_sha256=metadata['volume_sha256'], source_swc_sha256=metadata['swc_sha256'],
                        spacing_metadata=spacing_metadata(sample.pixel_size_path),
                        case_asset=asset.relative_to(args.output_dir).as_posix(), case_asset_sha256=digest(asset),
                        boundary_crossing_edges=crossings, reference_exposed=False)
            settings = initial_settings(manifest)
            base = MedialSamplingConfig(**manifest['medial_config'])
            candidate = build_review_candidate(image, lower, settings, base,
                         manifest['config']['candidate_points'], manifest['config']['guide_points'],
                         metadata['seed'], volume.header.datatype)
            np.testing.assert_array_equal(candidate['points'], cached_points)
            np.testing.assert_allclose(candidate['image_guide_xyz'], cached_guide, atol=1e-6)
            state = save_candidate(args.output_dir, case, candidate, 1, curve)
            case['original_state'] = state
            case['current_state'] = state
            settings = {**settings, 'candidate_method': 'spatial', 'guide_method': 'farthest'}
            alternative = build_review_candidate(image, lower, settings, base,
                          manifest['config']['candidate_points'], manifest['config']['guide_points'],
                          metadata['seed'], volume.header.datatype,
                          existing_skeleton=(candidate['full_guide_xyz'], candidate['guide_edges']))
            case['pool_comparison_state'] = save_candidate(args.output_dir, case, alternative, 2, curve)
            # Current review remains the original; the alternative is not promoted.
            result['cases'].append(case)
            dump(args.output_dir / 'session.json', result)
            print(f"prepared {case_id}: original + pool-only comparison, review pending", flush=True)
        result['status'] = 'ready_for_manual_review'
        dump(args.output_dir / 'session.json', result)
    except BaseException as exc:
        result.update(status='failed', error=str(exc))
        dump(args.output_dir / 'session.json', result)
        raise
    print(f'Review bundle: {args.output_dir.resolve()}', flush=True)


class ReviewSession:
    def __init__(self, bundle):
        self.bundle = Path(bundle).resolve()
        self.session = json.loads((self.bundle / 'session.json').read_text(encoding='utf-8'))
        if self.session['status'] != 'ready_for_manual_review':
            raise ValueError('Review session preparation did not complete')
        self.cases = {case['case_id']: case for case in self.session['cases']}
        self.images, self.references, self.segments = {}, {}, {}
        for key, case in self.cases.items():
            if digest(self.bundle / case['case_asset']) != case['case_asset_sha256']:
                raise ValueError('Review image/reference asset changed')
            with np.load(self.bundle / case['case_asset'], allow_pickle=False) as asset:
                self.images[key] = asset['image'].copy()
                self.references[key] = asset['reference_curve'].copy()
                self.segments[key] = asset['reference_segments'].copy()
        self.lock = threading.RLock()
        self.job = {'status': 'idle'}
        self.token = secrets.token_urlsafe(24)
        self.exposed = {key: bool(case.get('reference_exposed')) for key, case in self.cases.items()}

    def save_session(self):
        dump(self.bundle / 'session.json', self.session)

    def state(self, key):
        with self.lock:
            case = self.cases[key]
            state = case['current_state']
            artifact = self.bundle / state['artifact']
            if digest(artifact) != state['artifact_sha256']:
                raise ValueError('Candidate artifact changed')
            with np.load(artifact, allow_pickle=False) as data:
                geometry = {name: data[name].tolist() for name in
                            ('points', 'full_guide_xyz', 'guide_edges', 'image_guide_xyz')}
            return {'case': case, 'state': state, 'geometry': geometry,
                    'reference_segments': self.segments[key].tolist(),
                    'reference_curve': self.references[key].tolist(), 'job': self.job,
                    'reference_exposed': self.exposed[key]}

    def recompute(self, key, settings):
        with self.lock:
            if self.job['status'] == 'running':
                raise ValueError('Wait for the current extraction to finish')
            base = MedialSamplingConfig(**self.session['base_medial_config'])
            settings, _ = review_settings(settings, base)
            self.job = {'status': 'running', 'case_id': key}
        def work():
            try:
                case = self.cases[key]
                previous = case['current_state']
                extraction_keys = ('threshold_fraction', 'spacing_xyz', 'teasar_scale', 'teasar_const', 'dust_threshold')
                existing = None
                if all(previous['settings'][k] == settings[k] for k in extraction_keys):
                    if digest(self.bundle / previous['artifact']) != previous['artifact_sha256']:
                        raise ValueError('Candidate artifact changed')
                    with np.load(self.bundle / previous['artifact'], allow_pickle=False) as old:
                        existing = (old['full_guide_xyz'].copy(), old['guide_edges'].copy())
                candidate = build_review_candidate(self.images[key], case['lower_xyz'], settings, base,
                              self.session['pool_config']['candidate_points'], self.session['pool_config']['guide_points'],
                              case['seed'], case['datatype'], existing)
                revisions = list((self.bundle / 'revisions' / key).glob('revision_*.json'))
                revision = max(int(path.stem.split('_')[-1]) for path in revisions) + 1
                state = save_candidate(self.bundle, case, candidate, revision, self.references[key])
                with self.lock:
                    case['current_state'] = state
                    self.save_session()
                    self.job = {'status': 'completed', 'case_id': key, 'revision': revision}
            except Exception as exc:
                with self.lock:
                    self.job = {'status': 'failed', 'case_id': key, 'error': str(exc)}
        threading.Thread(target=work, daemon=True).start()

    def decide(self, payload):
        with self.lock:
            if self.job['status'] == 'running':
                raise ValueError('Cannot review while extraction is running')
            key = payload['case_id']
            case = self.cases[key]
            state = case['current_state']
            if payload['revision'] != state['revision']:
                raise ValueError('Displayed revision is stale; refresh before reviewing')
            path, record = write_review_decision(self.bundle, case, state, payload['decision'],
                          payload['reviewer'], payload.get('notes', ''), self.exposed[key])
            case['current_state'] = {**state, 'review_status': record['decision'],
                                      'review_record': path.relative_to(self.bundle).as_posix()}
            self.save_session()
            return {'path': str(path), 'review': record}

    def image(self, query):
        key = query['case'][0]
        image = self.images[key]
        plane = query.get('plane', ['xy'])[0]
        mode = query.get('mode', ['slice'])[0]
        axis = {'xy': 0, 'xz': 1, 'yz': 2}[plane]
        index = int(query.get('index', ['0'])[0])
        if mode == 'mip':
            pixels = image.max(axis=axis)
        elif mode == 'slice' and 0 <= index < image.shape[axis]:
            pixels = np.take(image, index, axis=axis)
        else:
            raise ValueError('Invalid image view')
        brightness = float(query.get('brightness', ['1'])[0])
        if not np.isfinite(brightness) or not .1 <= brightness <= 10:
            raise ValueError('Invalid brightness')
        limit = max(1., float(np.quantile(image, .998)))
        pixels = np.uint8(np.clip(pixels.astype(float) / limit * 255 * brightness, 0, 255))
        stream = BytesIO()
        Image.fromarray(pixels).save(stream, format='PNG')
        return stream.getvalue()


def serve(bundle, port):
    session = ReviewSession(bundle)
    html = (Path(__file__).with_name('review_sampling.html')).read_text(encoding='utf-8')
    html = html.replace('__REVIEW_TOKEN__', session.token)
    class Handler(BaseHTTPRequestHandler):
        def respond(self, data, mime='application/json', code=200):
            if not isinstance(data, bytes):
                data = json.dumps(data, allow_nan=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            parsed = urlsplit(self.path)
            try:
                if parsed.path == '/':
                    return self.respond(html.encode('utf-8'), 'text/html; charset=utf-8')
                if parsed.path == '/api/session':
                    return self.respond({'cases': [{'case_id': c['case_id'], 'sample_index': c['sample_index'],
                                 'partition': c['partition'], 'patch_index': c['patch_index']} for c in session.cases.values()],
                                 'bundle': str(session.bundle), 'job': session.job})
                query = parse_qs(parsed.query)
                if parsed.path == '/api/state':
                    return self.respond(session.state(query['case'][0]))
                if parsed.path == '/image':
                    return self.respond(session.image(query), 'image/png')
                self.respond({'error': 'Unknown route'}, code=404)
            except (ValueError, KeyError, IndexError) as exc:
                self.respond({'error': str(exc)}, code=400)

        def do_POST(self):
            # Bind locally and require a per-session token, preventing cross-site writes.
            if self.headers.get('X-Review-Token') != session.token:
                return self.respond({'error': 'Invalid review token'}, code=403)
            origin = self.headers.get('Origin')
            if origin and origin != f'http://127.0.0.1:{self.server.server_port}':
                return self.respond({'error': 'Wrong origin'}, code=403)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16384:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(size))
                key = payload['case_id']
                if key not in session.cases:
                    raise ValueError('Unknown review case')
                if self.path == '/api/recompute':
                    session.recompute(key, payload['settings'])
                    return self.respond({'status': 'running'})
                if self.path == '/api/decision':
                    return self.respond(session.decide(payload))
                if self.path == '/api/reference-exposure':
                    with session.lock:
                        session.exposed[key] = True
                        session.cases[key]['reference_exposed'] = True
                        session.save_session()
                    return self.respond({'recorded': True})
                self.respond({'error': 'Unknown route'}, code=404)
            except (ValueError, KeyError, TypeError) as exc:
                self.respond({'error': str(exc)}, code=400)

        def log_message(self, fmt, *args):
            if args and str(args[0]).startswith('POST'):
                super().log_message(fmt, *args)
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Manual review: http://127.0.0.1:{server.server_port}', flush=True)
    print(f'Bundle: {session.bundle}\nLocal CPU review server; no training is launched. Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO_ROOT.parent / 'data/gold166')
    parser.add_argument('--manifest', type=Path, default=REPO_ROOT / 'tmp/pointneuron2/sampling_training_cache_smoke/manifest.json')
    parser.add_argument('--partition', type=Path, default=REPO_ROOT / 'configs/pointneuron2/partition_seed0.json')
    parser.add_argument('--output-dir', type=Path, help='Fresh directory for original images and review candidates')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--serve', type=Path, help='Open an existing prepared review bundle')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if args.serve:
        if args.prepare_only or args.output_dir:
            parser.error('--serve cannot be combined with preparation options')
        serve(args.serve, args.port)
    else:
        if args.output_dir is None:
            args.output_dir = REPO_ROOT / 'tmp/pointneuron2/manual_reviews' / time.strftime('%Y%m%d_%H%M%S')
        prepare(args)
        if not args.prepare_only:
            serve(args.output_dir, args.port)


if __name__ == '__main__':
    main()
