"""Explicit diagnostic correction to pinned native interpolation, not a new model.

Adapted from the historical native operator, itself based on OpenPoints conventions.
MIT License: Copyright (c) Guocheng Qian.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch


def make_interpolator(ops):
    """Reuse validated historical helpers; only copy the selected index slice."""
    def interpolate_features(query, support, features, chunk_size=256):
        ops._xyz(query); ops._xyz(support)
        if (query.shape[0] != support.shape[0] or query.device != support.device
                or features.shape[:2] != support.shape[:2] or features.ndim != 3
                or features.device != support.device or features.dtype != torch.float32
                or not bool(torch.isfinite(features).all())):
            raise ValueError("Interpolation requires aligned float32 support features")
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("Invalid distance chunk size")
        chunks = []
        for start in range(0, query.shape[1], chunk_size):
            with torch.no_grad():
                distance = torch.cdist(query[:, start:start+chunk_size], support,
                                       compute_mode="donot_use_mm_for_euclid_dist")
                # The only algorithm-body difference: own just the selected indices.
                indices = distance.argsort(dim=-1, stable=True)[..., :min(3, support.shape[1])].clone()
                reciprocal = (distance.gather(-1, indices) + 1e-8).reciprocal()
                weights = reciprocal / reciprocal.sum(-1, keepdim=True)
            chunks.append((ops.gather_points(features, indices) * weights[..., None]).sum(-2))
        return torch.cat(chunks, 1)
    return interpolate_features
