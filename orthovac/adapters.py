import json
import os
import zlib
from pathlib import Path

from .constants import LORA_KEY_RE, SITE_KEY_RE, SITE_TYPES, WEIGHT_NAMES
from .naming import adapter_dir, make_record

try:
    import pandas as pd
except ModuleNotFoundError:   # the analysis extras are optional
    pd = None

_ADAPTER_CACHE = {}
_BASIS_CACHE = {}


def fetch_adapter(cfg, ref):
    """Local dir or HF repo id -> a local directory holding config + weights."""
    ref = str(ref).rstrip('/')
    if Path(ref).is_dir():
        return Path(ref)
    from huggingface_hub import snapshot_download
    local = cfg.local_adapter_base / ('_src__' + ref.replace('/', '__'))
    snapshot_download(repo_id=ref, repo_type='model', token=os.environ.get('HF_TOKEN'),
                      local_dir=str(local),
                      allow_patterns=['adapter_config.json', *WEIGHT_NAMES])
    return local


def load_adapter(cfg, ref):
    """-> (state_dict, config dict). Cached, since the sweep reuses each adapter."""
    from safetensors.torch import load_file as st_load
    import torch
    key = str(ref)
    if key in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[key]
    d = fetch_adapter(cfg, ref)
    adapter_cfg = json.loads((d / 'adapter_config.json').read_text())
    if adapter_cfg.get('use_dora'):
        raise ValueError(f'{ref}: DoRA adapters are not supported — the left-projection '
                         'identity assumes plain LoRA (dW = s*B*A).')
    st = d / 'adapter_model.safetensors'
    sd = st_load(str(st)) if st.exists() else torch.load(d / 'adapter_model.bin', map_location='cpu')
    _ADAPTER_CACHE[key] = (sd, adapter_cfg)
    return sd, adapter_cfg


def lora_pairs(sd, site_types=None):
    """{(layer, site): {'A': key, 'B': key}} for every LoRA module in the state dict."""
    pairs = {}
    for key in sd:
        m = LORA_KEY_RE.match(key)
        if not m:
            continue
        site = SITE_KEY_RE.search(m.group('module'))
        if not site:
            continue
        layer, block, proj = int(site.group(1)), site.group(2), site.group(3)
        site_type = f'{block}.{proj}'
        if site_types and site_type not in site_types:
            continue
        pairs.setdefault((layer, site_type), {})[m.group('ab')] = key
    return {k: v for k, v in pairs.items() if 'A' in v and 'B' in v}


def donor_write_basis(cfg, B_d, A_d, energy=None, k_cap=None):
    """Top left-singular vectors of dW_d = s*B_d@A_d, via an r x r SVD. -> U (k, out).

    Exact: with B_d = Qb Rb and A_d^T = Qa Ra, dW_d = s * Qb (Rb Ra^T) Qa^T, so the left
    singular vectors of dW_d are Qb times those of the small r x r matrix. The out x in
    matrix is never formed, and s drops out (positive scalar).
    """
    import torch
    energy = cfg.energy if energy is None else energy
    k_cap = cfg.k_cap if k_cap is None else k_cap
    B_d = B_d.to(torch.float32)
    A_d = A_d.to(torch.float32)
    Qb, Rb = torch.linalg.qr(B_d)           # Qb (out, r)
    Qa, Ra = torch.linalg.qr(A_d.T)         # Qa (in, r)
    Uh, S, _ = torch.linalg.svd(Rb @ Ra.T)  # r x r
    total = (S ** 2).sum()
    if total <= 0:
        return B_d.new_zeros((0, B_d.shape[0]))
    c = torch.cumsum(S ** 2, 0) / total
    k = min(int((c < energy).sum().item()) + 1, int(k_cap), S.numel())
    return (Qb @ Uh[:, :k]).T.contiguous()  # (k, out), orthonormal rows


def donor_bases(cfg, source, site_types=None, energy=None, k_cap=None):
    """{(layer, site): U} for one donor. Cached — it depends on neither lambda nor target."""
    energy = cfg.energy if energy is None else energy
    k_cap = cfg.k_cap if k_cap is None else k_cap
    cache_key = (str(source), energy, k_cap, tuple(site_types or ()))
    if cache_key in _BASIS_CACHE:
        return _BASIS_CACHE[cache_key]
    sd, _ = load_adapter(cfg, source)
    bases = {}
    for site_key, keys in lora_pairs(sd, site_types).items():
        U = donor_write_basis(cfg, sd[keys['B']], sd[keys['A']], energy=energy, k_cap=k_cap)
        if U.numel():
            bases[site_key] = U
    _BASIS_CACHE[cache_key] = bases
    return bases


def _removed(U, B):
    import torch
    return float(torch.linalg.norm(U @ B) / torch.linalg.norm(B))


def project_adapter(cfg, source, target, strength, mode='subtract',
                    site_types=None, energy=None, k_cap=None, with_control=True):
    """B' = B + sign*lambda*U^T(U B) per (layer, site). A untouched.

    Returns (new_state_dict, target_config, stats). `stats` carries the removed-energy
    diagnostic per site plus a random-U control at equal rank; chance level is
    sqrt(k/out), so a donor number at chance means there is no shared subspace there.
    """
    import torch
    energy = cfg.energy if energy is None else energy
    k_cap = cfg.k_cap if k_cap is None else k_cap
    sign = {'subtract': -1.0, 'add': +1.0}[mode]
    bases = donor_bases(cfg, source, site_types=site_types, energy=energy, k_cap=k_cap)
    tgt_sd, tgt_cfg = load_adapter(cfg, target)
    pairs = lora_pairs(tgt_sd, site_types)

    new_sd = dict(tgt_sd)
    stats, skipped = [], []
    for site_key, keys in sorted(pairs.items()):
        U = bases.get(site_key)
        B = tgt_sd[keys['B']]
        if U is None:
            skipped.append(site_key)
            continue
        if U.shape[1] != B.shape[0]:
            skipped.append(site_key)
            continue
        B32 = B.to(torch.float32)
        B_new = B32 + sign * float(strength) * (U.T @ (U @ B32))
        new_sd[keys['B']] = B_new.to(B.dtype).contiguous()

        layer, site_type = site_key
        k, out = U.shape
        row = {'layer': layer, 'site': site_type, 'k': k,
               'removed': _removed(U, B32), 'chance': (k / out) ** 0.5}
        if with_control:
            # zlib.crc32, not hash(): str hashing is salted per process, which would
            # make the control irreproducible across sessions.
            seed = (1000 * layer + zlib.crc32(site_type.encode()) % 997)
            Ur = torch.linalg.qr(torch.randn(out, k, generator=torch.Generator().manual_seed(seed)))[0].T
            row['random_U'] = _removed(Ur, B32)
        stats.append(row)

    if not stats:
        raise RuntimeError(f'no (layer, site) matched between {source} and {target}')
    if skipped:
        print(f'  note: {len(skipped)} site(s) had no donor basis or a shape mismatch, left unedited')
    return new_sd, tgt_cfg, stats


def save_projected_adapter(new_sd, adapter_cfg, out_dir, record=None):
    """Write config + weights. No PeftModel, no base model, no staging dance."""
    from safetensors.torch import save_file as st_save
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'adapter_config.json').write_text(json.dumps(adapter_cfg, indent=2) + '\n')
    st_save({k: v.contiguous() for k, v in new_sd.items()},
            str(out_dir / 'adapter_model.safetensors'), metadata={'format': 'pt'})
    if record is not None:
        (out_dir / 'run_metadata.json').write_text(json.dumps(record, indent=2) + '\n')
    return out_dir


def summarise_stats(stats, label=''):
    """Median removed energy per site type, against the random-U control."""
    if pd is None:
        med = sorted(s['removed'] for s in stats)[len(stats) // 2]
        print(f'{label} median removed={med:.4f} over {len(stats)} sites')
        return None
    df = pd.DataFrame(stats)
    by_site = df.groupby('site')[['removed', 'random_U', 'chance', 'k']].median()
    by_site['vs_chance'] = by_site['removed'] / by_site['chance']
    print(f'{label}removed energy by site type (median over layers):')
    print(by_site.round(4).to_string())
    beat = (df['removed'] > df['random_U']).mean()
    print(f'\n  overall median removed : {df["removed"].median():.4f}')
    print(f'  random-U control       : {df["random_U"].median():.4f}')
    print(f'  sites above control    : {100 * beat:.0f}% of {len(df)}')
    if beat < 0.5:
        print('  !! the donor basis is at or below chance — there is no shared write '
              'subspace to remove, and no strength will change behaviour')
    return df


def adapter_weights_present(path):
    path = Path(path)
    return path.is_dir() and (path / 'adapter_config.json').exists() and any((path / w).exists() for w in WEIGHT_NAMES)


def build_projected_adapter(cfg, source, target, mode, strength, force=False, verbose=True, record=None):
    """Build one adapter and return (path, stats). Skips if weights already exist."""
    out_dir = Path(adapter_dir(cfg, source, target, mode, strength))
    rec = record or make_record(cfg, source, target, mode, strength)
    if adapter_weights_present(out_dir) and not force:
        if verbose:
            print(f'[skip build] {rec["run_name"]} (weights present)')
        return out_dir, None
    new_sd, adapter_cfg, stats = project_adapter(cfg, source, target, strength, mode=mode,
                                         site_types=SITE_TYPES)
    save_projected_adapter(new_sd, adapter_cfg, out_dir, record=rec)
    if verbose:
        print(f'[built] {rec["run_name"]} -> {out_dir}')
    return out_dir, stats


def _self_test(cfg, out=64, r=32, inp=128, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    B_d, A_d = rnd(out, r), rnd(r, inp)
    B_t, A_t = rnd(out, r), rnd(r, inp)
    s_d, s_t = 2.0, 0.5

    U = donor_write_basis(cfg, B_d.float(), A_d.float()).double()
    k = U.shape[0]

    # 1. orthonormal rows, k respects the cap. U is built in float32, so the tolerance
    # is float32 orthogonality error, not float64.
    assert k <= cfg.k_cap, k
    assert (U @ U.T - torch.eye(k, dtype=U.dtype)).abs().max() < 1e-5

    # 2. the r x r SVD really gives the top singular subspace of the full dW_d.
    # Principal angles: svdvals(U @ Uf) are all 1 iff the spans agree. Loose tolerance
    # because U is float32 and near-degenerate singular values make the top-k subspace
    # ill-conditioned; a genuinely wrong subspace lands nowhere near 1.
    dW_d = s_d * (B_d @ A_d)
    Uf = torch.linalg.svd(dW_d)[0][:, :k]
    ang = torch.linalg.svdvals(U @ Uf)
    assert (ang - 1).abs().max() < 1e-3, ang

    # 3. EXACTNESS: editing B alone delivers the full left projection. Note this is an
    # algebraic identity for *any* U -- (I - l*U^T U) s B A == s (B - l*U^T(U B)) A --
    # so it holds to float64 roundoff and does not depend on U being orthonormal.
    lam = 0.7
    dW_t = s_t * (B_t @ A_t)
    want = dW_t - lam * (U.T @ (U @ dW_t))
    got = s_t * ((B_t - lam * (U.T @ (U @ B_t))) @ A_t)
    rel = float(torch.linalg.norm(want - got) / torch.linalg.norm(dW_t))
    assert rel < 1e-10, rel

    # 4. exactly affine in lambda
    proj = lambda x: B_t - x * (U.T @ (U @ B_t))
    B0, B05, B1 = proj(0.0), proj(0.5), proj(1.0)
    assert ((B05 - B0) - 0.5 * (B1 - B0)).abs().max() < 1e-10

    # 5. lambda = 0 is the identity, bit for bit
    assert torch.equal(B0, B_t)

    # 6. the diagnostic is calibrated: an unrelated target sits at sqrt(k/out)
    OUT, R = 3584, 32
    Bd2, Ad2 = rnd(OUT, R), rnd(R, OUT)
    U2 = donor_write_basis(cfg, Bd2.float(), Ad2.float()).double()
    k2 = U2.shape[0]
    chance = (k2 / OUT) ** 0.5
    unrelated = _removed(U2, rnd(OUT, R))
    shared = _removed(U2, Bd2 @ rnd(R, R) + 0.1 * rnd(OUT, R))
    assert abs(unrelated - chance) < 0.35 * chance, (unrelated, chance)
    assert shared > 3 * chance, (shared, chance)

    print('self-test passed')
    print(f'  k={k} (cap {cfg.k_cap}), orthonormal rows, span matches full SVD')
    print(f'  exactness   : rel err {rel:.1e}   <- the property the old pipeline lost')
    print('  linear in lambda, lambda=0 bit-identical')
    print(f'  diagnostic  : at out={OUT}, k={k2}: chance={chance:.4f}  '
          f'unrelated={unrelated:.4f}  shared-subspace={shared:.4f}')
