"""MaleCNS structure map: superclass compartments + live |x|.

The 166k connectome package has no soma xyz. Points are placed in a BANC-style
T (VNC left, brain + optic lobes right) from superclass / population labels.
Brightness is real network activity. Positions are a schematic, not measured.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field

import numpy as np

REGION_META: list[dict] = [
    {"id": "retina_bull", "label": "retina +", "hue": 168, "job": "bull half-retina (impulse UP)"},
    {"id": "retina_bear", "label": "retina −", "hue": 8, "job": "bear half-retina (impulse DOWN)"},
    {"id": "lamina", "label": "lamina", "hue": 185, "job": "first optic neuropil"},
    {"id": "ol_sensory", "label": "OL sensory", "hue": 195, "job": "optic-lobe sensory"},
    {"id": "ol_intrinsic", "label": "OL intrinsic", "hue": 210, "job": "optic-lobe local circuit"},
    {"id": "visual_projection", "label": "visual proj.", "hue": 230, "job": "OL → central brain"},
    {"id": "cb_sensory", "label": "CB sensory", "hue": 280, "job": "central-brain sensory"},
    {"id": "cb_intrinsic", "label": "CB intrinsic", "hue": 300, "job": "central brain"},
    {"id": "descending", "label": "descending", "hue": 268, "job": "brain → VNC motor intent"},
    {"id": "ascending", "label": "ascending", "hue": 250, "job": "VNC → brain"},
    {"id": "vnc_sensory", "label": "VNC sensory", "hue": 45, "job": "ventral-cord sense"},
    {"id": "vnc_intrinsic", "label": "VNC intrinsic", "hue": 155, "job": "ventral-cord local"},
    {"id": "vnc_motor", "label": "VNC motor", "hue": 35, "job": "leg / wing motor"},
    {"id": "sugar", "label": "sugar GRN", "hue": 48, "job": "23 taste cells"},
    {"id": "da", "label": "PAM/PPL DA", "hue": 328, "job": "dopamine traces"},
]

REGION_IDS = [r["id"] for r in REGION_META]
REGION_INDEX = {name: i for i, name in enumerate(REGION_IDS)}

# Sample budget: OLs dominate the animal; keep VNC/CB visible anyway.
_BUDGET = {
    "retina_bull": 140,
    "retina_bear": 140,
    "lamina": 280,
    "ol_sensory": 360,
    "ol_intrinsic": 1800,
    "visual_projection": 280,
    "cb_sensory": 180,
    "cb_intrinsic": 620,
    "descending": 260,
    "ascending": 180,
    "vnc_sensory": 240,
    "vnc_intrinsic": 520,
    "vnc_motor": 120,
    "sugar": 23,
    "da": 100,
}

_NOTE = (
    "MaleCNS T from superclass labels (optic lobes, central brain, neck, VNC). "
    "Not measured soma xyz — the 166k graph has none. Brightness is live |x|."
)


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _vnc_radius(x: np.ndarray) -> np.ndarray:
    """Cord radius along the long axis. Three thoracic bulbs on one tube."""
    r = np.full(x.shape, 0.042, dtype=np.float64)
    for bx, br, bw in ((0.13, 0.026, 0.040), (0.26, 0.038, 0.048), (0.37, 0.034, 0.046)):
        r = r + br * np.exp(-((x - bx) / bw) ** 2)
    return r


def _tube(
    rng: np.random.Generator,
    n: int,
    x0: float,
    x1: float,
    *,
    radius_scale: float = 1.0,
    shell: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filled cylinder along x. y and z share one radius so a turn stays round."""
    if n <= 0:
        z = np.zeros(0, dtype=np.float32)
        return z, z, z
    x = rng.uniform(x0, x1, n).astype(np.float64)
    rad = _vnc_radius(x) * float(radius_scale)
    ang = rng.random(n) * (2.0 * np.pi)
    # Bias toward the wall so the tube reads when it turns, without a hollow core.
    u = shell + (1.0 - shell) * np.sqrt(rng.random(n))
    rho = rad * u
    y = 0.50 + rho * np.sin(ang)
    z = 0.50 + rho * np.cos(ang)
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def _ellipsoid(
    rng: np.random.Generator,
    n: int,
    cx: float,
    cy: float,
    cz: float,
    rx: float,
    ry: float,
    rz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        z = np.zeros(0, dtype=np.float32)
        return z, z, z
    v = rng.normal(0.0, 1.0, size=(n, 3))
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
    rad = np.cbrt(rng.random(n))[:, None]
    p = v * rad
    return (
        (cx + rx * p[:, 0]).astype(np.float32),
        (cy + ry * p[:, 1]).astype(np.float32),
        (cz + rz * p[:, 2]).astype(np.float32),
    )


def _eye(
    rng: np.random.Generator,
    n: int,
    cx: float,
    cy: float,
    *,
    shell: float = 0.35,
    depth: float = 0.045,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Almond lens. Shallow in z so a turn shows an eye, not a block.

    The outline tapers at the left and right tips. Depth is thickest in the
    middle and the center bulges outward (larger x), the way a compound eye does.
    """
    if n <= 0:
        z = np.zeros(0, dtype=np.float32)
        return z, z, z
    # Hemisphere opening toward the brain (smaller x), dome outward.
    # Tall almond face, shallower depth than a ball, so a turn shows a lens.
    cos_a = np.sqrt(np.clip(shell + (1.0 - shell) * rng.random(n), 0.0, 1.0))
    sin_a = np.sqrt(np.clip(1.0 - cos_a * cos_a, 0.0, 1.0))
    phi = rng.random(n) * (2.0 * np.pi)
    pinch = 0.62 + 0.38 * np.abs(np.sin(phi))
    r = 0.115
    x = cx + r * 0.82 * cos_a
    y = cy + r * 1.20 * sin_a * np.sin(phi)
    z = 0.50 + (depth / 0.045) * r * 0.62 * sin_a * np.cos(phi) * pinch
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def _ang_bytes(rad: np.ndarray) -> np.ndarray:
    wrapped = np.mod(rad, 2.0 * np.pi)
    return np.clip(wrapped * (255.0 / (2.0 * np.pi)), 0, 255).astype(np.uint8)


def _pick(idx: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if idx.size == 0 or k <= 0:
        return np.array([], dtype=np.int32)
    if idx.size <= k:
        return idx.astype(np.int32, copy=False)
    take = rng.choice(idx.size, size=int(k), replace=False)
    return idx[take].astype(np.int32)


def _split_retina(retina: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if retina.size == 0:
        empty = np.array([], dtype=np.int32)
        return empty, empty
    mid = max(retina.size // 2, 1)
    return retina[:mid].astype(np.int32, copy=False), retina[mid:].astype(np.int32, copy=False)


def _sc_mask(sc_index: np.ndarray, labels: list[str], name: str) -> np.ndarray:
    if sc_index.size == 0 or name not in labels:
        return np.array([], dtype=np.int32)
    return np.flatnonzero(sc_index == labels.index(name)).astype(np.int32)


def region_masks(
    *,
    n: int,
    retina: np.ndarray,
    descending: np.ndarray,
    sugar: np.ndarray,
    da: np.ndarray,
    ol_sensory: np.ndarray,
    vnc_sensory: np.ndarray,
    lamina: np.ndarray | None = None,
    sc_index: np.ndarray | None = None,
    sc_labels: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Index arrays for every painted compartment. Empty if that set is missing."""
    labels = list(sc_labels or [])
    sc = sc_index if sc_index is not None and sc_index.size == n else np.array([], dtype=np.int32)
    bull, bear = _split_retina(retina)
    lamina = lamina if lamina is not None else np.array([], dtype=np.int32)
    return {
        "retina_bull": bull,
        "retina_bear": bear,
        "lamina": lamina.astype(np.int32, copy=False) if lamina.size else lamina,
        "ol_sensory": ol_sensory.astype(np.int32, copy=False) if ol_sensory.size else ol_sensory,
        "ol_intrinsic": _sc_mask(sc, labels, "ol_intrinsic"),
        "visual_projection": _sc_mask(sc, labels, "visual_projection"),
        "cb_sensory": _sc_mask(sc, labels, "cb_sensory"),
        "cb_intrinsic": _sc_mask(sc, labels, "cb_intrinsic"),
        "descending": descending.astype(np.int32, copy=False) if descending.size else descending,
        "ascending": _sc_mask(sc, labels, "ascending_neuron"),
        "vnc_sensory": vnc_sensory.astype(np.int32, copy=False) if vnc_sensory.size else vnc_sensory,
        "vnc_intrinsic": _sc_mask(sc, labels, "vnc_intrinsic"),
        "vnc_motor": _sc_mask(sc, labels, "vnc_motor"),
        "sugar": sugar.astype(np.int32, copy=False) if sugar.size else sugar,
        "da": da.astype(np.int32, copy=False) if da.size else da,
    }


def _place_region(
    name: str,
    idx: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y, z, angle. z is the anatomical depth, not a flat slab."""
    n = int(idx.size)
    empty = np.zeros(0, dtype=np.float32)
    if n == 0:
        return empty, empty, empty, empty

    even = (idx % 2) == 0
    ang = (rng.random(n) * (2.0 * np.pi)).astype(np.float32)

    def _both_eyes(shell: float, depth: float, cx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n0 = int(np.count_nonzero(even))
        n1 = n - n0
        x = np.empty(n, dtype=np.float32)
        y = np.empty(n, dtype=np.float32)
        z = np.empty(n, dtype=np.float32)
        if n0:
            x0, y0, z0 = _eye(rng, n0, cx, 0.26, shell=shell, depth=depth)
            x[even], y[even], z[even] = x0, y0, z0
        if n1:
            x1, y1, z1 = _eye(rng, n1, cx, 0.74, shell=shell, depth=depth)
            x[~even], y[~even], z[~even] = x1, y1, z1
        return x, y, z

    if name == "retina_bull":
        x, y, z = _eye(rng, n, 0.84, 0.26, shell=0.62, depth=0.038)
        return x, y, z, ang
    if name == "retina_bear":
        x, y, z = _eye(rng, n, 0.84, 0.74, shell=0.62, depth=0.038)
        return x, y, z, ang
    if name == "lamina":
        x, y, z = _both_eyes(0.45, 0.042, 0.84)
        return x, y, z, ang
    if name == "ol_sensory":
        x, y, z = _both_eyes(0.28, 0.048, 0.83)
        return x, y, z, ang
    if name == "ol_intrinsic":
        x, y, z = _both_eyes(0.08, 0.052, 0.86)
        return x, y, z, ang

    if name == "visual_projection":
        t = rng.random(n).astype(np.float64)
        dorsal = even
        x = 0.68 + 0.14 * (1.0 - t)
        y = np.where(dorsal, 0.32 + 0.16 * t, 0.68 - 0.16 * t)
        z = 0.50 + 0.03 * rng.normal(0.0, 1.0, n)
        return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32), ang

    if name in ("cb_intrinsic", "cb_sensory"):
        # Rounded loaf: taller than it is deep, so a turn is an oval, not a box.
        rz = 0.055 if name == "cb_sensory" else 0.072
        x, y, z = _ellipsoid(rng, n, 0.62, 0.50, 0.50, 0.09, 0.125, rz)
        return x, y, z, ang

    if name == "descending":
        x, y, z = _tube(rng, n, 0.50, 0.58, radius_scale=0.42, shell=0.15)
        return x, y, z, ang
    if name == "ascending":
        x, y, z = _tube(rng, n, 0.46, 0.56, radius_scale=0.55, shell=0.2)
        return x, y, z, ang
    if name == "vnc_sensory":
        x, y, z = _tube(rng, n, 0.06, 0.46, radius_scale=1.05, shell=0.72)
        return x, y, z, ang
    if name == "vnc_motor":
        x, y, z = _tube(rng, n, 0.08, 0.42, radius_scale=0.85, shell=0.35)
        return x, y, z, ang
    if name == "vnc_intrinsic":
        x, y, z = _tube(rng, n, 0.05, 0.48, radius_scale=1.0, shell=0.2)
        return x, y, z, ang
    if name == "sugar":
        x, y, z = _ellipsoid(rng, n, 0.60, 0.62, 0.50, 0.028, 0.03, 0.022)
        return x, y, z, ang
    if name == "da":
        x, y, z = _ellipsoid(rng, n, 0.64, 0.40, 0.50, 0.032, 0.03, 0.024)
        return x, y, z, ang

    x, y, z = _ellipsoid(rng, n, 0.62, 0.50, 0.50, 0.08, 0.08, 0.05)
    return x, y, z, ang


@dataclass
class CnsAtlas:
    n: int
    masks: dict[str, np.ndarray]
    idx: np.ndarray
    rid: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    ang: np.ndarray
    counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        n: int,
        retina: np.ndarray,
        descending: np.ndarray,
        sugar: np.ndarray,
        da: np.ndarray,
        ol_sensory: np.ndarray,
        vnc_sensory: np.ndarray,
        lamina: np.ndarray | None = None,
        sc_index: np.ndarray | None = None,
        sc_labels: list[str] | None = None,
        seed: int = 166700,
    ) -> "CnsAtlas":
        masks = region_masks(
            n=int(n),
            retina=retina,
            descending=descending,
            sugar=sugar,
            da=da,
            ol_sensory=ol_sensory,
            vnc_sensory=vnc_sensory,
            lamina=lamina,
            sc_index=sc_index,
            sc_labels=sc_labels,
        )
        rng = np.random.default_rng(int(seed))
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        zs: list[np.ndarray] = []
        angs: list[np.ndarray] = []
        rids: list[np.ndarray] = []
        ids: list[np.ndarray] = []
        counts: dict[str, int] = {}
        for name in REGION_IDS:
            full = masks.get(name, np.array([], dtype=np.int32))
            counts[name] = int(full.size)
            picked = _pick(full, int(_BUDGET.get(name, 80)), rng)
            if picked.size == 0:
                continue
            px, py, pz, pa = _place_region(name, picked, rng)
            xs.append(px)
            ys.append(py)
            zs.append(pz)
            angs.append(pa)
            rids.append(np.full(picked.size, REGION_INDEX[name], dtype=np.uint8))
            ids.append(picked)
        if not ids:
            idx = np.array([], dtype=np.int32)
            rid = np.array([], dtype=np.uint8)
            x = y = z = ang = np.array([], dtype=np.float32)
        else:
            idx = np.concatenate(ids).astype(np.int32)
            rid = np.concatenate(rids).astype(np.uint8)
            x = np.clip(np.concatenate(xs), 0.02, 0.98).astype(np.float32)
            y = np.clip(np.concatenate(ys), 0.06, 0.94).astype(np.float32)
            z = np.clip(np.concatenate(zs), 0.08, 0.92).astype(np.float32)
            ang = np.concatenate(angs).astype(np.float32)
        return cls(n=int(n), masks=masks, idx=idx, rid=rid, x=x, y=y, z=z, ang=ang, counts=counts)

    def layout_blob(self) -> dict:
        xy = np.empty(self.x.size * 2, dtype=np.int16)
        xy[0::2] = np.clip(self.x * 10000.0, 0, 10000).astype(np.int16)
        xy[1::2] = np.clip(self.y * 10000.0, 0, 10000).astype(np.int16)
        z = np.clip(self.z * 10000.0, 0, 10000).astype(np.int16)
        return {
            "n": int(self.idx.size),
            "neurons": int(self.n),
            "xy": _b64(xy),
            "z": _b64(z),
            "rid": _b64(self.rid),
            "ang": _b64(_ang_bytes(self.ang)),
            "regions": REGION_META,
            "counts": dict(self.counts),
            "note": _NOTE,
        }


def pack_cns_activity(atlas: CnsAtlas, x: np.ndarray) -> dict:
    """Region heat + sampled-cell |x|. Means come from the painted sample, not 89k gathers."""
    if x.size != atlas.n:
        x = np.zeros(atlas.n, dtype=np.float32)
    n_s = int(atlas.idx.size)
    if n_s:
        mag = np.abs(x[atlas.idx])
        cells = np.clip(mag * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        mag = np.zeros(0, dtype=np.float32)
        cells = np.zeros(0, dtype=np.uint8)
    reg: dict[str, float] = {name: 0.0 for name in REGION_IDS}
    fire: dict[str, float] = {name: 0.0 for name in REGION_IDS}
    if n_s:
        for rid, name in enumerate(REGION_IDS):
            sel = atlas.rid == rid
            if not np.any(sel):
                continue
            chunk = mag[sel]
            reg[name] = round(float(np.mean(chunk)), 3)
            fire[name] = round(float(np.mean(chunk > 0.35)), 3)
    ranked = sorted(reg.items(), key=lambda kv: kv[1], reverse=True)
    hot = [name for name, val in ranked if val >= 0.08][:4]
    return {
        "reg": reg,
        "fire": fire,
        "act": _b64(cells),
        "hot": hot,
        "peak": round(float(ranked[0][1]) if ranked else 0.0, 3),
    }


def pack_cns_maps(brain, xs: dict[str, np.ndarray | None]) -> dict:
    atlas = getattr(brain, "cns", None)
    if atlas is None:
        return {}
    out: dict[str, dict] = {}
    for name, x in xs.items():
        if x is None:
            continue
        out[str(name)] = pack_cns_activity(atlas, np.asarray(x, dtype=np.float32))
    return out
