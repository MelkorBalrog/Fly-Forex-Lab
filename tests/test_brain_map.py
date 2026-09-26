"""MaleCNS structure map: schematic compartments, real |x|."""

from __future__ import annotations

import base64

import numpy as np

from flyfx.ui.brain_map import (
    REGION_INDEX,
    REGION_IDS,
    CnsAtlas,
    pack_cns_activity,
    pack_cns_maps,
)


def _atlas() -> CnsAtlas:
    n = 400
    sc_labels = [
        "ol_intrinsic",
        "cb_intrinsic",
        "vnc_intrinsic",
        "visual_projection",
        "ascending_neuron",
        "vnc_motor",
        "cb_sensory",
    ]
    sc_index = np.zeros(n, dtype=np.int32)
    sc_index[140:240] = 0
    sc_index[240:300] = 1
    sc_index[300:360] = 2
    sc_index[360:380] = 3
    sc_index[380:390] = 4
    sc_index[390:396] = 5
    sc_index[396:400] = 6
    return CnsAtlas.build(
        n=n,
        retina=np.arange(0, 40, dtype=np.int32),
        descending=np.arange(40, 70, dtype=np.int32),
        sugar=np.arange(70, 85, dtype=np.int32),
        da=np.arange(85, 100, dtype=np.int32),
        ol_sensory=np.arange(100, 120, dtype=np.int32),
        vnc_sensory=np.arange(120, 140, dtype=np.int32),
        lamina=np.arange(20, 50, dtype=np.int32),
        sc_index=sc_index,
        sc_labels=sc_labels,
    )


def test_atlas_is_t_shaped():
    atlas = _atlas()
    assert atlas.idx.size >= 200
    assert atlas.x.min() < 0.25
    assert atlas.x.max() > 0.70
    assert 0.04 <= float(atlas.y.min()) <= float(atlas.y.max()) <= 0.96
    assert 0.04 <= float(atlas.z.min()) <= float(atlas.z.max()) <= 0.96
    blob = atlas.layout_blob()
    assert blob["n"] == atlas.idx.size
    xy = np.frombuffer(base64.b64decode(blob["xy"]), dtype=np.int16)
    z = np.frombuffer(base64.b64decode(blob["z"]), dtype=np.int16)
    assert xy.size == atlas.idx.size * 2
    assert z.size == atlas.idx.size
    assert set(atlas.counts) == set(REGION_IDS)
    assert atlas.counts["retina_bull"] == 20
    assert atlas.counts["sugar"] == 15


def test_cord_is_round_and_eyes_are_lenses():
    atlas = _atlas()
    bull = atlas.rid == REGION_INDEX["retina_bull"]
    assert int(np.count_nonzero(bull)) >= 8
    # Face of the eye is taller than its depth, so a turn is a lens, not a block.
    assert float(np.std(atlas.y[bull])) > float(np.std(atlas.z[bull])) * 1.15
    assert float(np.std(atlas.z[bull])) < 0.09
    cord = np.isin(
        atlas.rid,
        [REGION_INDEX["vnc_intrinsic"], REGION_INDEX["vnc_sensory"]],
    )
    dy = atlas.y[cord] - 0.5
    dz = atlas.z[cord] - 0.5
    assert abs(float(np.std(dy)) - float(np.std(dz))) < 0.02


def test_pack_activity_tracks_driven_regions():
    atlas = _atlas()
    x = np.zeros(atlas.n, dtype=np.float32)
    x[atlas.masks["retina_bull"]] = 0.9
    x[atlas.masks["descending"]] = 0.4
    packed = pack_cns_activity(atlas, x)
    assert packed["reg"]["retina_bull"] > packed["reg"]["retina_bear"]
    assert packed["reg"]["descending"] > packed["reg"]["vnc_intrinsic"]
    assert "retina_bull" in packed["hot"]
    act = np.frombuffer(base64.b64decode(packed["act"]), dtype=np.uint8)
    assert act.size == atlas.idx.size
    assert act.max() > 100


def test_pack_cns_maps_skips_missing_atlas():
    class _B:
        pass

    x = np.zeros(10, dtype=np.float32)
    assert pack_cns_maps(_B(), {"trend": x}) == {}


def test_dash_html_has_cns_card():
    from flyfx.paths import DASH_HTML

    html = DASH_HTML.read_text(encoding="utf-8")
    assert 'id="card-cns"' in html
    assert "/api/cns-layout" in html
    assert "MALECNS" in html
    assert "sphereSprite" in html
    assert "paintCnsCanvas" in html
    assert "52vh" not in html
    assert "720 / 380" in html
    assert "720px" in html
    assert "0.18 + 0.82" in html
    assert "0.42 + 0.58" not in html
