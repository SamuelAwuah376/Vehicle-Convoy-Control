"""The compiled lidar box fit (processing/_lidar_jit.py) against fit_box's numpy path (lidar._fit_box_py): the same
pose, covariance, sides and velocity sensitivities to rounding, on ray-cast clusters with noise, strays, free and
occluded ends, refinement on and off. Also checks that the kernel's constants are lidar.py's.

A fit near a discrete boundary (a point at RANSAC's tolerance, an ICP step at its limit) can jump when its inputs move
by rounding; such clusters are found by perturbing the numpy path's own input by 1e-15 and are left out (a few per
cent of these random views, none of the 1,458 fits of v9 seed 1000's first 3 s)."""
import numpy as np
import pytest
from nohesi.config import scenario_v9
from nohesi.perception import _jit
from nohesi.perception.config import default_lidars
from nohesi.perception.geometry import car_boxes, raycast
from nohesi.perception.processing import _lidar_jit as K
from nohesi.perception.processing import lidar as L

VEH = scenario_v9().vehicle
FRONT, REAR = default_lidars()


def test_kernel_constants_are_lidar_constants():
    for name in ('LSHAPE_POINTS', 'GATE', 'POSE_SIGMA', 'CORNER_MIN', 'STRAY_MAX', 'TAN_VIS', 'HEADING_PRIOR',
                 'ICP_ITERS', 'ICP_ROUNDS', 'HEADINGS', '_FACE_AX', '_FACE_SG'):
        assert np.array_equal(getattr(K, name.lstrip('_')), getattr(L, name)), name
    assert (K.STEP_MAX_C, K.STEP_MAX_H) == L.STEP_MAX
    assert L.LSHAPE_MIN == 8      # the kernel's own threshold for the L-shape search


def clusters(n_cases, seed=7):
    """Ray-cast clusters of one car each: (points, times, sensor, free, velocity, velocity_sigma, refine, sigma)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_cases:
        x = rng.uniform(-70.0, 140.0)
        y = rng.choice([-3.7, 0.0, 3.7])+rng.normal(0.0, 0.4)
        state = np.array([[x, y, rng.normal(0.0, 0.04)]])
        box = car_boxes(state, VEH)
        o = np.array([0.0, rng.normal(0.0, 1.5)])
        a = np.arctan2(box[0, :, 1]-o[1], box[0, :, 0]-o[0])
        if np.ptp(a) > np.pi:
            continue
        az = np.arange(a.min()-0.01, a.max()+0.01, FRONT.beam_spacing*rng.choice([1, 1, 3]))
        r, _ = raycast(o, az, box, FRONT.range_max)
        ok = np.isfinite(r) & (rng.random(len(r)) > 0.05)
        if ok.sum() < 3:
            continue
        r = r[ok]+FRONT.range_sigma*rng.standard_normal(ok.sum())
        if rng.random() < 0.3:                       # a mirror-like stray a few cm to 0.4 m off the face
            k = rng.integers(len(r))
            r[k] -= rng.uniform(0.05, 0.4)
        P = o+r[:, None]*np.column_stack((np.cos(az[ok]), np.sin(az[ok])))
        t = np.sort(rng.uniform(0.0, 0.1, len(P)))
        out.append((P, t, o, (bool(rng.random() < 0.6), bool(rng.random() < 0.6)),
                    np.array([rng.normal(0.0, 1.0), rng.normal(0.0, 0.2)]), (rng.uniform(0.3, 5.0), 1.0),
                    bool(rng.random() < 0.8), 0.003))
    return out


@pytest.mark.skipif(not _jit.ENABLED, reason='numba is off (NOHESI_JIT=0 or not installed)')
def test_fit_box_kernel_matches_numpy_path():
    n_sides = n_edge = 0
    for P, t, o, free, v, vs, refine, sm in clusters(400):
        a, b = {}, {}
        z0 = L._fit_box_py(P, t, o, VEH, FRONT, free, v, vs, refine, sm, None)[0]
        if any(np.abs(L._fit_box_py(P*(1+k*1e-15), t, o, VEH, FRONT, free, v, vs, refine, sm, None)[0]-z0).max() > 1e-9
               for k in (1, -1)):
            n_edge += 1
            continue
        z1, R1, s1, t1 = L.fit_box(P, t, o, VEH, FRONT, free, v, vs, refine, sm, out=a)
        z2, R2, s2, t2 = L._fit_box_py(P, t, o, VEH, FRONT, free, v, vs, refine, sm, b)
        assert s1 == s2
        assert abs(t1-t2) < 1e-12
        assert np.allclose(z1, z2, rtol=0, atol=1e-9)
        scale = np.abs(R2).max()
        assert np.allclose(R1, R2, rtol=0, atol=1e-9*scale)
        assert np.allclose(a['dz_dv'], b['dz_dv'], rtol=0, atol=1e-9*max(np.abs(b['dz_dv']).max(), 1e-12))
        assert np.allclose(a['R_fit'], b['R_fit'], rtol=0, atol=1e-9*np.abs(b['R_fit']).max())
        n_sides += s1 == 2
    assert n_sides > 20          # the cases include L-shaped views (two faces seen)
    assert n_edge <= 20


@pytest.mark.skipif(not _jit.ENABLED, reason='numba is off (NOHESI_JIT=0 or not installed)')
def test_lshape_kernel_matches_numpy():
    for P, t, o, *_ in clusters(150, seed=11):
        if len(P) >= L.LSHAPE_MIN:
            got = K.lshape_heading_kernel(np.ascontiguousarray(P), np.asarray(o, float), L.HEADINGS)
            assert abs(got-L.lshape_heading(P, o)) < 1e-12
