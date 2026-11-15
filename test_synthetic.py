import math
import time

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

import spherex_core as core

RA0, DEC0 = 150.0, 2.0
SCALE = 6.15 / 3600.0
N = 48
STARS = [(+0.0040, +0.0030, 400.0), (-0.0060, +0.0010, 250.0), (+0.0010, -0.0070, 300.0)]
MOVER_START = (-0.0100, -0.0030)
MOVER_STEP = 0.0035
SIGMA_PIX = 1.4


def make_wcs(shift_pix=(0.0, 0.0), rot_deg=0.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [N / 2 + shift_pix[0], N / 2 + shift_pix[1]]
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    w.wcs.cd = np.array([[-SCALE * c, SCALE * s], [SCALE * s, SCALE * c]])
    return w


def render_sky(w, sources, bg=5.0, noise=0.4, rng=None):
    yy, xx = np.mgrid[0:N, 0:N]
    img = np.full((N, N), bg, dtype=np.float64)
    for ra, dec, flux in sources:
        x, y = w.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        img += flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * SIGMA_PIX ** 2))
    if rng is not None:
        img += rng.normal(0, noise, img.shape)
    return img


def make_frames(n_frames=8, seed=1):
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(n_frames):
        w = make_wcs(shift_pix=(rng.uniform(-3, 3), rng.uniform(-3, 3)), rot_deg=rng.uniform(-8, 8))
        gain = 0.7 + 0.1 * k
        src = [(RA0 + dx, DEC0 + dy, f * gain) for dx, dy, f in STARS]
        mover = (RA0 + MOVER_START[0] + MOVER_STEP * k, DEC0 + MOVER_START[1], 200.0)
        zodi = 30.0 + 2 * k
        img = render_sky(w, src + [mover], bg=zodi, rng=rng)
        cleaned, frac = core.clean_image(img + 0.0, np.full_like(img, zodi), np.zeros(img.shape, dtype=np.int32))
        frames.append(core.Frame(f"synthetic_{k}.fits", 61000.0 + k, 1.0 + 0.1 * k, cleaned, w, frac))
    return frames


def centroid(img, x0, y0, r=4):
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    m = (xs - x0) ** 2 + (ys - y0) ** 2 <= r * r
    wgt = np.clip(np.nan_to_num(img), 0, None) * m
    return (xs * wgt).sum() / wgt.sum(), (ys * wgt).sum() / wgt.sum()


def test_app_imports_in_fresh_process():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-c", "import app; print('ok')"],
                       capture_output=True, text=True, cwd=".")
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-500:]


def test_clean_removes_zodi_and_flags():
    img = np.full((10, 10), 50.0); img[3, 3] = 80.0
    zodi = np.full((10, 10), 40.0)
    flags = np.zeros((10, 10), dtype=np.int32); flags[0, 0] = 4
    out, frac = core.clean_image(img, zodi, flags)
    assert np.isnan(out[0, 0])
    assert out[3, 3] > 20
    assert abs(np.nanmedian(out)) < 1e-4
    assert frac == pytest.approx(0.01)


def test_clean_ignores_flags_if_most_pixels_flagged():
    img = np.random.default_rng(0).normal(10, 1, (10, 10))
    flags = np.ones((10, 10), dtype=np.int32)
    out, frac = core.clean_image(img, np.zeros_like(img), flags)
    assert np.isfinite(out).all() and frac == 1.0


def test_alignment_keeps_static_stars_fixed_and_moves_mover():
    frames = make_frames(8)
    stack = core.build_stack(frames, RA0, DEC0, fov_deg=N * SCALE * 0.7)
    assert len(stack) == 8
    ny, nx = stack.aligned.shape[1:]
    grid, _ = core.make_grid(RA0, DEC0, N * SCALE * 0.7, 6.15)


    sx, sy = grid.world_to_pixel(SkyCoord((RA0 + STARS[0][0]) * u.deg, (DEC0 + STARS[0][1]) * u.deg))
    pos = [centroid(stack.aligned[i], sx, sy) for i in range(len(stack))]
    spread = max(math.hypot(p[0] - sx, p[1] - sy) for p in pos)
    assert spread < 0.5, f"the static star is moving on the grid: {spread:.2f}"


    mx = []
    for k in range(len(stack)):
        x, y = grid.world_to_pixel(SkyCoord((RA0 + MOVER_START[0] + MOVER_STEP * k) * u.deg,
                                            (DEC0 + MOVER_START[1]) * u.deg))
        mx.append(centroid(stack.aligned[k], x, y)[0])
    d = np.diff(mx)
    assert np.all(d < 0), f"the movement is not monotonous: {d}"
    expected = MOVER_STEP / (N * SCALE * 0.7 / ny)
    assert np.mean(-d) == pytest.approx(expected, rel=0.15)


def test_png_output_and_north_up():
    frames = make_frames(4)
    stack = core.build_stack(frames, RA0, DEC0, fov_deg=N * SCALE * 0.7)
    png = stack.png(0, "raw")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    import io
    im = Image.open(io.BytesIO(png))
    assert min(im.size) >= 512


def test_load_cutout_reads_local_mef(tmp_path):
    w = make_wcs()
    img = render_sky(w, [(RA0, DEC0, 300.0)], bg=40.0).astype(np.float32)
    hdr = w.to_header()
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(img, header=hdr, name="IMAGE"),
        fits.ImageHDU(np.zeros(img.shape, dtype=np.int32), name="FLAGS"),
        fits.ImageHDU(np.ones(img.shape, dtype=np.float32), name="VARIANCE"),
        fits.ImageHDU(np.full(img.shape, 40.0, dtype=np.float32), name="ZODI"),
    ])
    path = tmp_path / "fake_cutout.fits"
    hdul.writeto(path)
    f = core.load_cutout(str(path), SkyCoord(RA0 * u.deg, DEC0 * u.deg), 61234.5, retries=0)
    assert f is not None and f.time == 61234.5
    assert np.isnan(f.wavelength_um)
    assert np.nanmax(f.image) > 200


def test_parse_coordinates():
    assert core._parse_coords("210.8, 54.3") == (210.8, 54.3)
    ra, dec = core._parse_coords("08h55m10.831s -07d14m42.53s")
    assert ra == pytest.approx(133.79513, abs=1e-4) and dec == pytest.approx(-7.24515, abs=1e-4)
    t = core.resolve_target("08 55 10.831 -07 14 42.53")
    assert t.ra == pytest.approx(133.79513, abs=1e-4)
    with pytest.raises(ValueError):
        core.resolve_target("400, 10")


def test_names_are_rejected_without_any_network(monkeypatch):
    import socket

    def boom(*a, **k):
        raise AssertionError("network call in resolve_target")
    monkeypatch.setattr(socket.socket, "connect", boom)
    for name in ("Barnard's star", "eps Ind B", "RW Aur", "x"):
        with pytest.raises(ValueError):
            core.resolve_target(name)
    assert not hasattr(core, "resolve_name") and not hasattr(core, "propagate")


def test_normalize_target_gives_single_key():
    a = core.normalize_target("133.795 -7.245")
    b = core.normalize_target("133.795, -7.245")
    assert a == b == "133.79500, -7.24500"
    assert core.normalize_target("08h55m10.831s -07d14m42.53s") == \
        core.normalize_target("08 55 10.831 -07 14 42.53")
    assert core.normalize_target("359.999999 1") == "0.00000, 1.00000"


def test_presets_are_valid_coordinates():
    import app as appmod
    assert len(appmod.PRESETS) >= 5
    for p in appmod.PRESETS:
        t = core.resolve_target(f"{p['ra']}, {p['dec']}")
        assert 0.01 <= p["fov"] <= 0.3 and t.label


def test_http_pipeline_with_mocked_archive(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as appmod

    frames = make_frames(6)
    by_url = {f.url: f for f in frames}

    def fake_query(ra, dec, detector):

        return [(f.url, f.time) for f in frames] if detector == 2 else []

    monkeypatch.setattr(core, "query_frames", fake_query)
    monkeypatch.setattr(core, "cutout_url", lambda base, ra, dec, size: base)
    monkeypatch.setattr(core, "load_cutout", lambda url, coord, t, retries=2: by_url[url])
    monkeypatch.setattr(core, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(core, "mjd_to_jyear", lambda t: 2026.0)

    client = TestClient(appmod.app)
    assert client.get("/").status_code == 200
    assert len(client.get("/api/detectors").json()) == 6

    def run(det, target=f"{RA0}, {DEC0}"):
        r = client.post("/api/jobs", json={"target": target, "fov": 0.2 * 0.5, "detector": det})
        assert r.status_code == 200, r.text
        jid = r.json()["job_id"]
        for _ in range(100):
            s = client.get(f"/api/jobs/{jid}").json()
            if s["status"] in ("done", "error"):
                return jid, s
            time.sleep(0.1)
        raise AssertionError("the task did not complete in time")

    jid, s = run(2)
    assert s["status"] == "done", s
    res = s["result"]
    assert res["count"] == 6
    times = [f["time"] for f in res["frames"]]
    assert times == sorted(times)
    assert res["meta"]["detector"] == 2

    for mode in ("raw", "diff"):
        r = client.get(f"/api/jobs/{jid}/frame/0.png")
        assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert client.get(f"/api/jobs/{jid}/frame/99.png").status_code == 404

    _, s3 = run(3)
    assert s3["status"] == "error" and "not found" in s3["error"]

    r1 = client.post("/api/jobs", json={"target": f"{RA0}, {DEC0}", "fov": 0.1, "detector": 2}).json()
    assert r1["job_id"] == jid

    r = client.post("/api/jobs", json={"target": "Barnard's star", "fov": 0.1, "detector": 2})
    assert r.status_code == 422 and "coordinates" in r.json()["detail"].lower()
    assert len(client.get("/api/presets").json()) >= 5
    assert client.post("/api/jobs", json={"target": "x", "fov": 5, "detector": 2}).status_code == 422
    assert client.post("/api/jobs", json={"target": "x", "fov": 0.1, "detector": 9}).status_code == 422


@pytest.fixture
def cancel_env(monkeypatch, tmp_path):
    import threading
    from fastapi.testclient import TestClient
    import app as appmod

    frames = make_frames(6)
    by_url = {f.url: f for f in frames}
    gate = threading.Event()
    slow = {"on": True}

    def fake_load(url, coord, t, retries=2):
        if slow["on"]:
            gate.wait(timeout=10)
        return by_url[url]

    monkeypatch.setattr(core, "query_frames", lambda ra, dec, det: [(f.url, f.time) for f in frames])
    monkeypatch.setattr(core, "cutout_url", lambda base, ra, dec, size: base)
    monkeypatch.setattr(core, "load_cutout", fake_load)
    monkeypatch.setattr(core, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(core, "RAW_CACHE_DIR", str(tmp_path / "cache" / "raw"))
    monkeypatch.setattr(core, "mjd_to_jyear", lambda t: 2026.0)
    appmod.JOBS.clear()
    appmod.JOB_BY_KEY.clear()
    client = TestClient(appmod.app)
    yield client, appmod, slow, gate
    gate.set()
    time.sleep(0.3)


def _post(client, det, target=f"{RA0}, {DEC0}"):
    r = client.post("/api/jobs", json={"target": target, "fov": 0.1, "detector": det})
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def _wait(client, jid, wanted, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"/api/jobs/{jid}").json()
        if s["status"] in wanted:
            return s
        time.sleep(0.05)
    raise AssertionError(f"{jid}: did not reach {wanted}, currently {s['status']}/{s['stage']}")


def test_cancel_during_download_is_immediate_and_discards_result(cancel_env):
    client, appmod, slow, gate = cancel_env
    jid = _post(client, 1)
    _wait(client, jid, ("running",))
    r = client.post(f"/api/jobs/{jid}/cancel")
    assert r.status_code == 200 and r.json() == {"ok": True, "status": "cancelled"}
    assert client.get(f"/api/jobs/{jid}").json()["status"] == "cancelled"
    gate.set()
    time.sleep(0.7)
    s = client.get(f"/api/jobs/{jid}").json()
    assert s["status"] == "cancelled" and "result" not in s
    assert appmod.JOBS[jid]["stack"] is None
    assert client.get(f"/api/jobs/{jid}/frame/0.png").status_code == 404


def test_cancelled_jobs_do_not_block_new_jobs(cancel_env):
    client, appmod, slow, gate = cancel_env
    ids = [_post(client, d) for d in (1, 2)]
    for i in ids:
        _wait(client, i, ("running",))
    for i in ids:
        client.post(f"/api/jobs/{i}/cancel")
    slow["on"] = False
    new = _post(client, 3)
    s = _wait(client, new, ("done", "error"), timeout=6)
    assert s["status"] == "done", s
    assert not gate.is_set()


def test_cancel_queued_job(cancel_env):
    client, appmod, slow, gate = cancel_env
    running = [_post(client, d) for d in (1, 2)]
    for i in running:
        _wait(client, i, ("running",))
    queued = _post(client, 3)
    assert client.get(f"/api/jobs/{queued}").json()["status"] == "queued"
    client.post(f"/api/jobs/{queued}/cancel")
    gate.set()
    for i in running:
        assert _wait(client, i, ("done", "error"))["status"] == "done"
    assert client.get(f"/api/jobs/{queued}").json()["status"] == "cancelled"
    assert appmod._slots._value == appmod.MAX_PARALLEL_JOBS


def test_cancel_is_idempotent_and_noop_for_finished(cancel_env):
    client, appmod, slow, gate = cancel_env
    slow["on"] = False
    jid = _post(client, 1)
    assert _wait(client, jid, ("done", "error"))["status"] == "done"
    assert client.post(f"/api/jobs/{jid}/cancel").json() == {"ok": True, "status": "done"}
    assert client.get(f"/api/jobs/{jid}").json()["status"] == "done"
    slow["on"] = True
    j2 = _post(client, 2)
    assert client.post(f"/api/jobs/{j2}/cancel").status_code == 200
    assert client.post(f"/api/jobs/{j2}/cancel").json()["status"] in ("cancelled", "done")
    assert client.post("/api/jobs/nope/cancel").status_code == 404


def test_same_request_after_cancel_starts_new_job(cancel_env):
    client, appmod, slow, gate = cancel_env
    a = _post(client, 1)
    _wait(client, a, ("running",))
    client.post(f"/api/jobs/{a}/cancel")
    slow["on"] = False
    b = _post(client, 1)
    assert b != a
    assert _wait(client, b, ("done", "error"))["status"] == "done"
    assert client.get(f"/api/jobs/{a}").json()["status"] == "cancelled"
    assert appmod._slots._value == appmod.MAX_PARALLEL_JOBS
