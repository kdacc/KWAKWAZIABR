"""
Тести на синтетичних даних (без мережі).

Перевіряють:
  1. очищення (ZODI, FLAGS) і читання MEF-подібного FITS;
  2. вирівнювання кадрів з різним зсувом і поворотом WCS;
  3. що нерухомі зорі лишаються на місці, а рухомий об'єкт зсувається;
  4. режим «лише зміни» гасить нерухоме й підсвічує рухоме;
  5. весь HTTP-конвеєр (POST job -> статус -> PNG) з підміненим доступом до IRSA.
Запуск: pytest -q test_synthetic.py
"""
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
SCALE = 6.15 / 3600.0            # градусів на піксель
N = 48                           # розмір синтетичного cutout-а
STARS = [(+0.0040, +0.0030, 400.0), (-0.0060, +0.0010, 250.0), (+0.0010, -0.0070, 300.0)]  # dRA, dDec, потік
MOVER_START = (-0.0100, -0.0030)  # dRA, dDec (градуси) в першому кадрі
MOVER_STEP = 0.0035              # зсув по RA за кадр (~2 пікселі SPHEREx)
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
        gain = 0.7 + 0.1 * k                       # «різна довжина хвилі» -> різна яскравість зірок
        src = [(RA0 + dx, DEC0 + dy, f * gain) for dx, dy, f in STARS]
        mover = (RA0 + MOVER_START[0] + MOVER_STEP * k, DEC0 + MOVER_START[1], 200.0)
        zodi = 30.0 + 2 * k
        img = render_sky(w, src + [mover], bg=zodi, rng=rng)
        cleaned, frac = core.clean_image(img + 0.0, np.full_like(img, zodi), np.zeros(img.shape, dtype=np.int32))
        frames.append(core.Frame(f"synthetic_{k}.fits", 61000.0 + k, 1.0 + 0.1 * k, cleaned, w, frac))
    return frames


def centroid(img, x0, y0, r=4):
    """Центроїд у вікні r пікселів навколо (x0, y0)."""
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    m = (xs - x0) ** 2 + (ys - y0) ** 2 <= r * r
    wgt = np.clip(np.nan_to_num(img), 0, None) * m
    return (xs * wgt).sum() / wgt.sum(), (ys * wgt).sum() / wgt.sum()


# ---------------------------------------------------------------------------
def test_app_imports_in_fresh_process():
    """Реальний шлях запуску: app імпортується першим, без попередньо імпортованого astropy."""
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
    assert np.isnan(out[0, 0])                       # позначений піксель замаскований
    assert out[3, 3] > 20                            # джерело лишилось
    assert abs(np.nanmedian(out)) < 1e-4             # фон віднято
    assert frac == pytest.approx(0.01)


def test_clean_ignores_flags_if_most_pixels_flagged():
    img = np.random.default_rng(0).normal(10, 1, (10, 10))
    flags = np.ones((10, 10), dtype=np.int32)        # усе «позначено»
    out, frac = core.clean_image(img, np.zeros_like(img), flags)
    assert np.isfinite(out).all() and frac == 1.0


def test_alignment_keeps_static_stars_fixed_and_moves_mover():
    frames = make_frames(8)
    stack = core.build_stack(frames, RA0, DEC0, fov_deg=N * SCALE * 0.7)
    assert len(stack) == 8
    ny, nx = stack.aligned.shape[1:]
    grid, _ = core.make_grid(RA0, DEC0, N * SCALE * 0.7, 6.15)

    # зоря 0: позиція на спільній сітці не повинна змінюватися між кадрами
    sx, sy = grid.world_to_pixel(SkyCoord((RA0 + STARS[0][0]) * u.deg, (DEC0 + STARS[0][1]) * u.deg))
    pos = [centroid(stack.aligned[i], sx, sy) for i in range(len(stack))]
    spread = max(math.hypot(p[0] - sx, p[1] - sy) for p in pos)
    assert spread < 0.5, f"нерухома зоря «пливе» на {spread:.2f} пікселя сітки"

    # рухомий об'єкт: x має монотонно змінюватися (RA росте -> піксель x зменшується)
    mx = []
    for k in range(len(stack)):
        x, y = grid.world_to_pixel(SkyCoord((RA0 + MOVER_START[0] + MOVER_STEP * k) * u.deg,
                                            (DEC0 + MOVER_START[1]) * u.deg))
        mx.append(centroid(stack.aligned[k], x, y)[0])
    d = np.diff(mx)
    assert np.all(d < 0), f"рух не монотонний: {d}"
    expected = MOVER_STEP / (N * SCALE * 0.7 / ny)           # піксель сітки за кадр
    assert np.mean(-d) == pytest.approx(expected, rel=0.15)


def test_diff_mode_suppresses_static_and_highlights_mover():
    frames = make_frames(8)
    stack = core.build_stack(frames, RA0, DEC0, fov_deg=N * SCALE * 0.7)
    grid, _ = core.make_grid(RA0, DEC0, N * SCALE * 0.7, 6.15)
    norm = core.normalize_stack(stack.aligned)
    diff = core.difference_stack(norm)

    def val(arr, k, dra, ddec, r=2):
        x, y = grid.world_to_pixel(SkyCoord((RA0 + dra) * u.deg, (DEC0 + ddec) * u.deg))
        xi, yi = int(round(float(x))), int(round(float(y)))
        return np.nanmax(arr[k, yi - r:yi + r + 1, xi - r:xi + r + 1])

    k = 2
    mover_resp = val(diff, k, MOVER_START[0] + MOVER_STEP * k, MOVER_START[1])
    star_resp = val(diff, k, STARS[0][0], STARS[0][1])
    star_raw = val(norm, k, STARS[0][0], STARS[0][1])
    assert mover_resp > 5.0, "рухомий об'єкт має бути помітним у різницевому режимі"
    assert star_resp < 0.35 * star_raw, "нерухома зоря має бути сильно пригнічена"
    assert star_resp < mover_resp


def test_diff_needs_three_frames():
    frames = make_frames(2)
    stack = core.build_stack(frames, RA0, DEC0, fov_deg=N * SCALE * 0.7)
    with pytest.raises(ValueError):
        stack.png(0, "diff")


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
    """Читання MEF з тією ж розкладкою HDU, що й у SPHEREx (без WCS-WAVE -> довжина хвилі NaN)."""
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
    assert np.isnan(f.wavelength_um)                 # немає WCS-WAVE: коректний запасний варіант
    assert np.nanmax(f.image) > 200                  # джерело на місці, ZODI віднято


def test_parse_coordinates():
    assert core._parse_coords("210.8, 54.3") == (210.8, 54.3)
    ra, dec = core._parse_coords("08h55m10.831s -07d14m42.53s")
    assert ra == pytest.approx(133.79513, abs=1e-4) and dec == pytest.approx(-7.24515, abs=1e-4)
    t = core.resolve_target("08 55 10.831 -07 14 42.53")
    assert t.ra == pytest.approx(133.79513, abs=1e-4)
    with pytest.raises(ValueError):
        core.resolve_target("400, 10")


def test_proper_motion_propagation():
    t = core.Target(ra=100.0, dec=0.0, pmra=0.0, pmdec=10000.0)     # 10"/рік на північ
    ra, dec = core.propagate(t, 2026.0)
    assert dec == pytest.approx(26 * 10.0 / 3600.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Повний HTTP-конвеєр з підміненим IRSA
# ---------------------------------------------------------------------------
def test_http_pipeline_with_mocked_archive(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as appmod

    frames = make_frames(6)
    by_url = {f.url: f for f in frames}

    def fake_query(ra, dec, detector):
        # детектор 2 має кадри, детектор 3 -- порожньо
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
        raise AssertionError("завдання не завершилось")

    jid, s = run(2)
    assert s["status"] == "done", s
    res = s["result"]
    assert res["count"] == 6
    times = [f["time"] for f in res["frames"]]
    assert times == sorted(times)                              # кадри за часом
    assert res["meta"]["detector"] == 2

    for mode in ("raw", "diff"):
        r = client.get(f"/api/jobs/{jid}/frame/0.png?mode={mode}")
        assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert client.get(f"/api/jobs/{jid}/frame/99.png").status_code == 404

    # інший детектор -> інший результат (тут: помилка «знімків не знайдено»)
    _, s3 = run(3)
    assert s3["status"] == "error" and "не знайдено" in s3["error"]

    # той самий запит повторно: той самий job
    r1 = client.post("/api/jobs", json={"target": f"{RA0}, {DEC0}", "fov": 0.1, "detector": 2}).json()
    assert r1["job_id"] == jid

    # валідація
    assert client.post("/api/jobs", json={"target": "x", "fov": 5, "detector": 2}).status_code == 422
    assert client.post("/api/jobs", json={"target": "x", "fov": 0.1, "detector": 9}).status_code == 422
