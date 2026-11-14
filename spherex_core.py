"""
Ядро SPHEREx Time-Lapse.

Конвеєр:
  ціль (назва або RA/Dec) -> TAP-запит до IRSA -> cutout-и (паралельно)
  -> очищення (ZODI, FLAGS) -> вирівнювання на спільну небесну сітку (WCS)
  -> нормування -> PNG-кадри (звичайні та «лише зміни»).

Запит до TAP і формат cutout-URL взято з туторіалу IRSA
(https://caltech-ipac.github.io/irsa-tutorials/spherex-cutouts).
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import http.client
import io
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

# Важливо: спершу імпортуємо astropy, і лише потім чіпаємо його логер. Якщо створити
# logging.getLogger("astropy") до імпорту, astropy не зможе ініціалізувати власний логер.
from astropy import log as _astropy_log

_astropy_log.setLevel("ERROR")

IRSA_BASE = "https://irsa.ipac.caltech.edu/"
CACHE_DIR = os.environ.get("SPHEREX_CACHE", "cache")
CACHE_TTL_S = 168 * 3600          # метадані оновлюються щотижня; доба -- безпечно
MAX_FRAMES = int(os.environ.get("SPHEREX_MAX_FRAMES", "300"))
MAX_WORKERS = int(os.environ.get("SPHEREX_WORKERS", "10"))
MAX_GRID = 300                    # максимум пікселів на сторону вихідної сітки
OVERSAMPLE = 2                    # вихідна сітка вдвічі дрібніша за піксель SPHEREx
FLAG_MASK = 0xFFFFFFFF            # які біти FLAGS маскувати (див. Table 16 Explanatory Supplement)
FLAG_FALLBACK_FRAC = 0.6          # якщо «позначено» понад 60% пікселів -- прапорці ігноруємо

# Діапазони довжин хвиль детекторів (з документації IRSA)
DETECTOR_BANDS = {
    1: (0.75, 1.09), 2: (1.10, 1.62), 3: (1.63, 2.41),
    4: (2.42, 3.82), 5: (3.83, 4.41), 6: (4.42, 5.00),
}

ProgressCb = Callable[[str, int, int], None]


def _noop(stage: str, done: int, total: int) -> None:
    pass


# ---------------------------------------------------------------------------
# Ціль: розбір координат / назви
# ---------------------------------------------------------------------------
@dataclass
class Target:
    ra: float                 # градуси, J2000 (ICRS)
    dec: float
    pmra: float = 0.0         # мс дуги/рік (pm_ra * cos dec)
    pmdec: float = 0.0
    label: str = ""
    note: str = ""


def _parse_coords(text: str) -> Optional[tuple[float, float]]:
    """'210.8, 54.3' або '08h55m10.831s -07d14m42.53s' -> (ra, dec) у градусах."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    t = text.strip()
    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)[\s,;]+([+-]?\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        c = SkyCoord(t, frame="icrs")                       # 08h55m10.8s -07d14m42s
        return float(c.ra.deg), float(c.dec.deg)
    except Exception:
        pass
    try:
        c = SkyCoord(t, unit=(u.hourangle, u.deg), frame="icrs")  # 08 55 10.8 -07 14 42
        return float(c.ra.deg), float(c.dec.deg)
    except Exception:
        return None


def _to_float(v) -> float:
    try:
        if np.ma.is_masked(v):
            return 0.0
        f = float(v)
        return f if math.isfinite(f) else 0.0
    except Exception:
        return 0.0


def resolve_name(name: str) -> Target:
    """Назва -> позиція J2000 + власний рух із SIMBAD (якщо вдасться)."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    try:
        from astroquery.simbad import Simbad

        s = Simbad()
        s.add_votable_fields("pmra", "pmdec")
        tab = s.query_object(name)
        if tab is not None and len(tab) > 0:
            cols = {c.lower(): c for c in tab.colnames}
            ra_v, dec_v = tab[cols["ra"]][0], tab[cols["dec"]][0]
            if isinstance(ra_v, (str, bytes)):
                c = SkyCoord(str(ra_v), str(dec_v), unit=(u.hourangle, u.deg))
                ra, dec = float(c.ra.deg), float(c.dec.deg)
            else:
                ra, dec = float(ra_v), float(dec_v)
            pmra = _to_float(tab[cols["pmra"]][0]) if "pmra" in cols else 0.0
            pmdec = _to_float(tab[cols["pmdec"]][0]) if "pmdec" in cols else 0.0
            return Target(ra, dec, pmra, pmdec, label=name)
    except Exception as e:  # noqa: BLE001
        logging.warning("SIMBAD не спрацював для %r: %s", name, e)

    c = SkyCoord.from_name(name)   # запасний варіант: без власного руху
    return Target(float(c.ra.deg), float(c.dec.deg), 0.0, 0.0, label=name,
                  note="Власний рух не враховано (SIMBAD недоступний).")


def resolve_target(text: str) -> Target:
    text = (text or "").strip()
    if not text:
        raise ValueError("Вкажіть назву об'єкта або координати.")
    p = _parse_coords(text)
    if p is not None:
        ra, dec = p
        if not (0 <= ra < 360 and -90 <= dec <= 90):
            raise ValueError("RA має бути в [0, 360), Dec в [-90, 90] градусах.")
        return Target(ra, dec, label=f"{ra:.5f}, {dec:+.5f}")
    return resolve_name(text)


def propagate(t: Target, year: float) -> tuple[float, float]:
    """Лінійно переносить позицію J2000 на епоху `year` (власний рух у мс дуги/рік)."""
    dt = year - 2000.0
    dec = t.dec + t.pmdec * dt / 3.6e6
    ra = t.ra + t.pmra * dt / (3.6e6 * max(math.cos(math.radians(t.dec)), 1e-6))
    return ra % 360.0, max(-90.0, min(90.0, dec))


# ---------------------------------------------------------------------------
# Запит до IRSA TAP
# ---------------------------------------------------------------------------
def query_frames(ra: float, dec: float, detector: int) -> list[tuple[str, float]]:
    """
    Повертає [(base_uri, time), ...] відсортований за часом.
    base_uri -- шлях без параметрів cutout (їх додаємо пізніше, коли знаємо епоху).
    time -- time_bounds_lower із таблиці spherex.plane (у туторіалі підписаний як дні).
    """
    from astroquery.ipac.irsa import Irsa

    detector = int(detector)
    if detector not in DETECTOR_BANDS:
        raise ValueError("Детектор має бути від 1 до 6.")
    query = f"""
    SELECT a.uri AS uri, p.time_bounds_lower AS t
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {float(ra)!r}, {float(dec)!r}), p.poly)
      AND p.energy_bandpassname = 'SPHEREx-D{detector}'
    ORDER BY p.time_bounds_lower
    """
    res = Irsa.query_tap(query).to_table()
    rows = [(str(r["uri"]), float(r["t"])) for r in res]
    rows.sort(key=lambda r: r[1])

    # Дублікати (напр. l2b_retry) того ж спостереження: лишаємо один, віддаємо перевагу не-retry
    best: dict[int, tuple[str, float]] = {}
    for uri, t in rows:
        key = int(round(t * 1e6))
        if key not in best or ("retry" in best[key][0] and "retry" not in uri):
            best[key] = (uri, t)
    return sorted(best.values(), key=lambda r: r[1])


def cutout_url(base_uri: str, ra: float, dec: float, size_deg: float) -> str:
    return f"{IRSA_BASE}{base_uri}?center={ra},{dec}d&size={size_deg}"


# ---------------------------------------------------------------------------
# Завантаження та очищення одного cutout
# ---------------------------------------------------------------------------
@dataclass
class Frame:
    url: str
    time: float
    wavelength_um: float
    image: np.ndarray          # очищене: ZODI віднято, погані пікселі NaN, фон = 0
    wcs: object                # astropy WCS (просторовий)
    flagged_frac: float = 0.0


def clean_image(image, zodi, flags) -> tuple[Optional[np.ndarray], float]:
    """IMAGE - ZODI, маска за FLAGS, віднімання фону. Повертає (масив або None, частка позначених)."""
    img = np.asarray(image).astype(np.float32) - np.asarray(zodi).astype(np.float32)
    bad = ~np.isfinite(img)
    flagged = (np.asarray(flags).astype(np.int64) & FLAG_MASK) != 0
    frac = float(flagged.mean()) if flagged.size else 0.0
    if frac < FLAG_FALLBACK_FRAC:      # інакше прапорці, схоже, позначають «норму» -- ігноруємо
        bad |= flagged
    img[bad] = np.nan
    if not np.isfinite(img).any():
        return None, frac
    img -= np.float32(np.nanmedian(img))
    return img, frac


def load_cutout(url: str, coord, time_value: float, retries: int = 2) -> Optional[Frame]:
    """Завантажує cutout, читає HDU, рахує довжину хвилі в позиції цілі."""
    from astropy.io import fits
    from astropy.utils.data import conf
    from astropy.wcs import WCS

    conf.remote_timeout = 120      # читання SPHEREx може перевищувати стандартний ліміт
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with fits.open(url, cache=True) as hdul:                # HDU: 1 IMAGE, 2 FLAGS, 3 VARIANCE, 4 ZODI, 5 EPSF/PSF, 6 WCS-WAVE
                header = hdul[1].header
                img, frac = clean_image(hdul[1].data, hdul[4].data, hdul[2].data)
                if img is None:
                    return None
                wcs = WCS(header)
                wl = float("nan")
                try:
                    x, y = wcs.world_to_pixel(coord)
                    swcs = WCS(header, fobj=hdul, key="W")
                    swcs.sip = None
                    w, _bw = swcs.pixel_to_world(x, y)
                    wl = float(w.to("um").value)
                except Exception:      # noqa: BLE001  (WCS-WAVE не завжди читається)
                    pass
                return Frame(url, float(time_value), wl, img, wcs, frac)
        except (OSError, TimeoutError, http.client.HTTPException) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
        except (KeyError, IndexError, ValueError) as e:   # зіпсований/незвичний файл
            logging.warning("Пропущено %s: %s", url, e)
            return None
    logging.warning("Не вдалося завантажити %s: %s", url, last_err)
    return None


def fetch_frames(urls_times: list[tuple[str, float]], coord,
                 progress: ProgressCb = _noop) -> list[Frame]:
    """Паралельне завантаження; порядок за часом зберігається."""
    total = len(urls_times)
    done = 0
    results: list[Optional[Frame]] = [None] * total
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(load_cutout, u, coord, t): i for i, (u, t) in enumerate(urls_times)}
        for fut in cf.as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            progress("download", done, total)
    frames = [f for f in results if f is not None]
    frames.sort(key=lambda f: f.time)
    return frames


# ---------------------------------------------------------------------------
# Вирівнювання на спільну сітку
# ---------------------------------------------------------------------------
def make_grid(ra: float, dec: float, fov_deg: float, native_scale_arcsec: float):
    """Небесна сітка TAN навколо (ra, dec) з північчю вгору і сходом ліворуч."""
    from astropy.wcs import WCS

    scale = native_scale_arcsec / OVERSAMPLE                    # арксек/піксель
    n = int(round(fov_deg * 3600.0 / scale))
    n = max(16, min(n, MAX_GRID))
    scale = fov_deg * 3600.0 / n
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(n + 1) / 2.0, (n + 1) / 2.0]
    w.wcs.cdelt = [-scale / 3600.0, scale / 3600.0]
    w.wcs.cunit = ["deg", "deg"]
    return w, (n, n)


def native_pixel_scale_arcsec(wcs) -> float:
    from astropy.wcs.utils import proj_plane_pixel_scales

    s = proj_plane_pixel_scales(wcs)
    return float(np.mean(s) * 3600.0)


def align_frames(frames: list[Frame], grid_wcs, shape) -> tuple[np.ndarray, list[Frame]]:
    """Перепроєктовує кадри на сітку. Яскравість поверхнева (МДж/ср), тому лише інтерполяція."""
    from reproject import reproject_interp

    out, kept = [], []
    for f in frames:
        arr, _fp = reproject_interp((f.image, f.wcs), grid_wcs, shape_out=shape, order="bilinear")
        arr = arr.astype(np.float32)
        if np.isfinite(arr).mean() < 0.3:      # кадр майже не покриває сітку
            continue
        out.append(arr)
        kept.append(f)
    if not out:
        raise RuntimeError("Жоден кадр не покриває вибрану ділянку.")
    return np.stack(out), kept


# ---------------------------------------------------------------------------
# Нормування, різниця, рендер
# ---------------------------------------------------------------------------
def robust_sigma(a: np.ndarray) -> float:
    v = a[np.isfinite(a)]
    if v.size == 0:
        return 1.0
    s = 1.4826 * float(np.median(np.abs(v - np.median(v))))
    return s if s > 0 else (float(np.std(v)) or 1.0)


def normalize_stack(stack: np.ndarray) -> np.ndarray:
    """Кожен кадр ділимо на власний шум, щоб кадри різних довжин хвиль виглядали узгоджено."""
    out = np.empty_like(stack)
    for i in range(stack.shape[0]):
        out[i] = stack[i] / robust_sigma(stack[i])
    return out


def difference_stack(norm: np.ndarray) -> np.ndarray:
    """
    «Лише зміни»: від кожного кадру віднімаємо медіанний еталон нерухомого неба.
    Яскравість еталона підганяємо до кадру масштабом g (статичні джерела мають різну
    яскравість у різних довжинах хвиль). Результат у одиницях шуму.
    """
    if norm.shape[0] < 3:
        raise ValueError("Для режиму «лише зміни» потрібно щонайменше 3 кадри.")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # пікселі без покриття в усіх кадрах
        ref = np.nanmedian(norm, axis=0)
    bright = np.isfinite(ref) & (ref > 5.0)
    diff = np.empty_like(norm)
    for i in range(norm.shape[0]):
        f = norm[i]
        g = 1.0
        m = bright & np.isfinite(f)
        if m.sum() >= 3:
            # медіана відношень стійкіша до рухомого об'єкта, ніж сума (він не завищує масштаб)
            g = float(np.clip(np.median(f[m] / ref[m]), 0.2, 5.0))
        d = f - g * ref
        diff[i] = d / robust_sigma(d)
    return diff


def _asinh(x: np.ndarray, soft: float = 3.0) -> np.ndarray:
    return np.arcsinh(x / soft)


def to_gray_raw(norm: np.ndarray) -> np.ndarray:
    """Загальний для всього детектора розтяг -> uint8 (N, h, w)."""
    finite = norm[np.isfinite(norm)]
    lo = -3.0
    hi = float(np.clip(np.percentile(finite, 99.8), 8.0, 400.0)) if finite.size else 10.0
    t = (_asinh(norm) - _asinh(np.array(lo))) / (_asinh(np.array(hi)) - _asinh(np.array(lo)))
    t = np.nan_to_num(np.clip(t, 0, 1), nan=0.0)
    return (t * 255).astype(np.uint8)


def to_gray_diff(diff: np.ndarray, span: float = 8.0) -> np.ndarray:
    """Нуль змін = сірий; яскравіше = ставало яскравішим; темніше = слабшим."""
    t = 0.5 + 0.5 * np.clip(_asinh(diff) / _asinh(np.array(span)), -1, 1)
    t = np.nan_to_num(t, nan=0.5)
    return (t * 255).astype(np.uint8)


def gray_to_png(gray2d: np.ndarray, min_side: int = 512) -> bytes:
    from PIL import Image

    img = np.flipud(gray2d)            # північ угорі (масив має початок знизу)
    im = Image.fromarray(img, mode="L")
    k = max(1, math.ceil(min_side / max(im.size)))
    if k > 1:
        im = im.resize((im.size[0] * k, im.size[1] * k), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Результат конвеєра
# ---------------------------------------------------------------------------
def mjd_to_utc(t: float) -> str:
    try:
        if 40000 < t < 80000:
            from astropy.time import Time
            return Time(t, format="mjd").strftime("%Y-%m-%d %H:%M")
    except Exception:  # noqa: BLE001
        pass
    return ""


def mjd_to_jyear(t: float) -> Optional[float]:
    try:
        if 40000 < t < 80000:
            from astropy.time import Time
            return float(Time(t, format="mjd").jyear)
    except Exception:  # noqa: BLE001
        pass
    return None


class Stack:
    """Вирівняний набір кадрів одного детектора + ліниво рахований рендер."""

    def __init__(self, aligned: np.ndarray, times, wavelengths, urls, meta: dict):
        self.aligned = aligned.astype(np.float32)
        self.times = [float(t) for t in times]
        self.wavelengths = [float(w) for w in wavelengths]
        self.urls = list(urls)
        self.meta = dict(meta)
        self._raw = None
        self._diff = None
        self._png: dict[tuple[str, int], bytes] = {}

    def __len__(self) -> int:
        return self.aligned.shape[0]

    def _ensure(self, mode: str):
        if self._raw is None:
            norm = normalize_stack(self.aligned)
            self._raw = to_gray_raw(norm)
            self._norm = norm
        if mode == "diff" and self._diff is None:
            self._diff = to_gray_diff(difference_stack(self._norm))

    def png(self, i: int, mode: str = "raw") -> bytes:
        if mode not in ("raw", "diff"):
            raise ValueError("mode має бути raw або diff")
        if not 0 <= i < len(self):
            raise IndexError(i)
        key = (mode, i)
        if key not in self._png:
            self._ensure(mode)
            arr = self._raw if mode == "raw" else self._diff
            self._png[key] = gray_to_png(arr[i])
        return self._png[key]

    def describe(self) -> dict:
        frames = []
        for i, (t, w, u) in enumerate(zip(self.times, self.wavelengths, self.urls)):
            frames.append({
                "index": i,
                "time": t,
                "utc": mjd_to_utc(t),
                "wavelength_um": None if not math.isfinite(w) else round(w, 4),
                "source": u.split("/")[-1].split("?")[0],
            })
        return {"meta": self.meta, "frames": frames, "count": len(self)}


def build_stack(frames: list[Frame], ra: float, dec: float, fov_deg: float,
                meta: Optional[dict] = None) -> Stack:
    if not frames:
        raise RuntimeError("Немає кадрів для обробки.")
    scale = native_pixel_scale_arcsec(frames[0].wcs)
    grid, shape = make_grid(ra, dec, fov_deg, scale)
    aligned, kept = align_frames(frames, grid, shape)
    m = dict(meta or {})
    m.update({"grid_px": list(shape), "native_pixel_arcsec": round(scale, 3),
              "skipped_no_coverage": len(frames) - len(kept)})
    return Stack(aligned, [f.time for f in kept], [f.wavelength_um for f in kept],
                 [f.url for f in kept], m)


# ---------------------------------------------------------------------------
# Дисковий кеш (вирівняний стек)
# ---------------------------------------------------------------------------
def cache_key(target_text: str, fov: float, detector: int) -> str:
    raw = f"{target_text.strip().lower()}|{fov:.5f}|{int(detector)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def save_stack(path: str, s: Stack) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, aligned=s.aligned, times=np.array(s.times),
                        wavelengths=np.array(s.wavelengths), urls=np.array(s.urls),
                        meta=np.array(json.dumps(s.meta)))


def load_stack(path: str) -> Optional[Stack]:
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > CACHE_TTL_S:
        return None
    try:
        z = np.load(path, allow_pickle=False)
        return Stack(z["aligned"], z["times"], z["wavelengths"], z["urls"].tolist(),
                     json.loads(str(z["meta"])))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Повний конвеєр
# ---------------------------------------------------------------------------
def run_pipeline(target_text: str, fov_deg: float, detector: int,
                 progress: ProgressCb = _noop, use_cache: bool = True) -> Stack:
    if not 0.01 <= fov_deg <= 0.3:
        raise ValueError("Поле зору має бути від 0.01° до 0.3°.")
    detector = int(detector)
    if detector not in DETECTOR_BANDS:
        raise ValueError("Детектор має бути від 1 до 6.")

    path = os.path.join(CACHE_DIR, cache_key(target_text, fov_deg, detector) + ".npz")
    if use_cache:
        cached = load_stack(path)
        if cached is not None:
            progress("cache", 1, 1)
            return cached

    progress("resolve", 0, 1)
    tgt = resolve_target(target_text)

    # 1) Список знімків. Для пошуку беремо позицію на епоху ~2026.0
    #    (знімки SPHEREx великі, тож похибка в кілька арксекунд не впливає на покриття).
    ra0, dec0 = propagate(tgt, 2026.0)
    progress("query", 0, 1)
    rows = query_frames(ra0, dec0, detector)
    if not rows:
        raise RuntimeError("Для цієї ділянки і цього детектора знімків не знайдено.")
    notes = [tgt.note] if tgt.note else []
    if len(rows) > MAX_FRAMES:
        step = math.ceil(len(rows) / MAX_FRAMES)
        rows = rows[::step]
        notes.append(f"Знімків забагато, показано кожен {step}-й.")

    # 2) Епоха центру -- медіана часів знімків. Для швидких зір сітка центрується на цю епоху,
    #    тож рухомий об'єкт зсувається навколо центру, а не вилітає за межі кадру.
    years = [y for y in (mjd_to_jyear(t) for _, t in rows) if y is not None]
    epoch = float(np.median(years)) if years else 2026.0
    ra_c, dec_c = propagate(tgt, epoch)

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    coord = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg, frame="icrs")

    size = fov_deg * 1.5         # запас на обертання кадрів (>= sqrt(2) * fov)
    urls_times = [(cutout_url(b, ra_c, dec_c, size), t) for b, t in rows]
    progress("download", 0, len(urls_times))
    frames = fetch_frames(urls_times, coord, progress)
    if not frames:
        raise RuntimeError("Не вдалося завантажити жодного cutout-а.")
    failed = len(urls_times) - len(frames)
    if failed:
        notes.append(f"Не завантажилося {failed} з {len(urls_times)} знімків.")

    progress("align", 0, 1)
    stack = build_stack(frames, ra_c, dec_c, fov_deg, meta={
        "target": tgt.label, "ra": round(ra_c, 6), "dec": round(dec_c, 6),
        "epoch": round(epoch, 2), "fov_deg": fov_deg, "detector": detector,
        "band_um": list(DETECTOR_BANDS[detector]),
        "pm_mas_yr": [tgt.pmra, tgt.pmdec],
        "notes": notes,
    })
    try:
        save_stack(path, stack)
    except Exception as e:  # noqa: BLE001
        logging.warning("Кеш не збережено: %s", e)
    progress("done", 1, 1)
    return stack
