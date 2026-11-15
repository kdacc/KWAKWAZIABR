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
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from astropy import log as _astropy_log

_astropy_log.setLevel("ERROR")

IRSA_BASE = "https://irsa.ipac.caltech.edu/"
CACHE_DIR = os.environ.get("SPHEREX_CACHE", "cache")
CACHE_TTL_S = 168 * 3600         
MAX_FRAMES = int(os.environ.get("SPHEREX_MAX_FRAMES", "300"))
MAX_WORKERS = int(os.environ.get("SPHEREX_WORKERS", "10"))
MAX_GRID = 300                   
OVERSAMPLE = 2                    
FLAG_MASK = 0xFFFFFFFF            
FLAG_FALLBACK_FRAC = 0.6    
MAX_FOV_DEG = 0.3          
CUTOUT_SIZE = MAX_FOV_DEG * 1.5      


DETECTOR_BANDS = {
    1: (0.75, 1.09), 2: (1.10, 1.62), 3: (1.63, 2.41),
    4: (2.42, 3.82), 5: (3.83, 4.41), 6: (4.42, 5.00),
}

class Cancelled(Exception):
    pass


ProgressCb = Callable[[str, int, int], None]


def _noop(stage: str, done: int, total: int) -> None:
    pass


@dataclass
class Target:
    ra: float
    dec: float
    label: str = ""


def _parse_coords(text: str) -> Optional[tuple[float, float]]:
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    t = text.strip()
    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)[\s,;]+([+-]?\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        c = SkyCoord(t, frame="icrs")                       
        return float(c.ra.deg), float(c.dec.deg)
    except Exception:
        pass
    try:
        c = SkyCoord(t, unit=(u.hourangle, u.deg), frame="icrs")  
        return float(c.ra.deg), float(c.dec.deg)
    except Exception:
        return None


def resolve_target(text: str) -> Target:
    text = (text or "").strip()
    if not text:
        raise ValueError("Please specify RA, Dec coordinates.")
    p = _parse_coords(text)
    if p is None:
        raise ValueError(
            "Coordinates RA, Dec are required. Examples: '133.795 -7.245' (degrees) or '08h55m10.8s -07d14m42s'. Object names are not supported."
            )
    ra, dec = p
    if not (0 <= ra < 360 and -90 <= dec <= 90):
        raise ValueError("RA must be in [0, 360), Dec in [-90, 90] degrees.")
    return Target(ra, dec, label=f"{ra:.5f}, {dec:+.5f}")


def normalize_target(text: str) -> str:
    t = resolve_target(text)
    ra = round(t.ra, 5) % 360.0
    return f"{ra:.5f}, {round(t.dec, 5):.5f}"


def query_frames(ra: float, dec: float, detector: int) -> list[tuple[str, float]]:
    from astroquery.ipac.irsa import Irsa

    detector = int(detector)
    if detector not in DETECTOR_BANDS:
        raise ValueError("Detector must be from 1 to 6.")
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

    best: dict[int, tuple[str, float]] = {}
    for uri, t in rows:
        key = int(round(t * 1e6))
        if key not in best or ("retry" in best[key][0] and "retry" not in uri):
            best[key] = (uri, t)
    return sorted(best.values(), key=lambda r: r[1])


def cutout_url(base_uri: str, ra: float, dec: float, size_deg: float) -> str:
    return f"{IRSA_BASE}{base_uri}?center={ra},{dec}d&size={size_deg}"


@dataclass
class Frame:
    url: str
    time: float
    wavelength_um: float
    image: np.ndarray          
    wcs: object                
    flagged_frac: float = 0.0


def clean_image(image, zodi, flags) -> tuple[Optional[np.ndarray], float]:
    img = np.asarray(image).astype(np.float32) - np.asarray(zodi).astype(np.float32)
    bad = ~np.isfinite(img)
    flagged = (np.asarray(flags).astype(np.int64) & FLAG_MASK) != 0
    frac = float(flagged.mean()) if flagged.size else 0.0
    if frac < FLAG_FALLBACK_FRAC:     
        bad |= flagged
    img[bad] = np.nan
    if not np.isfinite(img).any():
        return None, frac
    img -= np.float32(np.nanmedian(img))
    return img, frac


def load_cutout(url: str, coord, time_value: float, retries: int = 2) -> Optional[Frame]:
    from astropy.io import fits
    from astropy.utils.data import conf
    from astropy.wcs import WCS

    conf.remote_timeout = 120      
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with fits.open(url, cache=True) as hdul:                
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
                except Exception:      
                    pass
                return Frame(url, float(time_value), wl, img, wcs, frac)
        except (OSError, TimeoutError, http.client.HTTPException) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
        except (KeyError, IndexError, ValueError) as e:   
            logging.warning("Пропущено %s: %s", url, e)
            return None
    logging.warning("Не вдалося завантажити %s: %s", url, last_err)
    return None


def fetch_frames(urls_times: list[tuple[str, float]], coord,
                 progress: ProgressCb = _noop) -> list[Frame]:
    total = len(urls_times)
    done = 0
    results: list[Optional[Frame]] = [None] * total
    stop = threading.Event()

    def task(u, t):
        if stop.is_set():
            return None
        return load_cutout(u, coord, t)

    ex = cf.ThreadPoolExecutor(max_workers=MAX_WORKERS)
    try:
        futs = {ex.submit(task, u, t): i for i, (u, t) in enumerate(urls_times)}
        for fut in cf.as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            progress("download", done, total)
    except BaseException:
        stop.set()
        raise
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    frames = [f for f in results if f is not None]
    frames.sort(key=lambda f: f.time)
    return frames


def make_grid(ra: float, dec: float, fov_deg: float, native_scale_arcsec: float):
    from astropy.wcs import WCS

    scale = native_scale_arcsec / OVERSAMPLE                    
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
    from reproject import reproject_interp

    out, kept = [], []
    for f in frames:
        arr, _fp = reproject_interp((f.image, f.wcs), grid_wcs, shape_out=shape, order="bilinear")
        arr = arr.astype(np.float32)
        if np.isfinite(arr).mean() < 0.3:      
            continue
        out.append(arr)
        kept.append(f)
    if not out:
        raise RuntimeError("No frame covers the selected area.")
    return np.stack(out), kept


def robust_sigma(a: np.ndarray) -> float:
    v = a[np.isfinite(a)]
    if v.size == 0:
        return 1.0
    s = 1.4826 * float(np.median(np.abs(v - np.median(v))))
    return s if s > 0 else (float(np.std(v)) or 1.0)


def normalize_stack(stack: np.ndarray) -> np.ndarray:
    out = np.empty_like(stack)
    for i in range(stack.shape[0]):
        out[i] = stack[i] / robust_sigma(stack[i])
    return out


def _asinh(x: np.ndarray, soft: float = 3.0) -> np.ndarray:
    return np.arcsinh(x / soft)


def to_gray_raw(norm: np.ndarray) -> np.ndarray:
    finite = norm[np.isfinite(norm)]
    lo = -3.0
    hi = float(np.clip(np.percentile(finite, 99.8), 8.0, 400.0)) if finite.size else 10.0
    t = (_asinh(norm) - _asinh(np.array(lo))) / (_asinh(np.array(hi)) - _asinh(np.array(lo)))
    t = np.nan_to_num(np.clip(t, 0, 1), nan=0.0)
    return (t * 255).astype(np.uint8)


def to_gray_diff(diff: np.ndarray, span: float = 8.0) -> np.ndarray:
    t = 0.5 + 0.5 * np.clip(_asinh(diff) / _asinh(np.array(span)), -1, 1)
    t = np.nan_to_num(t, nan=0.5)
    return (t * 255).astype(np.uint8)


def gray_to_png(gray2d: np.ndarray, min_side: int = 512) -> bytes:
    from PIL import Image

    img = np.flipud(gray2d)
    im = Image.fromarray(img, mode="L")
    k = max(1, math.ceil(min_side / max(im.size)))
    if k > 1:
        im = im.resize((im.size[0] * k, im.size[1] * k), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def mjd_to_utc(t: float) -> str:
    try:
        if 40000 < t < 80000:
            from astropy.time import Time
            return Time(t, format="mjd").strftime("%Y-%m-%d %H:%M")
    except Exception:  
        pass
    return ""


def mjd_to_jyear(t: float) -> Optional[float]:
    try:
        if 40000 < t < 80000:
            from astropy.time import Time
            return float(Time(t, format="mjd").jyear)
    except Exception:  
        pass
    return None


class Stack:

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

    def png(self, i: int, mode: str = "raw") -> bytes:
        if mode not in ("raw", "diff"):
            raise ValueError("mode must be 'raw' or 'diff'")
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
        raise RuntimeError("No frames available for processing.")
    scale = native_pixel_scale_arcsec(frames[0].wcs)
    grid, shape = make_grid(ra, dec, fov_deg, scale)
    aligned, kept = align_frames(frames, grid, shape)
    m = dict(meta or {})
    m.update({"grid_px": list(shape), "native_pixel_arcsec": round(scale, 3),
              "skipped_no_coverage": len(frames) - len(kept)})
    return Stack(aligned, [f.time for f in kept], [f.wavelength_um for f in kept],
                 [f.url for f in kept], m)


RAW_CACHE_DIR = os.path.join(CACHE_DIR, "raw")

def raw_cache_key(target_text: str, detector: int) -> str:
    raw = f"{target_text.strip().lower()}|raw|{int(detector)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def save_raw_frames(path: str, frames: list[Frame], meta: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "times": np.array([f.time for f in frames]),
        "wavelengths": np.array([f.wavelength_um for f in frames]),
        "urls": np.array([f.url for f in frames]),
        "flagged_frac": np.array([f.flagged_frac for f in frames]),
        "meta": np.array(json.dumps(meta)),
    }

    images = [f.image for f in frames]
    wcs_headers = [dict(f.wcs.to_header(relax=True)) for f in frames]
    np.savez_compressed(path, **payload, images=np.array(images, dtype=object),
                        wcs_headers=np.array(wcs_headers, dtype=object))


def cache_key(target_text: str, fov: float, detector: int) -> str:
    raw = f"{target_text.strip().lower()}|{fov:.5f}|{int(detector)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def save_stack(path: str, s: Stack) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        aligned=s.aligned,
        times=np.array(s.times),
        wavelengths=np.array(s.wavelengths),
        urls=np.array(s.urls),
        meta=np.array(json.dumps(s.meta)),
    )


def load_stack(path: str) -> Optional[Stack]:
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > CACHE_TTL_S:
        return None
    try:
        z = np.load(path, allow_pickle=False)
        return Stack(
            z["aligned"], z["times"], z["wavelengths"],
            z["urls"].tolist(), json.loads(str(z["meta"])),
        )
    except Exception:
        return None


def load_raw_frames(path: str) -> Optional[tuple[list[Frame], dict]]:
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > CACHE_TTL_S:
        return None
    try:
        from astropy.wcs import WCS
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        frames = []
        for i in range(len(z["times"])):
            wcs = WCS(dict(z["wcs_headers"][i]), relax=True)
            frames.append(Frame(
                url=str(z["urls"][i]),
                time=float(z["times"][i]),
                wavelength_um=float(z["wavelengths"][i]),
                image=np.asarray(z["images"][i], dtype=np.float32),
                wcs=wcs,
                flagged_frac=float(z["flagged_frac"][i]),
            ))
        return frames, meta
    except Exception:
        return None


def run_pipeline(target_text: str, fov_deg: float, detector: int,
                 progress: ProgressCb = _noop, use_cache: bool = True) -> Stack:
    if not 0.01 <= fov_deg <= 0.3:
        raise ValueError("Field of view must be between 0.01° and 0.3°.")
    detector = int(detector)
    if detector not in DETECTOR_BANDS:
        raise ValueError("Detector must be from 1 to 6.")

    aligned_path = os.path.join(CACHE_DIR, cache_key(target_text, fov_deg, detector) + ".npz")
    if use_cache:
        cached = load_stack(aligned_path)
        if cached is not None:
            progress("cache", 1, 1)
            return cached

    raw_path = os.path.join(RAW_CACHE_DIR, raw_cache_key(target_text, detector) + ".npz")
    frames: Optional[list[Frame]] = None
    notes: list[str] = []
    ra_c = dec_c = epoch = None
    tgt = None

    if use_cache:
        loaded = load_raw_frames(raw_path)
        if loaded is not None:
            frames, raw_meta = loaded
            notes = list(raw_meta.get("notes", []))
            ra_c = raw_meta["ra"]
            dec_c = raw_meta["dec"]
            epoch = raw_meta["epoch"]
            tgt = Target(ra_c, dec_c, label=raw_meta.get("target", target_text))
            progress("cache_raw", 1, 1)

    if frames is None:
        progress("resolve", 0, 1)
        tgt = resolve_target(target_text)

        ra_c, dec_c = tgt.ra, tgt.dec
        progress("query", 0, 1)
        rows = query_frames(ra_c, dec_c, detector)
        if not rows:
            raise RuntimeError("No frames found for the selected area and detector.")
        notes = []
        if len(rows) > MAX_FRAMES:
            step = math.ceil(len(rows) / MAX_FRAMES)
            rows = rows[::step]
            notes.append(f"Too many frames, showing every {step}-th one.")

        years = [y for y in (mjd_to_jyear(t) for _, t in rows) if y is not None]
        epoch = float(np.median(years)) if years else 2026.0

        from astropy import units as u
        from astropy.coordinates import SkyCoord
        coord = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg, frame="icrs")

        urls_times = [(cutout_url(b, ra_c, dec_c, CUTOUT_SIZE), t) for b, t in rows]
        progress("download", 0, len(urls_times))
        frames = fetch_frames(urls_times, coord, progress)
        if not frames:
            raise RuntimeError("Failed to download any cutouts.")
        failed = len(urls_times) - len(frames)
        if failed:
            notes.append(f"Failed to download {failed} out of {len(urls_times)} frames.")

        raw_meta = {
            "target": tgt.label, "ra": round(ra_c, 6), "dec": round(dec_c, 6),
            "epoch": round(epoch, 2), "detector": detector,
            "notes": notes,
        }
        try:
            save_raw_frames(raw_path, frames, raw_meta)
        except Exception as e: 
            logging.warning("Failed to save raw frames cache: %s", e)

    progress("align", 0, 1)
    stack = build_stack(frames, ra_c, dec_c, fov_deg, meta={
        "target": tgt.label, "ra": round(ra_c, 6), "dec": round(dec_c, 6),
        "epoch": round(epoch, 2), "fov_deg": fov_deg, "detector": detector,
        "band_um": list(DETECTOR_BANDS[detector]),
        "notes": notes,
    })
    try:
        save_stack(aligned_path, stack)
    except Exception as e: 
        logging.warning("Failed to save aligned stack cache: %s", e)
    progress("done", 1, 1)
    return stack
