"""The local ingestor: raw Copernicus products to rasters on the site grid.

Between `gnn-sar search` (which downloads archives) and `gnn-sar ingest` (which
wants one aligned GeoTIFF per acquisition) sits the step that decides whether a
time series measures the ground or the processing. This module runs it locally:

- **Sentinel-1** goes through ESA SNAP. The graph is data, not code
  (`docs/esempi/snap_s1_grd_preprocessing.xml`), so the chain can be changed
  without touching Python; this module finds `gpt`, feeds it one product at a
  time, names the output after the sensing date, skips what is already done and
  reports what failed instead of stopping at the first error.
- **Sentinel-2** needs no SNAP: L2A is already surface reflectance, so the work
  is finding the band files inside the ``.SAFE``, putting them on the study grid
  — bilinear for reflectance, nearest for the scene classification, which is a
  classification and must not be interpolated — and stacking them into one file.
- **Coherence** pairs SLC acquisitions of the same relative orbit at a fixed
  temporal baseline and runs the interferometric graph over each pair.

Everything lands on the **exact grid of a `StudySite`**. Alignment goes through
`rasterio.warp.reproject` rather than a `gdalwarp` subprocess: one dependency
fewer, and the destination grid is the site's by construction instead of by a
correctly typed command line.

What this module does *not* do is hide SNAP. If `gpt` is missing, it says so and
points at the download; if a scene fails, the tail of SNAP's own error is in the
report. A preprocessing chain you cannot inspect is one you cannot trust.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from gnn_sar.data.raster import write_raster
from gnn_sar.data.sentinel import sensing_date
from gnn_sar.data.sites import StudySite

__all__ = [
    "SnapNotFound",
    "PreprocessReport",
    "find_gpt",
    "gpt_command",
    "run_gpt",
    "align_to_site",
    "preprocess_sentinel1",
    "find_sentinel2_bands",
    "stack_sentinel2_product",
    "preprocess_sentinel2",
    "coherence_pairs",
    "preprocess_coherence",
    "declared_parameters",
]

S2_DEFAULT_BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B08", "B11", "B12")
"""Blue, green, red, NIR and the two SWIR bands: enough for NDVI, NDBI and BSI."""

_S2_EXTENSIONS = (".jp2", ".tif", ".tiff")
_GPT_ENV_VARS = ("GNN_SAR_GPT", "GPT_PATH")
_GPT_CANDIDATES = (
    "/opt/snap/bin/gpt",
    "/usr/local/snap/bin/gpt",
    "/Applications/esa-snap/bin/gpt",
    "~/esa-snap/bin/gpt",
    "~/snap/bin/gpt",
)


class SnapNotFound(RuntimeError):
    """ESA SNAP's ``gpt`` could not be located."""


@dataclass
class PreprocessReport:
    """What a batch did, product by product.

    One failing scene must not stop a run of three hundred: a swath truncated
    at the edge of the area, a missing precise orbit or a corrupt download are
    all normal, and the answer is to finish the batch and print what needs
    attention.
    """

    written: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def describe(self) -> str:
        lines = [
            f"written  {len(self.written)}",
            f"skipped  {len(self.skipped)} (already present)",
            f"failed   {len(self.failed)}",
        ]
        for path, reason in self.failed:
            first = reason.strip().splitlines()
            lines.append(f"  {path.name}: {first[-1] if first else 'unknown error'}")
        return "\n".join(lines)


def find_gpt(explicit: str | Path | None = None) -> Path:
    """Locate SNAP's ``gpt``: an explicit path, the environment, then PATH.

    Args:
        explicit: a path given on the command line, which always wins.

    Raises:
        SnapNotFound: with the places that were tried, so the fix is obvious.
    """
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate
        raise SnapNotFound(f"{candidate} is not an executable file")

    for variable in _GPT_ENV_VARS:
        value = os.environ.get(variable)
        if value and Path(value).expanduser().is_file():
            return Path(value).expanduser()

    found = shutil.which("gpt")
    if found:
        return Path(found)

    for candidate in _GPT_CANDIDATES:
        path = Path(candidate).expanduser()
        if path.is_file():
            return path

    raise SnapNotFound(
        "ESA SNAP's `gpt` was not found. Install SNAP (10 or later) from "
        "https://step.esa.int/main/download/snap-download/ and either put its "
        "bin/ directory on PATH, set GNN_SAR_GPT to the gpt executable, or pass "
        "--gpt. Tried: $GNN_SAR_GPT, $GPT_PATH, PATH, "
        f"{', '.join(_GPT_CANDIDATES)}"
    )


def gpt_command(
    graph: str | Path,
    parameters: dict[str, str] | None = None,
    gpt: str | Path = "gpt",
    memory: str = "8G",
    threads: int = 4,
    cache: str | None = None,
) -> list[str]:
    """The argv `gpt` is invoked with. Pure, so it can be asserted on.

    Args:
        graph: the SNAP graph XML.
        parameters: ``${name}`` substitutions inside the graph, passed as
            ``-Pname=value``.
        gpt: the executable.
        memory: JVM heap, SNAP's ``-c`` cache is derived from it.
        threads: SNAP's ``-q``.
        cache: explicit tile cache size; defaults to ``memory``.
    """
    if threads < 1:
        raise ValueError("threads must be >= 1")
    command = [str(gpt), str(graph), "-c", cache or memory, "-q", str(threads)]
    for name, value in sorted((parameters or {}).items()):
        command.append(f"-P{name}={value}")
    return command


def run_gpt(
    graph: str | Path,
    parameters: dict[str, str] | None = None,
    gpt: str | Path | None = None,
    memory: str = "8G",
    threads: int = 4,
    timeout: float | None = None,
) -> str:
    """Run one SNAP graph and return its output.

    Raises:
        RuntimeError: with SNAP's own message when the graph fails. The JVM
            writes the useful part to stdout, so both streams are kept.
    """
    executable = find_gpt(gpt)
    command = gpt_command(graph, parameters, executable, memory, threads)
    process = subprocess.run(  # noqa: S603 - the argv is built above, not a shell
        command, capture_output=True, text=True, timeout=timeout
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"SNAP failed ({process.returncode}) on {Path(str(graph)).name}:\n"
            f"{(process.stdout + process.stderr).strip()[-2000:]}"
        )
    return process.stdout


def align_to_site(
    source: str | Path,
    destination: str | Path,
    site: StudySite,
    bands: Sequence[int] | None = None,
    resampling: str = "bilinear",
    band_resampling: dict[int, str] | None = None,
    dtype: str = "float32",
    nodata: float = float("nan"),
) -> Path:
    """Reproject and crop a raster onto a site's exact grid.

    This is the `gdalwarp -t_srs -tr -te` of the guide, done in process: the
    destination grid is :meth:`~gnn_sar.data.sites.StudySite.grid`, so the
    result matches by construction and `gnn-sar ingest` cannot later reject it.

    Args:
        source: any raster GDAL can open.
        destination: the GeoTIFF to write.
        site: the study area.
        bands: 1-based band indices to keep, in order.
        resampling: default method for every band — ``bilinear`` for continuous
            quantities, ``nearest`` for classifications.
        band_resampling: per-band override, keyed by **position in the output**
            (0-based), e.g. ``{6: "nearest"}`` for a scene-classification band.
        dtype: the written type.
        nodata: the written nodata value.
    """
    rasterio = _require_rasterio()
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    grid = site.grid()
    with rasterio.open(source) as handle:
        indexes = list(range(1, handle.count + 1)) if bands is None else list(bands)
        out = np.full((len(indexes), grid.height, grid.width), np.nan, dtype=np.float64)
        for position, index in enumerate(indexes):
            method = (band_resampling or {}).get(position, resampling)
            reproject(
                source=rasterio.band(handle, index),
                destination=out[position],
                src_nodata=handle.nodata,
                dst_transform=Affine.from_gdal(*grid.transform),
                dst_crs=grid.crs,
                dst_nodata=np.nan,
                resampling=_resampling(Resampling, method),
            )
    return write_raster(destination, out, grid, nodata=nodata, dtype=dtype)


def preprocess_sentinel1(
    products: Iterable[str | Path],
    output_dir: str | Path,
    site: StudySite,
    graph: str | Path,
    gpt: str | Path | None = None,
    memory: str = "8G",
    threads: int = 4,
    jobs: int = 1,
    overwrite: bool = False,
    keep_intermediate: bool = False,
    parameters: dict[str, str] | None = None,
    log: Any = print,
) -> PreprocessReport:
    """SNAP-process a batch of Sentinel-1 products onto the site grid.

    For each product: run the graph, then align the result to the site. The
    output is named after the sensing date, which is what
    :func:`gnn_sar.data.sentinel.sensing_date` reads back during ingestion.

    Args:
        products: the downloaded ``.zip`` or ``.SAFE`` products.
        output_dir: where the aligned GeoTIFFs go.
        site: the study area.
        graph: the SNAP graph, taking ``${input}`` and ``${output}``.
        gpt: an explicit path to SNAP's ``gpt``.
        memory, threads: passed to SNAP.
        jobs: how many products to process at once. SNAP already uses several
            threads per product, so more than two is usually counter-productive
            and multiplies the memory.
        overwrite: redo products whose output already exists.
        keep_intermediate: keep SNAP's output next to the aligned file, which is
            what you want the first time you check a chain.
        parameters: extra ``${name}`` values for the graph. ``crs`` defaults to
            the site's, and only the variables the graph actually declares are
            passed, so a custom graph is never handed a value it cannot use.
        log: where progress goes.

    Returns:
        A :class:`PreprocessReport`.
    """
    products = [Path(product) for product in products]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = output_dir / "_snap"
    report = PreprocessReport()
    if not products:
        return report

    find_gpt(gpt)  # fail once, before the batch, rather than per product
    extra = declared_parameters(graph, {"crs": site.crs, **(parameters or {})})

    def one(product: Path) -> None:
        destination = output_dir / f"{_output_stem(product)}.tif"
        if destination.exists() and not overwrite:
            report.skipped.append(destination)
            return
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        intermediate = intermediate_dir / f"{_output_stem(product)}_snap.tif"
        try:
            log(f"snap   {product.name}")
            run_gpt(
                graph,
                {"input": str(product), "output": str(intermediate), **extra},
                gpt=gpt,
                memory=memory,
                threads=threads,
            )
            log(f"align  {destination.name}")
            align_to_site(intermediate, destination, site)
            report.written.append(destination)
        except Exception as error:  # a failing scene must not stop the batch
            report.failed.append((product, str(error)))
        finally:
            if not keep_intermediate and intermediate.exists():
                intermediate.unlink()

    _run_batch(one, products, jobs)
    if not keep_intermediate and intermediate_dir.is_dir():
        shutil.rmtree(intermediate_dir, ignore_errors=True)
    return report


def find_sentinel2_bands(
    product: str | Path,
    bands: Sequence[str] = S2_DEFAULT_BANDS,
    scl: bool = True,
) -> dict[str, Path]:
    """Locate the band files inside an unzipped Sentinel-2 ``.SAFE`` product.

    A L2A product carries the same band at several resolutions; the finest is
    taken, and the caller resamples. Works on the ``SAFE`` layout and on any
    flat directory whose file names carry the band code.

    Args:
        product: the ``.SAFE`` directory (or any directory holding the bands).
        bands: the band codes wanted, e.g. ``("B02", "B08", "B11")``.
        scl: also locate the scene-classification layer.

    Returns:
        ``{band code: path}``, with ``"SCL"`` included when found and asked for.

    Raises:
        FileNotFoundError: naming the bands that are missing — a band left out
            silently would become a missing index later, with no trace.
    """
    root = Path(product)
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} is not a directory; unzip the .SAFE product first "
            "(gnn-sar does not read band data out of the archive)"
        )

    wanted = [*bands, *(["SCL"] if scl else [])]
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _S2_EXTENSIONS
    ]
    found: dict[str, Path] = {}
    for code in wanted:
        matches = [
            path
            for path in files
            if f"_{code}_" in path.name or path.stem.endswith(f"_{code}")
        ]
        if matches:
            found[code] = min(matches, key=_resolution_rank)

    missing = [code for code in bands if code not in found]
    if missing:
        raise FileNotFoundError(
            f"{root.name}: no file found for {missing}. Expected names like "
            "T33TTG_20240117T100319_B02_10m.jp2 under GRANULE/*/IMG_DATA/."
        )
    return found


def stack_sentinel2_product(
    product: str | Path,
    destination: str | Path,
    site: StudySite,
    bands: Sequence[str] = S2_DEFAULT_BANDS,
    scl: bool = True,
    dtype: str = "float32",
) -> Path:
    """Put one Sentinel-2 L2A product on the site grid as a single stack.

    Band order in the output is ``bands`` followed by ``SCL``, which is the
    order an ingestion config declares (``"bands"`` plus ``"scl_band"``).
    Reflectance is resampled bilinearly, the scene classification with nearest
    neighbour — interpolating class codes would invent classes that do not
    exist, and the mask is what decides which pixels are believed.
    """
    rasterio = _require_rasterio()
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    located = find_sentinel2_bands(product, bands, scl)
    order = [*bands, *(["SCL"] if scl and "SCL" in located else [])]
    grid = site.grid()

    out = np.full((len(order), grid.height, grid.width), np.nan, dtype=np.float64)
    for position, code in enumerate(order):
        method = Resampling.nearest if code == "SCL" else Resampling.bilinear
        with rasterio.open(located[code]) as handle:
            reproject(
                source=rasterio.band(handle, 1),
                destination=out[position],
                src_nodata=handle.nodata,
                dst_transform=Affine.from_gdal(*grid.transform),
                dst_crs=grid.crs,
                dst_nodata=np.nan,
                resampling=method,
            )
    return write_raster(destination, out, grid, nodata=float("nan"), dtype=dtype)


def preprocess_sentinel2(
    products: Iterable[str | Path],
    output_dir: str | Path,
    site: StudySite,
    bands: Sequence[str] = S2_DEFAULT_BANDS,
    scl: bool = True,
    jobs: int = 1,
    overwrite: bool = False,
    log: Any = print,
) -> PreprocessReport:
    """Stack and align a batch of Sentinel-2 L2A products. No SNAP involved."""
    products = [Path(product) for product in products]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = PreprocessReport()

    def one(product: Path) -> None:
        destination = output_dir / f"{_output_stem(product)}.tif"
        if destination.exists() and not overwrite:
            report.skipped.append(destination)
            return
        try:
            log(f"stack  {product.name}")
            stack_sentinel2_product(product, destination, site, bands, scl)
            report.written.append(destination)
        except Exception as error:
            report.failed.append((product, str(error)))

    _run_batch(one, products, jobs)
    return report


def coherence_pairs(
    dates: Sequence[datetime],
    baseline_days: int = 12,
    tolerance_days: int = 2,
    max_per_date: int = 1,
) -> list[tuple[int, int]]:
    """Which acquisitions to interfere with which, as index pairs.

    Coherence is a property of a *pair*, so a series of ``K`` acquisitions gives
    at most ``K-1`` coherence values, and the temporal baseline has to be held
    constant: coherence falls with time on its own, so mixing 6-day and 12-day
    pairs would put a decorrelation signal in the series that has nothing to do
    with construction.

    Args:
        dates: sensing dates, in any order.
        baseline_days: the temporal baseline to aim for — 6 or 12 days for
            Sentinel-1, depending on the constellation configuration of the
            period.
        tolerance_days: how far from that baseline a pair may fall. Keep it
            small; widening it is how an inconsistent series is built.
        max_per_date: how many partners a date may take, closest first.

    Returns:
        ``(earlier, later)`` index pairs into ``dates``, chronological.
    """
    if baseline_days < 1:
        raise ValueError("baseline_days must be >= 1")
    if tolerance_days < 0:
        raise ValueError("tolerance_days must be >= 0")
    if max_per_date < 1:
        raise ValueError("max_per_date must be >= 1")

    order = sorted(range(len(dates)), key=lambda i: dates[i])
    target = timedelta(days=baseline_days)
    tolerance = timedelta(days=tolerance_days)

    pairs: list[tuple[int, int]] = []
    for position, first in enumerate(order):
        candidates = []
        for second in order[position + 1 :]:
            gap = dates[second] - dates[first]
            if gap > target + tolerance:
                break
            if gap >= target - tolerance:
                candidates.append((abs(gap - target), second))
        for _, second in sorted(candidates)[:max_per_date]:
            pairs.append((first, second))
    return pairs


def preprocess_coherence(
    products: Sequence[str | Path],
    output_dir: str | Path,
    site: StudySite,
    graph: str | Path,
    baseline_days: int = 12,
    tolerance_days: int = 2,
    gpt: str | Path | None = None,
    memory: str = "12G",
    threads: int = 4,
    jobs: int = 1,
    overwrite: bool = False,
    keep_intermediate: bool = False,
    parameters: dict[str, str] | None = None,
    log: Any = print,
) -> PreprocessReport:
    """InSAR coherence for every SLC pair at a fixed temporal baseline.

    The output of a pair is dated with its **second** acquisition: coherence
    describes the interval that ends there, and a product available at time `t`
    may not be dated with anything later than `t` — the same causality the model
    enforces.

    Feed the result to the ingestion either as an extra band of the Sentinel-1
    file or as a source of kind ``raw``; see ``docs/roadmap.md`` for what is and
    is not settled about this path.

    Args:
        products: the SLC products, **all of the same relative orbit**. Pairs
            from different orbits are not interferable, and this function cannot
            tell them apart — restrict the download instead
            (``gnn-sar search --orbit``).
        graph: the SNAP graph, taking ``${input1}``, ``${input2}``, ``${output}``.
        parameters: extra graph variables. The shipped graph also declares
            ``subswath``, ``polarisation`` and ``crs``; the last defaults to the
            site's CRS, the first two have no sensible default because they
            depend on where the area falls in the swath.

    Returns:
        A :class:`PreprocessReport`.
    """
    paths = [Path(product) for product in products]
    dates = [sensing_date(path) for path in paths]
    pairs = coherence_pairs(dates, baseline_days, tolerance_days)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = output_dir / "_snap"
    report = PreprocessReport()
    if not pairs:
        log(
            f"no pair at {baseline_days} ± {tolerance_days} days among "
            f"{len(paths)} products: check the relative orbit and the baseline"
        )
        return report

    find_gpt(gpt)
    extra = declared_parameters(graph, {"crs": site.crs, **(parameters or {})})
    missing = _undeclared(graph, {"input1", "input2", "output", *extra})
    if missing:
        raise ValueError(
            f"{Path(str(graph)).name} needs the graph variables {sorted(missing)} "
            "which were not given; pass them with --parameter name=value "
            "(subswath and polarisation have no default)"
        )
    log(f"pairs  {len(pairs)} at {baseline_days} ± {tolerance_days} days")

    def one(pair: tuple[int, int]) -> None:
        first, second = pair
        stem = f"COH_{dates[second]:%Y%m%dT%H%M%S}_{baseline_days}d"
        destination = output_dir / f"{stem}.tif"
        if destination.exists() and not overwrite:
            report.skipped.append(destination)
            return
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        intermediate = intermediate_dir / f"{stem}_snap.tif"
        try:
            log(f"snap   {paths[first].name} x {paths[second].name}")
            run_gpt(
                graph,
                {
                    "input1": str(paths[first]),
                    "input2": str(paths[second]),
                    "output": str(intermediate),
                    **extra,
                },
                gpt=gpt,
                memory=memory,
                threads=threads,
            )
            align_to_site(intermediate, destination, site)
            report.written.append(destination)
        except Exception as error:
            report.failed.append((paths[second], str(error)))
        finally:
            if not keep_intermediate and intermediate.exists():
                intermediate.unlink()

    _run_batch(one, pairs, jobs)
    if not keep_intermediate and intermediate_dir.is_dir():
        shutil.rmtree(intermediate_dir, ignore_errors=True)
    return report


def declared_parameters(
    graph: str | Path, candidates: dict[str, str]
) -> dict[str, str]:
    """Keep only the values the graph actually declares as ``${name}``.

    A custom graph should not have to accept every variable this package knows
    about, and handing SNAP a value with nowhere to go is at best noise.
    """
    text = Path(str(graph)).read_text(encoding="utf-8")
    return {name: value for name, value in candidates.items() if f"${{{name}}}" in text}


def _undeclared(graph: str | Path, provided: set[str]) -> set[str]:
    """Graph variables the caller has not supplied."""
    import re

    text = Path(str(graph)).read_text(encoding="utf-8")
    return set(re.findall(r"\$\{(\w+)\}", text)) - provided


def _run_batch(work: Any, items: Sequence[Any], jobs: int) -> None:
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    if jobs == 1:
        for item in items:
            work(item)
        return
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(work, items))


def _output_stem(product: Path) -> str:
    """A name carrying the sensing date, which the ingestion reads back."""
    stem = product.name
    for suffix in (".zip", ".SAFE", ".safe"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _resolution_rank(path: Path) -> tuple[int, str]:
    """Prefer the finest resolution a Sentinel-2 band is offered at."""
    for rank, marker in enumerate(("10m", "20m", "60m")):
        if marker in path.name:
            return (rank, path.name)
    return (len(("10m", "20m", "60m")), path.name)


def _resampling(resampling_enum: Any, name: str) -> Any:
    try:
        return getattr(resampling_enum, name)
    except AttributeError:
        raise ValueError(
            f"unknown resampling {name!r}; use one of nearest, bilinear, cubic, "
            "average, mode"
        ) from None


def _require_rasterio() -> Any:
    from gnn_sar.data.raster import _require_rasterio as require

    return require()
