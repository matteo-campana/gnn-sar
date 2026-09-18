"""From Sentinel products to the model's channels (proposal §5.1).

Everything here is pure NumPy: it takes the pixel arrays that
:mod:`gnn_sar.data.raster` (or any other reader) produced and turns them into
the channels the graph consumes, plus the calendar bookkeeping that turns an
irregular acquisition list into the regular composite axis ``T``.

Three jobs:

*when*
    :class:`PeriodGrid` maps sensing dates to composites and composites to
    years. The year map is what lets an annual ISPRA/SNPA label supervise a
    bag of sub-annual predictions, so it has to be derived from the calendar,
    not assumed.
*Sentinel-1*
    :func:`sentinel1_channels` converts calibrated backscatter to dB and adds
    the VV/VH ratio. The dB conversion is not cosmetic: speckle is
    multiplicative in linear power and additive in dB, which is the domain the
    standardiser and the network expect.
*Sentinel-2*
    :func:`scl_valid_mask` and :func:`sentinel2_indices` implement the optical
    half of the fusion the proposal keeps optional — masked by the L2A scene
    classification, then reduced to spectral indices rather than raw
    reflectance, so the extra channels stay few and physically readable.

Sentinel-2 arrives with holes (cloud), Sentinel-1 does not; that asymmetry is
why the compositing helpers take an ``ignore_nan`` flag rather than assuming
one behaviour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

from gnn_sar.data.features import to_decibel

__all__ = [
    "SCL_CLASSES",
    "DEFAULT_INVALID_SCL",
    "PeriodGrid",
    "sensing_date",
    "sensing_dates",
    "normalized_difference",
    "scl_valid_mask",
    "sentinel1_channels",
    "sentinel2_indices",
    "mask_invalid",
]

# Sentinel-1 and Sentinel-2 product names both carry the sensing start as
# ``YYYYMMDDTHHMMSS``; granule files (``T33TTG_20240117T100319_B02_10m.jp2``)
# and user-renamed tiles usually keep at least the ``YYYYMMDD`` part.
_DATETIME_PATTERN = re.compile(r"(?<!\d)(\d{8}T\d{6})(?!\d)")
_DATE_PATTERN = re.compile(r"(?<!\d)(\d{8})(?!\d)")

SCL_CLASSES: dict[int, str] = {
    0: "no_data",
    1: "saturated_or_defective",
    2: "cast_shadows",
    3: "cloud_shadows",
    4: "vegetation",
    5: "not_vegetated",
    6: "water",
    7: "unclassified",
    8: "cloud_medium_probability",
    9: "cloud_high_probability",
    10: "thin_cirrus",
    11: "snow_or_ice",
}

DEFAULT_INVALID_SCL: tuple[int, ...] = (0, 1, 2, 3, 8, 9, 10, 11)
"""Scene-classification classes an optical feature must not be computed from.

Water (6) and unclassified (7) are kept: both are legitimate ground states for
this task, and dropping them would silently remove the wet fringe of a site.
"""

_S2_NORMALISED_DIFFERENCES: dict[str, tuple[str, str]] = {
    # index: (positive band, negative band)
    "ndvi": ("B08", "B04"),   # vegetation — falls when land is cleared
    "ndbi": ("B11", "B08"),   # built-up — the optical counterpart of the task
    "ndwi": ("B03", "B08"),   # open water (McFeeters)
    "mndwi": ("B03", "B11"),  # open water, modified (Xu)
}


@dataclass(frozen=True)
class PeriodGrid:
    """The regular composite axis an irregular revisit is projected onto.

    Sentinel-1 passes over a given area every few days, on dates that depend on
    the orbit and change whenever the constellation is reconfigured. The model
    consumes a fixed-length sequence, so the two have to be reconciled once,
    explicitly, and recorded — which composite covers which calendar span is
    what every delay in composites later has to be read against.

    Attributes:
        period: ``"month"``, ``"dekad"`` (the 1–10 / 11–20 / 21–end thirds of a
            month used by agro-meteorological services), or a number of days.
        start: first day of the first composite.
        n_periods: length of the composite axis, ``T``.
    """

    period: Literal["month", "dekad"] | int
    start: date
    n_periods: int

    def __post_init__(self) -> None:
        if isinstance(self.period, int):
            if self.period < 1:
                raise ValueError("a period given in days must be >= 1")
        elif self.period not in ("month", "dekad"):
            raise ValueError(
                f"period must be 'month', 'dekad' or a number of days, got {self.period!r}"
            )
        if self.n_periods < 1:
            raise ValueError("n_periods must be >= 1")

    @classmethod
    def covering(
        cls,
        dates: Sequence[date | datetime | np.datetime64 | str],
        period: Literal["month", "dekad"] | int = "month",
        start: date | datetime | str | None = None,
        n_periods: int | None = None,
    ) -> "PeriodGrid":
        """The shortest grid of ``period`` composites containing every date.

        Args:
            dates: the sensing dates of the acquisitions.
            period: composite length.
            start: force the first composite. Defaults to the composite the
                earliest acquisition falls in. Pin it when two study areas, or
                two sources, must share one time axis.
            n_periods: force the length. Defaults to just covering the dates.
        """
        parsed = [_as_date(value) for value in dates]
        if not parsed:
            raise ValueError("at least one date is required")
        first = _period_start(min(parsed), period) if start is None else _as_date(start)
        if first > min(parsed):
            raise ValueError(
                f"start {first} is after the earliest acquisition {min(parsed)}"
            )
        grid = cls(period=period, start=first, n_periods=1)
        needed = int(grid._raw_index(max(parsed))) + 1
        if n_periods is not None and n_periods < needed:
            raise ValueError(
                f"n_periods={n_periods} cannot hold {needed} composites of "
                f"{period} starting on {first}"
            )
        return cls(period=period, start=first, n_periods=n_periods or needed)

    def index_of(
        self, dates: Sequence[date | datetime | np.datetime64 | str]
    ) -> np.ndarray:
        """``(K,)`` composite index of each acquisition.

        Raises:
            ValueError: if an acquisition falls outside the grid. Silently
                dropping it would shorten the series without saying so.
        """
        parsed = [_as_date(value) for value in dates]
        index = np.array([self._raw_index(value) for value in parsed], dtype=np.int64)
        outside = [
            str(parsed[i])
            for i in np.flatnonzero((index < 0) | (index >= self.n_periods))
        ]
        if outside:
            raise ValueError(
                f"{len(outside)} acquisitions fall outside the grid "
                f"[{self.start}, {self.n_periods} x {self.period}): {outside[:5]}"
            )
        return index

    def starts(self) -> list[date]:
        """First day of each composite."""
        return [self._offset(self.start, step) for step in range(self.n_periods)]

    def labels(self) -> list[str]:
        """Human-readable composite names, e.g. ``2024-03`` or ``2024-03-D2``."""
        if self.period == "month":
            return [f"{d.year:04d}-{d.month:02d}" for d in self.starts()]
        if self.period == "dekad":
            return [
                f"{d.year:04d}-{d.month:02d}-D{min((d.day - 1) // 10 + 1, 3)}"
                for d in self.starts()
            ]
        return [d.isoformat() for d in self.starts()]

    def years(self) -> np.ndarray:
        """``(T,)`` calendar year of each composite."""
        return np.array([d.year for d in self.starts()], dtype=np.int64)

    def year_index(self) -> np.ndarray:
        """``(T,)`` row of ``annual_labels`` each composite is supervised by.

        Zero-based and contiguous, which is the contract
        :class:`gnn_sar.data.graph.SpatioTemporalGraph` checks. Pair it with
        :meth:`year_labels` to know which calendar year each row means.
        """
        years = self.years()
        return (years - years.min()).astype(np.int64)

    def year_labels(self) -> list[int]:
        """The calendar year behind each row of ``annual_labels``."""
        years = self.years()
        return list(range(int(years.min()), int(years.max()) + 1))

    def _raw_index(self, value: date) -> int:
        if self.period == "month":
            return (value.year - self.start.year) * 12 + (value.month - self.start.month)
        if self.period == "dekad":
            months = (value.year - self.start.year) * 12 + (
                value.month - self.start.month
            )
            return months * 3 + _dekad(value) - _dekad(self.start)
        return (value - self.start).days // int(self.period)

    def _offset(self, origin: date, steps: int) -> date:
        if self.period == "month":
            month = origin.month - 1 + steps
            return date(origin.year + month // 12, month % 12 + 1, 1)
        if self.period == "dekad":
            total = (_dekad(origin) - 1) + steps
            month = origin.month - 1 + total // 3
            return date(origin.year + month // 12, month % 12 + 1, (total % 3) * 10 + 1)
        from datetime import timedelta

        return origin + timedelta(days=steps * int(self.period))


def sensing_date(name: str | Path) -> datetime:
    """The sensing start encoded in a Sentinel product or file name.

    Works on full product names (``S1A_IW_GRDH_1SDV_20240117T051253_...``,
    ``S2B_MSIL2A_20240117T100319_...``), on granule files
    (``T33TTG_20240117T100319_B02_10m.jp2``) and on anything else that carries
    a ``YYYYMMDD`` or ``YYYYMMDDTHHMMSS`` token — which covers the names a SNAP
    graph or a ``gdalwarp`` step typically produces.

    Raises:
        ValueError: when no date can be read. Renaming the file to include the
            sensing date is the fix; guessing from the modification time would
            put the series in the wrong order the first time a file is copied.
    """
    text = Path(name).name
    match = _DATETIME_PATTERN.search(text)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
    match = _DATE_PATTERN.search(text)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d")
        except ValueError:  # an eight-digit token that is not a date
            pass
    raise ValueError(
        f"no sensing date in {text!r}: expected a YYYYMMDD or YYYYMMDDTHHMMSS "
        "token, as in S1A_IW_GRDH_1SDV_20240117T051253_..."
    )


def sensing_dates(names: Iterable[str | Path]) -> list[datetime]:
    """:func:`sensing_date` over many names, order preserved."""
    return [sensing_date(name) for name in names]


def normalized_difference(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """``(a - b) / (a + b)``, ``NaN`` where the sum vanishes."""
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    total = positive + negative
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(total) > 0, (positive - negative) / total, np.nan)


def scl_valid_mask(
    scl: np.ndarray, invalid: Sequence[int] = DEFAULT_INVALID_SCL
) -> np.ndarray:
    """Boolean mask of the Sentinel-2 pixels an index may be computed from.

    Args:
        scl: the L2A scene-classification layer, same shape as the bands.
        invalid: classes to reject, see :data:`SCL_CLASSES`.
    """
    scl = np.asarray(scl)
    unknown = [c for c in invalid if c not in SCL_CLASSES]
    if unknown:
        raise ValueError(f"unknown SCL classes {unknown}, expected 0..11")
    return ~np.isin(scl, list(invalid))


def mask_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``NaN`` wherever ``valid`` is false, broadcasting over leading axes."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    return np.where(valid, values, np.nan)


def sentinel1_channels(
    bands: Mapping[str, np.ndarray],
    linear: bool = True,
    ratio: bool = True,
    db_floor: float = -35.0,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the Sentinel-1 channels from calibrated backscatter.

    Args:
        bands: at least ``VV`` and/or ``VH``, each ``(..., H, W)``; optionally
            ``coherence`` (InSAR, in ``[0, 1]``), which is passed through
            unchanged because it is already a ratio.
        linear: whether the inputs are linear power :math:`\\sigma^0` (what
            SNAP's calibration and the CDSE RTC product give) rather than dB.
            Non-positive samples become ``NaN``: they are nodata, not −∞.
        ratio: append ``VV − VH`` in dB, i.e. the VV/VH ratio in linear power.
            It is invariant to the multiplicative part of an orbit-to-orbit
            gain change, so it survives the radiometric drift a long series has.
        db_floor: floor of the dB conversion, see
            :func:`gnn_sar.data.features.to_decibel`.

    Returns:
        ``(values, names)`` with ``values`` of shape ``(C, ...)`` stacked along
        a new leading channel axis.
    """
    known = {"VV", "VH", "HH", "HV", "coherence"}
    unknown = sorted(set(bands) - known)
    if unknown:
        raise ValueError(
            f"unknown Sentinel-1 bands {unknown}, expected any of {sorted(known)}"
        )
    polarisations = [name for name in ("VV", "VH", "HH", "HV") if name in bands]
    if not polarisations:
        raise ValueError("at least one polarisation (VV, VH, HH, HV) is required")

    channels: list[np.ndarray] = []
    names: list[str] = []
    decibels: dict[str, np.ndarray] = {}
    for name in polarisations:
        values = np.asarray(bands[name], dtype=np.float64)
        if linear:
            # Nodata in a calibrated product shows up as 0 or as a small
            # negative; neither is a backscatter level, so neither becomes one.
            with np.errstate(invalid="ignore"):
                positive = np.where(values > 0.0, values, np.nan)
            values = to_decibel(positive, floor=db_floor)
        decibels[name] = values
        channels.append(values)
        names.append(f"{name}_db")

    if ratio:
        pairs = [("VV", "VH"), ("HH", "HV")]
        for co, cross in pairs:
            if co in decibels and cross in decibels:
                channels.append(decibels[co] - decibels[cross])
                names.append(f"{co}{cross}_ratio_db")

    if "coherence" in bands:
        channels.append(np.asarray(bands["coherence"], dtype=np.float64))
        names.append("coherence")

    return np.stack(channels), names


def sentinel2_indices(
    bands: Mapping[str, np.ndarray],
    indices: Sequence[str] = ("ndvi", "ndbi"),
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Spectral indices from Sentinel-2 L2A surface reflectance.

    Indices rather than raw reflectance, for two reasons: they are ratios, so
    they survive the residual atmospheric and illumination differences between
    two dates far better than a band does, and they keep the optical
    contribution to a handful of channels that a reader can interpret next to
    the backscatter.

    Args:
        bands: reflectance arrays keyed by Sentinel-2 band name (``B02``,
            ``B03``, ``B04``, ``B08``, ``B11``, ``B12``), each ``(..., H, W)``
            and already scaled to reflectance (divide the L2A integers by
            10000). Every band an index needs must be present.
        indices: any of ``ndvi``, ``ndbi``, ``ndwi``, ``mndwi``, ``bsi``.
        valid: optional boolean mask from :func:`scl_valid_mask`; rejected
            pixels come out as ``NaN``.

    Returns:
        ``(values, names)`` with ``values`` of shape ``(len(indices), ...)``.
    """
    supported = {*_S2_NORMALISED_DIFFERENCES, "bsi"}
    unknown = [name for name in indices if name not in supported]
    if unknown:
        raise ValueError(f"unknown indices {unknown}, expected any of {sorted(supported)}")
    if not indices:
        raise ValueError("at least one index is required")

    def band(name: str, index: str) -> np.ndarray:
        if name not in bands:
            raise ValueError(f"index {index!r} needs band {name}, which was not given")
        return np.asarray(bands[name], dtype=np.float64)

    channels = []
    for name in indices:
        if name in _S2_NORMALISED_DIFFERENCES:
            positive, negative = _S2_NORMALISED_DIFFERENCES[name]
            values = normalized_difference(band(positive, name), band(negative, name))
        else:  # bsi — bare soil, which is what a cleared building site is
            values = normalized_difference(
                band("B11", name) + band("B04", name),
                band("B08", name) + band("B02", name),
            )
        channels.append(values if valid is None else mask_invalid(values, valid))

    return np.stack(channels), [f"s2_{name}" for name in indices]


def _dekad(value: date) -> int:
    """1, 2 or 3 — which third of its month a date falls in."""
    return min((value.day - 1) // 10 + 1, 3)


def _period_start(value: date, period: Literal["month", "dekad"] | int) -> date:
    if period == "month":
        return date(value.year, value.month, 1)
    if period == "dekad":
        return date(value.year, value.month, (_dekad(value) - 1) * 10 + 1)
    return value


def _as_date(value: date | datetime | np.datetime64 | str) -> date:
    if isinstance(value, np.datetime64):
        value = value.astype("datetime64[s]").astype(datetime)
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"cannot read a date from {value!r}")
