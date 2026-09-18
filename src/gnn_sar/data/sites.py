"""The pilot study areas, as coordinates rather than as prose.

Roma, Viterbo and Milano: the three areas the project works on. Lazio covers
the two dynamics the proposal names — fragmented peri-urban expansion around
Rome, ground-mounted photovoltaic pressure in the Viterbo province — and Milano
adds the Pianura Padana, where haze and cloud cover are what make a SAR-first
design worth having.

A site is not a nice-to-have constant: it is the **one grid** everything else
has to land on. Sentinel-1, Sentinel-2, the ISPRA/SNPA cartography and the road
mask must share a CRS, a pixel size and an extent, or the ingestion refuses
them — so those four numbers belong in one place, versioned, instead of being
retyped into every ``gdalwarp`` call.

The extents were computed by projecting a WGS84 bounding box into the site's
UTM zone and snapping outward to 100 m, so every site starts on a round
coordinate and its width and height are whole numbers of 10 m pixels.

Sizes are deliberately modest — roughly 25 × 22 km, 5–7 megapixels — because a
study area is one full-batch graph. Cut a larger region into several sites
rather than growing one.
"""

from __future__ import annotations

from dataclasses import dataclass

from gnn_sar.data.raster import RasterGrid

__all__ = ["StudySite", "PILOT_SITES", "get_site", "format_sites"]


@dataclass(frozen=True)
class StudySite:
    """One area of interest, and the raster grid every input must share.

    Attributes:
        name: short identifier, used on the command line.
        description: what the area is, and why it is in the study.
        bbox: ``(west, south, east, north)`` in degrees, WGS84 — what the
            Copernicus catalogue is queried with.
        crs: the projected CRS the analysis runs in.
        extent: ``(xmin, ymin, xmax, ymax)`` in ``crs`` units — the exact
            footprint every raster is warped to.
        pixel_size: metres; 10 m is the Sentinel-1 IW and Sentinel-2 grid, and
            the resolution of the ISPRA/SNPA cartography.
        note: anything a reader needs before using the area.
    """

    name: str
    description: str
    bbox: tuple[float, float, float, float]
    crs: str
    extent: tuple[float, float, float, float]
    pixel_size: float = 10.0
    note: str = ""

    def __post_init__(self) -> None:
        west, south, east, north = self.bbox
        if west >= east or south >= north:
            raise ValueError(f"{self.name}: bbox must be (west, south, east, north)")
        xmin, ymin, xmax, ymax = self.extent
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"{self.name}: extent must be (xmin, ymin, xmax, ymax)")
        if self.pixel_size <= 0:
            raise ValueError(f"{self.name}: pixel_size must be > 0")
        for span, axis in (((xmax - xmin), "width"), ((ymax - ymin), "height")):
            if abs(span / self.pixel_size - round(span / self.pixel_size)) > 1e-6:
                raise ValueError(
                    f"{self.name}: the {axis} of the extent is not a whole number "
                    f"of {self.pixel_size} m pixels"
                )

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` in pixels."""
        xmin, ymin, xmax, ymax = self.extent
        return (
            int(round((ymax - ymin) / self.pixel_size)),
            int(round((xmax - xmin) / self.pixel_size)),
        )

    @property
    def size_km(self) -> tuple[float, float]:
        xmin, ymin, xmax, ymax = self.extent
        return ((xmax - xmin) / 1000.0, (ymax - ymin) / 1000.0)

    def grid(self) -> RasterGrid:
        """The :class:`~gnn_sar.data.raster.RasterGrid` every input must match."""
        height, width = self.shape
        xmin, _, _, ymax = self.extent
        return RasterGrid(
            height=height,
            width=width,
            transform=(xmin, self.pixel_size, 0.0, ymax, 0.0, -self.pixel_size),
            crs=self.crs,
        )

    def bbox_wkt(self) -> str:
        """The catalogue footprint, for an OData ``Intersects`` filter."""
        from gnn_sar.data.cdse import bbox_to_wkt

        return bbox_to_wkt(*self.bbox)

    def gdalwarp_options(self) -> str:
        """The flags that put any raster on this exact grid."""
        xmin, ymin, xmax, ymax = self.extent
        return (
            f"-t_srs {self.crs} -tr {self.pixel_size:g} {self.pixel_size:g} "
            f"-te {xmin:.0f} {ymin:.0f} {xmax:.0f} {ymax:.0f}"
        )

    def describe(self) -> str:
        height, width = self.shape
        return "\n".join(
            [
                f"{self.name}  —  {self.description}",
                f"  bbox (WGS84)  {', '.join(f'{v:g}' for v in self.bbox)}",
                f"  grid          {width} x {height} px, {self.pixel_size:g} m, {self.crs}",
                f"  extent        {', '.join(f'{v:.0f}' for v in self.extent)}"
                f"  ({self.size_km[0]:.1f} x {self.size_km[1]:.1f} km)",
                f"  gdalwarp      {self.gdalwarp_options()}",
                *([f"  note          {self.note}"] if self.note else []),
            ]
        )


PILOT_SITES: dict[str, StudySite] = {
    "roma": StudySite(
        name="roma",
        description="Rome's eastern and south-eastern peri-urban fringe, "
        "across the ring road — fragmented expansion at the edge of a "
        "metropolis",
        bbox=(12.40, 41.78, 12.72, 41.99),
        crs="EPSG:32633",
        extent=(283900.0, 4627800.0, 311200.0, 4652000.0),
        note="the densest already-built area of the three: expect most nodes to "
        "be excluded as pre-existing, and check that enough transitions remain",
    ),
    "viterbo": StudySite(
        name="viterbo",
        description="Viterbo and its countryside — the province the SNPA report "
        "puts first for annual land take, driven by ground-mounted photovoltaic",
        bbox=(11.95, 42.35, 12.25, 42.53),
        crs="EPSG:32633",
        extent=(248700.0, 4692300.0, 274200.0, 4713200.0),
        note="ground-mounted PV is *reversible* land take in the ISPRA "
        "classification: decide explicitly whether to include those codes "
        "(docs/dati_etichette_ispra.md). The western edge sits just outside UTM "
        "zone 33; the distortion is negligible, but keep one CRS for the series. "
        "For the Tarquinia/Montalto di Castro PV cluster, define a second site "
        "around 11.45-11.80 E, 42.20-42.45 N rather than widening this one",
    ),
    "milano": StudySite(
        name="milano",
        description="Milan south and its hinterland — the Pianura Padana case, "
        "where haze and cloud are the reason for a SAR-first design",
        bbox=(9.05, 45.38, 9.35, 45.58),
        crs="EPSG:32632",
        extent=(503900.0, 5025100.0, 527500.0, 5047500.0),
        note="UTM zone 32, unlike the two Lazio sites: never mix the two in one "
        "study area. Sentinel-2 contributes little here for months at a time, "
        "which is the point of the comparison with Lazio",
    ),
}


def get_site(name: str) -> StudySite:
    """Look a pilot site up by name."""
    try:
        return PILOT_SITES[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown site {name!r}; the pilot areas are "
            f"{', '.join(sorted(PILOT_SITES))}"
        ) from None


def format_sites() -> str:
    """Every pilot site, for ``gnn-sar sites``."""
    return "\n\n".join(site.describe() for site in PILOT_SITES.values())
