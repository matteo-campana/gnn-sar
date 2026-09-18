"""Searching and downloading from the Copernicus Data Space Ecosystem.

CDSE is the current distribution point for Sentinel data (it replaced the
Copernicus Open Access Hub). Its OData catalogue answers two questions this
project needs — *which products cover my area in this period* and *give me that
one* — and needs nothing but an account and HTTP.

The query builders (:func:`bbox_to_wkt`, :func:`product_filter`) are pure
strings and are unit-tested; the two functions that talk to the network are
not, and they are thin on purpose, so that a failure is a plain HTTP error you
can read rather than a layer of retries hiding it.

Credentials come from the environment (``CDSE_USERNAME`` / ``CDSE_PASSWORD``)
and are never written to disk by this module. A token is valid for a few
minutes only: fetch one per batch of downloads rather than one per session.

Requires the optional ``download`` extra (``requests``).

See ``docs/guida_dati_sentinel.md`` for the full workflow, including the
alternatives this module deliberately does not wrap (the Copernicus Browser,
S3/``eodata``, openEO, Sentinel Hub).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "CATALOGUE_URL",
    "DOWNLOAD_URL",
    "TOKEN_URL",
    "CdseCredentials",
    "bbox_to_wkt",
    "product_filter",
    "access_token",
    "search_products",
    "download_product",
    "format_products",
]

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
_PUBLIC_CLIENT = "cdse-public"


@dataclass(frozen=True)
class CdseCredentials:
    """A Copernicus Data Space Ecosystem login.

    Register once, free, at https://dataspace.copernicus.eu — the same account
    works for the browser, the OData API, S3 and openEO.
    """

    username: str
    password: str

    @classmethod
    def from_env(
        cls, username_var: str = "CDSE_USERNAME", password_var: str = "CDSE_PASSWORD"
    ) -> "CdseCredentials":
        username, password = os.environ.get(username_var), os.environ.get(password_var)
        if not username or not password:
            raise RuntimeError(
                f"set {username_var} and {password_var} to your Copernicus Data Space "
                "Ecosystem credentials (register at https://dataspace.copernicus.eu)"
            )
        return cls(username=username, password=password)


def bbox_to_wkt(west: float, south: float, east: float, north: float) -> str:
    """A WKT polygon for an OData ``OData.CSC.Intersects`` filter.

    Coordinates are longitude/latitude in WGS84 (EPSG:4326), which is what the
    catalogue indexes footprints in regardless of the CRS you later work in.
    """
    if west >= east or south >= north:
        raise ValueError(
            f"expected west < east and south < north, got "
            f"({west}, {south}, {east}, {north})"
        )
    corners = [(west, south), (east, south), (east, north), (west, north), (west, south)]
    points = ", ".join(f"{lon} {lat}" for lon, lat in corners)
    return f"POLYGON(({points}))"


def product_filter(
    collection: str,
    start: date | datetime | str,
    end: date | datetime | str,
    footprint_wkt: str | None = None,
    product_type: str | None = None,
    attributes: dict[str, Any] | None = None,
    max_cloud_cover: float | None = None,
    name_contains: str | None = None,
    online_only: bool = False,
) -> str:
    """Build the OData ``$filter`` expression of a product search.

    Args:
        collection: ``"SENTINEL-1"`` or ``"SENTINEL-2"``.
        start, end: sensing interval; ``end`` is exclusive.
        footprint_wkt: area of interest, see :func:`bbox_to_wkt`. A product is
            returned when its footprint *intersects* it, so a small area still
            matches a whole 250 km swath — that is the intended behaviour, but
            it does mean the count is not a count of useful scenes.
        product_type: e.g. ``"IW_GRDH_1S"``, ``"SLC"``, ``"S2MSI2A"``.
        attributes: any further ``name -> value`` string attributes to match
            exactly, e.g. ``{"orbitDirection": "DESCENDING",
            "relativeOrbitNumber": 22}``. Restricting a Sentinel-1 series to one
            relative orbit is nearly always right: backscatter depends on the
            viewing geometry, so mixing orbits injects a change signal that has
            nothing to do with the ground.
        max_cloud_cover: Sentinel-2 only, in percent.
        name_contains: substring of the product name, e.g. ``"GRDH"``.
        online_only: keep only products available for immediate download.

    Returns:
        The ``$filter`` value, unencoded.
    """
    clauses = [
        f"Collection/Name eq '{collection}'",
        f"ContentDate/Start ge {_odata_time(start)}",
        f"ContentDate/Start lt {_odata_time(end)}",
    ]
    if footprint_wkt:
        clauses.append(
            f"OData.CSC.Intersects(area=geography'SRID=4326;{footprint_wkt}')"
        )
    if name_contains:
        clauses.append(f"contains(Name,'{name_contains}')")
    if product_type:
        clauses.append(_string_attribute("productType", product_type))
    for name, value in (attributes or {}).items():
        clauses.append(
            _string_attribute(name, value)
            if isinstance(value, str)
            else _numeric_attribute(name, value)
        )
    if max_cloud_cover is not None:
        clauses.append(
            "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
            f"and att/OData.CSC.DoubleAttribute/Value lt {float(max_cloud_cover)})"
        )
    if online_only:
        clauses.append("Online eq true")
    return " and ".join(clauses)


def access_token(credentials: CdseCredentials | None = None, timeout: float = 60.0) -> str:
    """Exchange a username and password for a short-lived bearer token."""
    requests = _require_requests()
    credentials = credentials or CdseCredentials.from_env()
    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": _PUBLIC_CLIENT,
            "grant_type": "password",
            "username": credentials.username,
            "password": credentials.password,
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"CDSE refused the credentials ({response.status_code}): {response.text[:200]}"
        )
    return str(response.json()["access_token"])


def search_products(
    filter_expression: str,
    top: int = 100,
    order_by: str = "ContentDate/Start asc",
    expand_attributes: bool = False,
    timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """Run an OData product search and return every page of results.

    The catalogue is public: no token is needed to search, only to download.

    Args:
        filter_expression: from :func:`product_filter`.
        top: page size, at most 1000.
        order_by: sort order; chronological by default, which is the order a
            time series wants.
        expand_attributes: include the per-product attribute list (orbit,
            polarisations, cloud cover). Heavier, and usually worth it.
        timeout: per-request timeout in seconds.

    Returns:
        The products, each a dict with at least ``Id``, ``Name``,
        ``ContentDate`` and ``ContentLength``.
    """
    requests = _require_requests()
    if not 1 <= top <= 1000:
        raise ValueError("top must be between 1 and 1000")

    params: dict[str, Any] | None = {
        "$filter": filter_expression,
        "$orderby": order_by,
        "$top": top,
    }
    if expand_attributes:
        params["$expand"] = "Attributes"

    url: str | None = CATALOGUE_URL
    products: list[dict[str, Any]] = []
    while url:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        products.extend(payload.get("value", []))
        # The next link already carries the query; passing params again would
        # duplicate it and the catalogue would reject the request.
        url, params = payload.get("@odata.nextLink"), None
    return products


def download_product(
    product_id: str,
    destination: str | Path,
    token: str,
    chunk_size: int = 4 * 1024 * 1024,
    timeout: float = 600.0,
    overwrite: bool = False,
) -> Path:
    """Download one product as a zip, streaming it to ``destination``.

    Products are large — a Sentinel-1 IW GRDH scene is around 1 GB, a
    Sentinel-2 L2A tile 0.5–1 GB — so the transfer is streamed and an existing
    file is kept unless ``overwrite`` is set. CDSE allows a small number of
    concurrent downloads per account; four parallel transfers is a safe budget.

    Args:
        product_id: the ``Id`` field of a search result.
        destination: the file to write (``.zip``).
        token: from :func:`access_token`.
        chunk_size: streaming chunk in bytes.
        timeout: per-request timeout in seconds.
        overwrite: re-download a file that already exists.

    Returns:
        The path written.
    """
    requests = _require_requests()
    destination = Path(destination)
    if destination.exists() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    url = f"{DOWNLOAD_URL}({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    # Rename only once the transfer completed, so an interrupted download can
    # never be mistaken for a product by the ingestion step.
    partial.replace(destination)
    return destination


def _string_attribute(name: str, value: str) -> str:
    return (
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq '{name}' "
        f"and att/OData.CSC.StringAttribute/Value eq '{value}')"
    )


def _numeric_attribute(name: str, value: float | int) -> str:
    kind = "IntegerAttribute" if isinstance(value, int) else "DoubleAttribute"
    return (
        f"Attributes/OData.CSC.{kind}/any(att:att/Name eq '{name}' "
        f"and att/OData.CSC.{kind}/Value eq {value})"
    )


def _odata_time(value: date | datetime | str) -> str:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day)
    else:
        raise TypeError(f"cannot read a date from {value!r}")
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _require_requests() -> Any:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "talking to the CDSE catalogue requires requests; install it with "
            "`poetry install --extras download` (or `pip install requests`)"
        ) from exc
    return requests


def format_products(products: Sequence[dict[str, Any]]) -> str:
    """One line per product: name, sensing start and size — for the CLI."""
    lines = []
    for product in products:
        size = float(product.get("ContentLength", 0)) / 1e9
        start = str(product.get("ContentDate", {}).get("Start", "?"))[:19]
        lines.append(f"{start}  {size:6.2f} GB  {product.get('Name', product.get('Id'))}")
    return "\n".join(lines)
