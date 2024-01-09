import json

import numpy as np
import xarray as xr

from cedalion.dataclasses import PointType


def read_mrk_json(fname: str, crs: str) -> xr.DataArray:
    with open(fname) as fin:
        x = json.load(fin)

    units = []
    labels = []
    positions = []
    types = []

    for markup in x["markups"]:
        units.append(markup["coordinateUnits"])  # FIXME handling of units

        for cp in markup["controlPoints"]:
            labels.append(cp["label"])

            # 3x3 matrix. column vectors are coordinate axes
            orientation = np.asarray(cp["orientation"]).reshape(3, 3)

            pos = cp["position"]
            positions.append(pos @ orientation)
            types.append(PointType.LANDMARK)

    unique_units = list(set(units))
    if len(unique_units) > 1:
        raise ValueError(f"more than one unit found in {fname}: {unique_units}")

    pos = np.vstack(pos)

    result = xr.DataArray(
        positions,
        dims=["label", crs],
        coords={"label": ("label", labels), "type": ("label", types)},
        attrs={"units": unique_units[0]},
    )

    result = result.pint.quantify()

    return result


def read_digpts(fname: str, units="mm") -> xr.DataArray:
    with open(fname) as fin:
        lines = fin.readlines()

    labels = []
    coordinates = []

    for line in lines:
        label, coords = line.strip().split(":")
        coords = list(map(float, coords.split()))
        coordinates.append(coords)
        labels.append(label)

    result = xr.DataArray(
        coordinates,
        dims=["label", "pos"],
        coords={"label": labels},
        attrs={"units": units},
    )
    result = result.pint.quantify()

    return result
