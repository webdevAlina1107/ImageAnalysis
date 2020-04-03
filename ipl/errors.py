import warnings

import rasterio

IGNORED_WARNINGS = (rasterio.errors.NotGeoreferencedWarning,)
for warning_type in IGNORED_WARNINGS:
    warnings.filterwarnings("ignore",
                            category=warning_type)


class IPLError(Exception):
    pass
