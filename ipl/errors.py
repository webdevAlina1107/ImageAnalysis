import rasterio
import warnings


# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class IPLError(Exception):
    pass
