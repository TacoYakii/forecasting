"""
Configuration management for unified NWP preprocessing pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import yaml


@dataclass
class NWPPreprocessConfig:
    """
    Central configuration for the NWP preprocessing pipeline.

    All paths and parameters are managed here to eliminate hardcoded values.

    Attributes:
        input_dir: Directory containing raw NWP files (GRIB2 or TXT).
        output_dir: Directory where per-coordinate CSV results are saved.
        reader_type: Selects which Reader to use.
            - "grib2": for ECMWF, NOAA, and any GRIB2-based source
            - "kma_txt": for KMA text-format data
        wind_components: Mapping of height suffix to (u_col, v_col) pairs.
            e.g., {"10": ("u10", "v10"), "100": ("u100", "v100")}
            Derived variables will be named wspd10, wdir10, wspd100, wdir100, etc.
        geopotential_col: Column name for geopotential height (provider-specific).
        derive_variables: List of derived variables to compute.
            Supported: "wspd" (wind speed), "wdir" (wind direction), "height" (geopotential → geometric height)
        n_workers: Number of parallel worker processes.
    """

    input_dir: Path
    output_dir: Path
    reader_type: Literal["grib2", "kma_txt"]

    # Variable name mapping (provider-specific)
    wind_components: Dict[str, Tuple[str, str]] = field(
        default_factory=lambda: {"10": ("u10", "v10")}
    )
    geopotential_col: str = "geo_potential_height"

    # Derived variable configuration
    derive_variables: List[str] = field(
        default_factory=lambda: ["wspd", "wdir"]
    )

    # Processing options
    n_workers: int = 4

    def __post_init__(self):
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)

        valid_readers = {"grib2", "kma_txt"}
        if self.reader_type not in valid_readers:
            raise ValueError(
                f"Invalid reader_type '{self.reader_type}'. "
                f"Must be one of: {valid_readers}"
            )

        valid_derive = {"wspd", "wdir", "height"}
        invalid = set(self.derive_variables) - valid_derive
        if invalid:
            raise ValueError(
                f"Invalid derive_variables: {invalid}. "
                f"Supported: {valid_derive}"
            )

        # Convert wind_components values to tuples if loaded from YAML as lists
        self.wind_components = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in self.wind_components.items()
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NWPPreprocessConfig":
        """Load configuration from a YAML file.

        Example YAML:
            input_dir: /data/original/gasiri/ECMWF
            output_dir: /data/preprocess/gasiri/ECMWF
            reader_type: grib2
            wind_components:
                "10": ["u10", "v10"]
                "100": ["u100", "v100"]
            derive_variables:
                - wspd
                - wdir
            n_workers: 4
        """
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

