"""
NWP preprocessing pipeline orchestrator.

Coordinates the full preprocessing flow:
    raw NWP files → Reader → Deriver → Validator → MissingHandler → CSV output

Uses concurrent.futures.ProcessPoolExecutor for parallel file processing.
"""
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .config import NWPPreprocessConfig
from .processors.deriver import VariableDeriver
from .processors.missing_handler import MissingHandler
from .processors.validator import DataValidator
from .readers.base import AbstractReader
from .readers.grib2_reader import Grib2Reader
from .readers.kma_txt_reader import KMATxtReader

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""

    total_files: int = 0
    processed: int = 0
    failed: int = 0
    skipped_invalid: int = 0
    failed_files: list = field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "processed": self.processed,
            "failed": self.failed,
            "skipped_invalid": self.skipped_invalid,
            "failed_files": self.failed_files,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


def _process_single_file(args: tuple) -> dict:
    """
    Process a single NWP file: read → derive → validate → save.

    This is a module-level function (not a method) so it can be pickled
    for multiprocessing.

    Args:
        args: Tuple of (file_path, config_dict, reader_kwargs)

    Returns:
        Dict with status information.
    """
    file_path, config_dict, reader_type, reader_kwargs = args

    try:
        config = NWPPreprocessConfig(**config_dict)

        # Create reader
        if reader_type == "grib2":
            reader = Grib2Reader(**reader_kwargs)
        elif reader_type == "kma_txt":
            reader = KMATxtReader(**reader_kwargs)
        else:
            raise ValueError(f"Unknown reader_type: {reader_type}")

        file_path = Path(file_path)

        # Validate file
        if not reader.validate_file(file_path):
            return {"status": "skipped", "file": str(file_path), "reason": "invalid"}

        # Read file → per-coordinate DataFrames
        coord_dfs = reader.read(file_path)

        # Create processors
        deriver = VariableDeriver(
            wind_components=config.wind_components,
            geopotential_col=config.geopotential_col,
        )
        validator = DataValidator()
        missing_handler = MissingHandler()

        # Process each coordinate
        for coord_key, df in coord_dfs.items():
            # 1. Derive variables
            df = deriver.apply_all(df, config.derive_variables)

            # 2. Validate
            df = validator.validate(df)

            # 3. Fill missing from past data
            basis_time = df["basis_time"].iloc[0] if "basis_time" in df.columns else None
            if basis_time is not None:
                df = missing_handler.fill(
                    df=df,
                    coord_key=coord_key,
                    output_dir=config.output_dir,
                    basis_time=basis_time,
                )

            # 4. Save (normalize filename to YYYY-MM-DD_HH.csv)
            if basis_time is not None:
                basis_time_str = basis_time.strftime("%Y-%m-%d_%H")
            else:
                basis_time_str = file_path.stem
            out_dir = config.output_dir / coord_key
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{basis_time_str}.csv"
            df.to_csv(out_path, index=True)

        return {
            "status": "success",
            "file": str(file_path),
            "coordinates": len(coord_dfs),
        }

    except Exception as e:
        logger.error("Failed to process %s: %s", file_path, e)
        return {"status": "failed", "file": str(file_path), "error": str(e)}


class NWPPreprocessPipeline:
    """
    Unified NWP preprocessing pipeline.

    Orchestrates the full flow from raw NWP files to per-coordinate CSVs:
        1. Discover files (via Reader.list_files)
        2. For each file (parallel via ProcessPoolExecutor):
            a. Validate file integrity
            b. Read and split by coordinate
            c. Compute derived variables (wspd, wdir, height)
            d. Validate data and set is_valid flags
            e. Fill missing data from nearest past
            f. Save as CSV

    Usage:
        config = NWPPreprocessConfig(
            input_dir="/data/original/gasiri/ECMWF",
            output_dir="/data/preprocess/gasiri/ECMWF",
            reader_type="grib2",
        )
        pipeline = NWPPreprocessPipeline(config)
        stats = pipeline.run()
    """

    def __init__(
        self,
        config: NWPPreprocessConfig,
        reader_kwargs: Optional[dict] = None,
    ):
        """
        Args:
            config: Pipeline configuration.
            reader_kwargs: Additional keyword arguments passed to the Reader
                constructor (e.g., cfgrib_kwargs for Grib2Reader,
                parameter_codes for KmaTxtReader).
        """
        self.config = config
        self.reader_kwargs = reader_kwargs or {}
        self.reader = self._create_reader()

    def _create_reader(self) -> AbstractReader:
        """Create the appropriate reader based on config."""
        if self.config.reader_type == "grib2":
            return Grib2Reader(**self.reader_kwargs)
        elif self.config.reader_type == "kma_txt":
            return KMATxtReader(**self.reader_kwargs)
        else:
            raise ValueError(f"Unknown reader_type: {self.config.reader_type}")

    def run(self) -> PipelineStats:
        """
        Execute the full preprocessing pipeline.

        Returns:
            PipelineStats with processing results summary.
        """
        stats = PipelineStats(start_time=datetime.now().isoformat())

        # Discover files
        file_paths = self.reader.list_files(self.config.input_dir)
        stats.total_files = len(file_paths)

        if not file_paths:
            logger.warning("No files found in %s", self.config.input_dir)
            stats.end_time = datetime.now().isoformat()
            return stats

        logger.info("Found %d files to process", len(file_paths))

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare arguments for parallel processing
        config_dict = {
            "input_dir": str(self.config.input_dir),
            "output_dir": str(self.config.output_dir),
            "reader_type": self.config.reader_type,
            "wind_components": {k: list(v) for k, v in self.config.wind_components.items()},
            "geopotential_col": self.config.geopotential_col,
            "derive_variables": self.config.derive_variables,
            "n_workers": self.config.n_workers,
        }

        task_args = [
            (str(fp), config_dict, self.config.reader_type, self.reader_kwargs)
            for fp in file_paths
        ]

        # Process files in parallel
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = {
                executor.submit(_process_single_file, args): args[0]
                for args in task_args
            }

            with tqdm(total=len(futures), desc="Processing NWP files", unit="files") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        file_path = futures[future]
                        logger.error("Unexpected error processing %s: %s", file_path, e)
                        stats.failed += 1
                        stats.failed_files.append(str(file_path))
                        pbar.update(1)
                        continue

                    if result["status"] == "success":
                        stats.processed += 1
                    elif result["status"] == "skipped":
                        stats.skipped_invalid += 1
                    elif result["status"] == "failed":
                        stats.failed += 1
                        stats.failed_files.append(result["file"])

                    pbar.update(1)

        stats.end_time = datetime.now().isoformat()

        # Save processing log
        log_path = self.config.output_dir / "preprocess_log.json"
        with open(log_path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)

        # Print summary
        logger.info(
            "Pipeline complete: %d/%d processed, %d failed, %d skipped",
            stats.processed, stats.total_files, stats.failed, stats.skipped_invalid,
        )

        return stats
