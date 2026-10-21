"""Real-video ingestion: folder of videos + captions + PROVENANCE.json -> sharded safetensors dataset."""
from .report import inspect_dataset
from .pipeline import BUCKETS, IngestError, IngestOptions, ingest_folder, load_provenance

__all__ = ["BUCKETS", "IngestError", "IngestOptions", "ingest_folder", "inspect_dataset", "load_provenance"]
