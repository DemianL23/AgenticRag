"""Adapters that turn source files into page-level documents."""

from agenticrag.ingestion.parsers.pymupdf import PdfParseError, parse_pdf

__all__ = ["PdfParseError", "parse_pdf"]
