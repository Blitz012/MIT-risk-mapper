"""Tests for taxonomy loading.

These cover load_taxonomy, which is pure file parsing and needs no model. The
embedding based find_top_risks is exercised by the app and by manual runs; it is
left out here to keep the suite fast and offline.
"""
from __future__ import annotations

import json

import pytest

from nlp_mapper import RiskEntry, load_taxonomy


def _write_taxonomy(tmp_path, data):
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_taxonomy_flattens_domains(tmp_path):
    data = {
        "Privacy & Security": [
            {"subdomain": "Compromise of privacy", "definition": "Sensitive data exposure."},
            {"subdomain": "Security vulnerabilities", "definition": "Exploitable weaknesses."},
        ],
        "Discrimination": [
            {"subdomain": "Unfair discrimination", "definition": "Biased outcomes."},
        ],
    }
    entries = load_taxonomy(_write_taxonomy(tmp_path, data))
    assert len(entries) == 3
    assert all(isinstance(e, RiskEntry) for e in entries)
    assert {e.domain for e in entries} == {"Privacy & Security", "Discrimination"}


def test_load_taxonomy_skips_entries_without_definition(tmp_path):
    data = {
        "Domain": [
            {"subdomain": "Has definition", "definition": "real text"},
            {"subdomain": "No definition", "definition": "   "},
            {"subdomain": "Missing key"},
        ]
    }
    entries = load_taxonomy(_write_taxonomy(tmp_path, data))
    assert len(entries) == 1
    assert entries[0].subdomain == "Has definition"


def test_load_taxonomy_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_taxonomy(tmp_path / "does_not_exist.json")


def test_load_taxonomy_empty_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        load_taxonomy(_write_taxonomy(tmp_path, {"Domain": []}))
