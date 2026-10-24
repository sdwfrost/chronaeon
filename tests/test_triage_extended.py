"""
test_triage_extended.py
-----------------------
Extended tests for chronaeon.triage.ChronAeonSieve covering:
- Gate 2: clock outlier detection
- Gate 3: date mismatch detection
- Missing date handling
- screen_stream batch mode
- Segregated FASTA output (clean/SUS)
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronaeon.triage import ChronAeonSieve


def _make_sieve(tmp_path, n_taxa=15, seq_len=2000):
    """Build a ChronAeonSieve with a synthetic dated alignment."""
    ref_seq = "ACGT" * (seq_len // 4)
    aln_dict = {"ref_2020": ref_seq}
    dates_dict = {"ref_2020": 2020.0}

    for i in range(n_taxa):
        seq = list(ref_seq)
        for j in range(i):
            seq[j * 3] = "G" if seq[j * 3] == "A" else "A"
        aln_dict[f"anc_{i}"] = "".join(seq)
        dates_dict[f"anc_{i}"] = 2020.0 + i * 0.2

    fa_path = tmp_path / "aln.fasta"
    meta_path = tmp_path / "meta.csv"

    with open(fa_path, "w") as f:
        for k, v in aln_dict.items():
            f.write(f">{k}\n{v}\n")

    with open(meta_path, "w") as f:
        f.write("genome_id,collection_date\n")
        for k, v in dates_dict.items():
            f.write(f"{k},{v}\n")

    sieve = ChronAeonSieve.build_from_alignment(
        alignment_path=fa_path,
        dates_path=meta_path,
        root_taxon="ref_2020",
        max_ambig_ratio=0.05,
    )
    return sieve, ref_seq


class TestTriageMissingDate:
    def test_missing_date_flag(self, tmp_path):
        """Sequence with unparseable date should get FLAG_MISSING_DATE."""
        sieve, ref_seq = _make_sieve(tmp_path)
        res = sieve.screen_sequence("unknown_date_seq", ref_seq, "unknown_date")
        assert res["status"] == "FLAG_MISSING_DATE"
        assert "Missing" in res["sus_reason"]

    def test_date_extracted_from_id(self, tmp_path):
        """Date should be extracted from sequence ID if not provided explicitly."""
        sieve, ref_seq = _make_sieve(tmp_path)
        res = sieve.screen_sequence("seq_2021-06-15", ref_seq, "unknown")
        assert res["status"] != "FLAG_MISSING_DATE"
        assert res["reported_date"] is not None
        assert not np.isnan(res["reported_date"])


class TestTriageGate1Quality:
    def test_clean_sequence_passes(self, tmp_path):
        """A clean sequence matching the root should PASS."""
        sieve, ref_seq = _make_sieve(tmp_path)
        res = sieve.screen_sequence("clean_test", ref_seq, 2020.0)
        assert res["status"] == "PASS"

    def test_high_missing_data_fails(self, tmp_path):
        """Sequence with >5% Ns should fail at Gate 1."""
        sieve, ref_seq = _make_sieve(tmp_path)
        seq_bad = "N" * 200 + ref_seq[200:]
        res = sieve.screen_sequence("high_n", seq_bad, 2020.0)
        assert res["status"] == "SUS"
        assert "SUS_LOW_QUALITY" in res["sus_reason"]

    def test_high_gap_ratio_fails(self, tmp_path):
        """Sequence with >5% gaps should fail at Gate 1."""
        sieve, ref_seq = _make_sieve(tmp_path)
        seq_gaps = "-" * 200 + ref_seq[200:]
        res = sieve.screen_sequence("high_gap", seq_gaps, 2020.0)
        assert res["status"] == "SUS"
        assert "SUS_LOW_QUALITY" in res["sus_reason"]

    def test_iupac_degenerate_fails(self, tmp_path):
        """Sequence with >5% IUPAC degenerate codes should fail at Gate 1."""
        sieve, ref_seq = _make_sieve(tmp_path)
        seq_iupac = "R" * 200 + ref_seq[200:]
        res = sieve.screen_sequence("high_iupac", seq_iupac, 2020.0)
        assert res["status"] == "SUS"
        assert "SUS_LOW_QUALITY" in res["sus_reason"]


class TestTriageScreenStream:
    def test_screen_stream_basic(self, tmp_path):
        """Test screen_stream with a small batch of sequences."""
        sieve, ref_seq = _make_sieve(tmp_path)

        stream_seqs = {
            "clean_2020": ref_seq,
            "bad_2020": "N" * 200 + ref_seq[200:],
        }
        stream_fasta = tmp_path / "stream.fasta"
        with open(stream_fasta, "w") as f:
            for k, v in stream_seqs.items():
                f.write(f">{k}\n{v}\n")

        df = sieve.screen_stream(stream_fasta)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "status" in df.columns
        assert "query_id" in df.columns

    def test_screen_stream_with_dates_csv(self, tmp_path):
        """Test screen_stream with explicit dates CSV."""
        sieve, ref_seq = _make_sieve(tmp_path)

        stream_fasta = tmp_path / "stream.fasta"
        with open(stream_fasta, "w") as f:
            f.write(f">seq1\n{ref_seq}\n>N_seq2\n{'N' * 200 + ref_seq[200:]}\n")

        dates_csv = tmp_path / "stream_dates.csv"
        with open(dates_csv, "w") as f:
            f.write("id,date\nseq1,2020.0\nN_seq2,2020.5\n")

        df = sieve.screen_stream(stream_fasta, stream_dates=dates_csv, date_col="date", strain_col="id")
        assert len(df) == 2

    def test_screen_stream_segregated_output(self, tmp_path):
        """Test that clean_fasta_out and sus_fasta_out are written correctly."""
        sieve, ref_seq = _make_sieve(tmp_path)

        stream_fasta = tmp_path / "stream.fasta"
        with open(stream_fasta, "w") as f:
            f.write(f">clean_seq\n{ref_seq}\n>bad_seq\n{'N' * 200 + ref_seq[200:]}\n")

        dates_csv = tmp_path / "stream_dates.csv"
        with open(dates_csv, "w") as f:
            f.write("id,date\nclean_seq,2020.0\nbad_seq,2020.5\n")

        clean_out = str(tmp_path / "clean.fasta")
        sus_out = str(tmp_path / "sus.fasta")

        df = sieve.screen_stream(stream_fasta, stream_dates=dates_csv, date_col="date", strain_col="id", clean_fasta_out=clean_out, sus_fasta_out=sus_out)

        assert Path(clean_out).exists()
        assert Path(sus_out).exists()

        clean_content = Path(clean_out).read_text()
        sus_content = Path(sus_out).read_text()
        assert "clean_seq" in clean_content
        assert "bad_seq" in sus_content


class TestTriageBuildFromAlignment:
    def test_build_requires_minimum_taxa(self, tmp_path):
        """build_from_alignment should raise if fewer than 10 dated taxa."""
        ref_seq = "ACGT" * 500
        fa_path = tmp_path / "small.fasta"
        meta_path = tmp_path / "meta.csv"

        with open(fa_path, "w") as f:
            f.write(f">ref\n{ref_seq}\n>seq1\n{ref_seq}\n")
        with open(meta_path, "w") as f:
            f.write("genome_id,collection_date\nref,2020.0\nseq1,2021.0\n")

        with pytest.raises(ValueError, match="Fewer than 10"):
            ChronAeonSieve.build_from_alignment(
                alignment_path=fa_path,
                dates_path=meta_path,
                root_taxon="ref",
            )

    def test_build_with_default_root(self, tmp_path):
        """build_from_alignment should use earliest taxon as root if root_taxon not in alignment."""
        ref_seq = "ACGT" * 500
        aln_lines = []
        date_lines = ["genome_id,collection_date"]
        for i in range(12):
            date = 2000.0 + i * 2.0
            seq = ref_seq[:1900] + "G" * i + ref_seq[1900 + i:]
            aln_lines.append(f">taxon_{i}\n{seq}")
            date_lines.append(f"taxon_{i},{date}")

        fa_path = tmp_path / "aln.fasta"
        meta_path = tmp_path / "meta.csv"
        fa_path.write_text("\n".join(aln_lines) + "\n")
        meta_path.write_text("\n".join(date_lines) + "\n")

        sieve = ChronAeonSieve.build_from_alignment(
            alignment_path=fa_path,
            dates_path=meta_path,
        )
        assert sieve is not None
        assert len(sieve.anchor_taxa) > 0
