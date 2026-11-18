"""M5 task 2: the report's number check, and that its "Do not" rules bite.

``report/check_frontogenesis_report.py`` reads the JSON summaries in ``data/``
(never a store), so these tests need the products and are skipped without them.
"""
import importlib
import sys
from pathlib import Path

import pytest

REPORT_DIR = Path(__file__).resolve().parents[2] / 'report'
DATA = REPORT_DIR.parent / 'data'
pytestmark = pytest.mark.skipif(
    not (DATA / 'm3_closure_summary.json').exists() or not (DATA / 'm4_strength.json').exists(),
    reason='the JSON summaries in data/ are not present')


@pytest.fixture(scope='module')
def chk():
    sys.path.insert(0, str(REPORT_DIR))
    try:
        yield importlib.import_module('check_frontogenesis_report')
    finally:
        sys.path.remove(str(REPORT_DIR))


def test_the_report_matches_the_summaries(chk, capsys):
    assert chk.main() == 0
    out = capsys.readouterr().out
    assert 'OK' in out and 'FAILED' not in out


@pytest.mark.parametrize('inserted, complaint', [
    ('The slope is 1.342 at the grid scale.', 'FORBIDDEN outside Appendix C'),
    ('Criterion 4 is met.', 'criterion 4 "met" without "pre-registered"'),
    ('The gap is diabatic damping.', '"damping" in a line that does not negate it'),
    ('The hourly rates agree at 0.951 and 1.008.', '0.951 / 1.008 without "not an efficiency"'),
    ('The frontogenesis efficiency is high.', '"efficiency" in a line that does not negate it'),
])
def test_the_do_not_rules_bite(chk, tmp_path, monkeypatch, capsys, inserted, complaint):
    text = chk.REPORT.read_text()
    head = '## 6. The slope, and what it is read against'
    assert head in text
    bad = tmp_path / 'frontogenesis_report.md'
    bad.write_text(text.replace(head, head + '\n\n' + inserted))
    monkeypatch.setattr(chk, 'REPORT', bad)
    monkeypatch.setattr(chk, 'HERE', chk.HERE)          # figures still resolve from report/
    assert chk.main() == 1
    assert complaint in capsys.readouterr().out


def test_a_changed_number_is_caught(chk, tmp_path, monkeypatch, capsys):
    text = chk.REPORT.read_text()
    assert '0.813 / 0.841 / 0.898 / 1.042' in text
    bad = tmp_path / 'frontogenesis_report.md'
    bad.write_text(text.replace('0.898', '0.899'))
    monkeypatch.setattr(chk, 'REPORT', bad)
    assert chk.main() == 1
    assert 'MISSING (body) L=4 residual / measured' in capsys.readouterr().out
