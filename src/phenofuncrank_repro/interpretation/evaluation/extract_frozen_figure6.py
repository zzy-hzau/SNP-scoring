"""Read published Figure6 scoring tables; no workbook or model mutations."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REVIEW = 'NEEDS_HUMAN_REVIEW'


def read_tables(path, sheet_name):
    """Read raw values and structured-table ranges using Python's standard library."""
    with zipfile.ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(t.text or '' for t in s.findall('.//m:t', NS)) for s in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        workbook = ET.fromstring(z.read('xl/workbook.xml'))
        sheet = next(s for s in workbook.find('m:sheets', NS) if s.attrib['name'] == sheet_name)
        rels = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        target = rels[sheet.attrib[f'{{{REL}}}id']]
        sheet_path = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
        document = ET.fromstring(z.read(sheet_path))
        values = {}
        for c in document.findall('.//m:sheetData/m:row/m:c', NS):
            v = c.find('m:v', NS)
            kind = c.attrib.get('t')
            if kind == 'inlineStr': value = ''.join(t.text or '' for t in c.findall('.//m:t', NS))
            elif v is None: continue
            elif kind == 's': value = strings[int(v.text)]
            elif kind in {'str', 'e'}: value = v.text
            elif kind == 'b': value = bool(int(v.text))
            else: value = float(v.text)
            values[c.attrib['r']] = value
        relationship_path = posixpath.dirname(sheet_path) + '/_rels/' + posixpath.basename(sheet_path) + '.rels'
        rels = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(z.read(relationship_path))}
        tables = {}
        for ref in document.findall('m:tableParts/m:tablePart', NS):
            target = rels[ref.attrib[f'{{{REL}}}id']]
            target = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.dirname(sheet_path) + '/' + target)
            table = ET.fromstring(z.read(target))
            c0, r0, c1, r1 = re.fullmatch(r'([A-Z]+)(\d+):([A-Z]+)(\d+)', table.attrib['ref']).groups()
            def number(col):
                result = 0
                for char in col: result = result * 26 + ord(char) - 64
                return result
            def letters(col):
                result = ''
                while col:
                    col, mod = divmod(col - 1, 26); result = chr(65 + mod) + result
                return result
            columns = [letters(i) for i in range(number(c0), number(c1) + 1)]
            headers = [c.attrib['name'] for c in table.find('m:tableColumns', NS)]
            rows = []
            for r in range(int(r0) + 1, int(r1) + 1):
                row = {header: values.get(f'{col}{r}') for header, col in zip(headers, columns)}
                if any(v is not None for v in row.values()):
                    row['_source_row'] = r; rows.append(row)
            tables[table.attrib['displayName']] = rows
        return tables


REASON_FIELDS = {'Evidence grounding': 'grounding', 'Missing-evidence detection': 'missing_link_detection', 'Conclusion calibration': 'conclusion_calibration', 'Unsupported evidence leap': 'evidence_leap', 'Causal overclaim': 'unsupported_causal_upgrade', 'Evidence-ID error': 'evidence_id_error'}
CRITIC_FIELDS = {'Adjudicated reference state': 'reference_state', 'Adjudicated reference action': 'reference_action', 'Action appropriate': 'action_appropriate', 'Semantic error detected': 'semantic_error_detected', 'Supported claim retained': 'supported_retained', 'Partial support calibrated': 'partial_calibrated', 'Contradiction detected': 'contradiction_detected', 'Evidence gap useful': 'gap_useful'}
REFINEMENT_FIELDS = {'Conclusion correct': 'conclusion_correct', 'Causal calibration': 'causal_calibration', 'Evidence grounding': 'evidence_grounding', 'Missing-link handling': 'missing_link_handling', 'Maximum defensible conclusion': 'max_defensible_conclusion', 'Causal overclaim': 'unsupported_causal_upgrade'}
CONDITIONS = {'Unconstrained reasoning': 'vanilla', 'Evidence-grounded reasoning': 'full', 'Rule-based critic': 'M1', 'Evidence-aware critic': 'M2', 'Initial output': 't0', 'Refined output': 't1'}


def rows_to_scores(rows, fields, panel, judged=False):
    result = []
    for row in rows:
        item = {'case_id': row['Case ID'], 'condition': CONDITIONS[row['Stage'] if panel == 'refinement' else row['Condition']], 'scores': {field: row.get(header) or REVIEW for header, field in fields.items()}, '_source_row': row['_source_row']}
        if panel == 'reasoning': item['generation'] = int(row['Generation'])
        if judged: item['judge_pass'] = int(row['Judge pass'])
        if 'Included in paired analysis' in row:
            item['included_in_paired_analysis'] = bool(row['Included in paired analysis'])
            item['exclusion_reason'] = row.get('Exclusion reason')
        result.append(item)
    return result


def extract(source_data, supplementary_data):
    source = read_tables(source_data, 'Figure 6')
    critic = read_tables(supplementary_data, 'Critic benchmark')['BenchmarkCleanSupp_3']
    definitions = [{'case_id': row['Case ID'], 'analysis_subset': row['Analysis subset'], 'category': row['Challenge category']} for row in critic]
    return {
        'schema': 'published_figure6_scoring_v1',
        'source_artifacts': {'Source_Data.xlsx': hashlib.sha256(Path(source_data).read_bytes()).hexdigest(), 'Supplementary_Data_1.xlsx': hashlib.sha256(Path(supplementary_data).read_bytes()).hexdigest()},
        'critic_definitions': definitions,
        'reasoning_consensus': rows_to_scores(source['BenchmarkCleanSource_3'], REASON_FIELDS, 'reasoning'),
        'reasoning_judges': rows_to_scores(source['BenchmarkCleanSource_4'], REASON_FIELDS, 'reasoning', True),
        'critic_consensus': rows_to_scores(source['BenchmarkCleanSource_5'], CRITIC_FIELDS, 'critic'),
        'critic_judges': rows_to_scores(source['BenchmarkCleanSource_6'], CRITIC_FIELDS, 'critic', True),
        'refinement_consensus': rows_to_scores(source['BenchmarkCleanSource_7'], REFINEMENT_FIELDS, 'refinement'),
        'refinement_judges': rows_to_scores(source['BenchmarkCleanSource_8'], REFINEMENT_FIELDS, 'refinement', True),
        'scope': 'Published scoring labels only. Responses, judge rationales, provider traces and private blind mappings are not exported. Originals are unchanged.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data', type=Path, required=True)
    parser.add_argument('--supplementary-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = extract(args.source_data, args.supplementary_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
