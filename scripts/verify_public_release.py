"""Recompute public evidence without weakening the private freeze gate."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from analyze_human_annotation_workbooks import kappa
from finalize_human_semantic_analysis import (
    FIELDS, split_conditions, validate_conditions, primary_summary, holm_adjusted,
    repair_summary, utility_summary, boundary_mechanism_summary,
)


def same(actual, expected, where='result'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError('Dictionary mismatch at ' + where)
        for key in expected:
            same(actual[key], expected[key], where + '.' + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('List mismatch at ' + where)
        for index, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, where + '.' + str(index))
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('Numeric mismatch at ' + where)
    elif actual != expected:
        raise ValueError('Value mismatch at ' + where)


def verify_response_bindings(load, records):
    expected = {}
    main = load('artifacts/full_pipeline_cert_m5_qwen2p5_7b_deterministic_96_v5.json')['records']
    control = {row['incident_id']: row for row in load('artifacts/full_pipeline_cert_m5_all_owner_control_qwen2p5_7b_deterministic_96_a100_v6.json')['records']}
    for row in main:
        incident = row['incident_id']
        for condition, key in {'initial': 'initial', 'f0': 'f0_trace_mask', 'partial_f2': 'f2_fresh_exact_rebuild'}.items():
            expected[incident + '|' + condition] = row[key]['answer']
        expected[incident + '|complete_f2'] = control[incident]['f2_fresh_exact_rebuild']['answer']
    for row in load('artifacts/boundary_migration_cert_50_qwen2p5_7b_96_v3_control.json')['records']:
        for condition, key in {'initial': 'initial', 'f0': 'f0_trace_mask', 'rebuilt': 'f2_rechunk_rebuild', 'complete_lineage': 'complete_lineage_quarantine'}.items():
            expected['boundary|' + row['incident_id'] + '|' + condition] = row[key]['answer']
    reference = load('artifacts/repair_lattice_cert_30_qwen2p5_7b_96_v1.json')['records']
    for row in reference:
        prefix = 'repair|' + row['incident_id']
        expected[prefix + '|initial'] = row['initial']['answer']
        for action, details in row['actions'].items():
            expected[prefix + '|' + action + '|f0'] = details['f0_trace']['answer']
            expected[prefix + '|' + action + '|f2'] = details['f2_physical_rebuild']['answer']
    same({row['item_id']: row['response'] for row in records}, expected, '846 stimulus bindings')
    certificates = load('artifacts/repair_lattice_independent_cert_30_qwen2p5_7b_96_v2.json')
    reference_by_id = {row['incident_id']: row for row in reference}
    checked_actions = 0
    checked_answers = 0
    for row in certificates['records']:
        old = reference_by_id[row['incident_id']]
        same(row['initial']['answer'], old['initial']['answer'], 'certificate initial answer')
        checked_answers += 1
        same(set(row['actions']), set(old['actions']), 'certificate action names')
        for action, details in row['actions'].items():
            for phase in ('f0_trace', 'f2_physical_rebuild'):
                same(details[phase]['answer'], old['actions'][action][phase]['answer'], '270 reference answers')
                checked_answers += 1
            same(details['edit_tokens'], old['actions'][action]['edit_tokens'], 'certificate edit cost')
            same(details['f1_source_revision_sha256'], details['f2_source_revision_sha256'], 'source revision')
            for key in ('answer', 'answer_sha256', 'dense_ids', 'served_ids', 'prompt_sha256',
                        'metadata_sha256', 'source_vector_sha256', 'combined_vector_sha256'):
                same(details['f1_exact_view'][key], details['f2_physical_rebuild'][key], '120 F1/F2 ' + key)
            for phase in ('f0_trace', 'f1_exact_view', 'f2_physical_rebuild'):
                value = details[phase]
                same(hashlib.sha256(value['answer'].encode('utf-8')).hexdigest(), value['answer_sha256'], 'certificate answer digest')
            specification = json.dumps(details['action_spec'], sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            same(hashlib.sha256(specification.encode('utf-8')).hexdigest(), details['action_spec_sha256'], 'action digest')
            for key, value in details['certificate'].items():
                same(value, 0.0 if key == 'source_vectors_max_abs_error' else True, 'stored certificate check ' + key)
            checked_actions += 1
    same(checked_actions, 120)
    same(checked_answers, 270)
    return checked_actions, checked_answers


def main():
    root = Path(__file__).resolve().parents[1]
    load = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    manifest = load('PUBLIC_MANIFEST.json')
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Public file digest mismatch: ' + name)
    records = load('artifacts/human_labels_public.json')['records']
    if len(records) != 846 or len({row['item_id'] for row in records}) != 846:
        raise ValueError('Expected all 846 unique outputs')
    for row in records:
        if hashlib.sha256(row['response'].encode('utf-8')).hexdigest() != row['response_sha256']:
            raise ValueError('Response hash mismatch')
        if set(row['final_labels']) != set(FIELDS) or not all(type(v) is bool for v in row['final_labels'].values()):
            raise ValueError('Malformed final labels')
    checked_actions, checked_answers = verify_response_bindings(load, records)
    stats = load('artifacts/paper_statistics_human_final_v1.json')
    same(sum(row['adjudicated'] for row in records), stats['human_annotation']['adjudicated_items'])
    same(sum(not row['final_labels']['complete'] for row in records), stats['human_annotation']['final_incomplete_outputs'])
    for field in FIELDS:
        left = [row['independent_a'][field] for row in records]
        right = [row['independent_b'][field] for row in records]
        expected = stats['human_annotation']['agreement'][field]
        same(kappa(left, right), expected['cohen_kappa_three_class'])
        same(dict(Counter(left)), expected['annotator_a_counts'])
        same(dict(Counter(right)), expected['annotator_b_counts'])
    groups = split_conditions({row['item_id']: row['final_labels'] for row in records})
    validate_conditions(groups[0], {'initial', 'f0', 'partial_f2', 'complete_f2'}, 'substitution', 94)
    validate_conditions(groups[1], {'initial', 'f0', 'rebuilt', 'complete_lineage'}, 'boundary', 50)
    for mode in ('adjudicated', 'least_favorable', 'most_favorable'):
        results = {
            'substitution': primary_summary(groups[0], 'partial_f2', 'complete_f2', 10000, 2027, mode),
            'boundary': primary_summary(groups[1], 'rebuilt', 'complete_lineage', 10000, 2028, mode),
        }
        adjusted = holm_adjusted({key: value['exact_mcnemar_two_sided_p'] for key, value in results.items()})
        for key in results:
            results[key]['holm_adjusted_p'] = adjusted[key]
        expected = stats['primary_hypotheses'] if mode == 'adjudicated' else stats['incomplete_output_sensitivity'][mode]
        same(results, expected, mode)
    repair = repair_summary(groups[2], load('artifacts/repair_lattice_cert_30_qwen2p5_7b_96_v1.json'))
    expected = stats['repair_lattice']
    same(repair, {key: expected[key] for key in repair}, 'repair')
    same(utility_summary(load('artifacts/global_utility_nq150_100k_v1.json'), 10000, 2027), stats['utility'], 'utility')
    same(boundary_mechanism_summary(groups[1], load('artifacts/boundary_migration_cert_50_qwen2p5_7b_96_v3_control.json')),
         stats['boundary_mechanism'], 'boundary mechanism')
    certificate = load('artifacts/repair_lattice_independent_cert_30_qwen2p5_7b_96_v2.json')['summary']
    for key, count in {'actions_certified': 120, 'actions_total': 120, 'reference_answer_matches': 270, 'reference_answer_checks': 270}.items():
        same(certificate[key], count)
    print(json.dumps({'status': 'public_evidence_verified', 'files': len(manifest['files']),
                      'outputs': 846, 'primary_and_sensitivity': 'exactly_recomputed',
                      'repair_utility_mechanism_agreement': 'recomputed',
                      'stimulus_bindings': 846, 'stored_action_consistency_checks': checked_actions,
                      'reference_answer_bindings': checked_answers,
                      'physical_vector_cache_reexecution': 'requires GPU rerun; not claimed offline',
                      'private_annotation_key_gate': 'not_claimed'}))


if __name__ == '__main__':
    main()
