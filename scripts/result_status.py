"""Case outcome accounting shared by ingestion receipts and Pages.

Count distinct case names per category, not backend/thread rows. A case can
occur in more than one category; its original rows remain the source of truth.
"""

FAILED_STATUSES = frozenset({
    'failed', 'compile_failed', 'runtime_failed', 'verification_failed', 'oom', 'not_configured',
})


def case_outcomes(rows):
    cases = {'failed': set(), 'skipped': set(), 'unsupported': set()}
    for row in rows:
        status = row['status']
        if status == 'ok':
            continue
        category = ('unsupported' if status.startswith('unsupported:') else
                    status if status in ('skipped', 'unsupported') else 'failed')
        cases[category].add(row['case'])
    return {kind: {'count': len(names), 'cases': sorted(names)} for kind, names in cases.items()}
