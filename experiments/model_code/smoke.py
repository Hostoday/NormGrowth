"""Check component construction and source syntax without model weights."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from diagnostics.run_llama_counterfact_parallel_control import construct

    # Include feasible and infeasible norm matches, and negative projections.
    h0 = np.tile([1., 0., 0.], (4, 1))
    h = np.asarray([[2., .2, .3], [-2., .2, .3], [.1, 3., 4.], [1., 0., 0.]])
    checks = []
    for dose in (.25, .5):
        result = construct(h, h0, dose)
        feasible = result['feasible']
        assert feasible.tolist() == [True, True, False, True]
        assert np.allclose(result['B'][feasible, 1:], h[feasible, 1:])
        assert np.allclose(np.linalg.norm(result['B'][feasible], axis=-1),
                           result['desired_norm'][feasible])
        assert np.array_equal(result['B'][~feasible], h[~feasible])
        assert np.array_equal(np.sign(result['B'][feasible, 0]), np.sign(h[feasible, 0]))
        checks.append({'dose': dose, 'feasible_cases': int(feasible.sum()),
                       'infeasible_cases': int((~feasible).sum())})
    compiled = 0
    for subdir in ('EasyEdit', 'diagnostics', 'evaluate', 'model_code'):
        for path in (ROOT / subdir).rglob('*.py'):
            compile(path.read_text(), str(path), 'exec')
            compiled += 1
    print(json.dumps({'passed': True, 'compiled_files': compiled,
                      'component_checks': checks, 'model_inference': False}))


if __name__ == '__main__':
    main()
