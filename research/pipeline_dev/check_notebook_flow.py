"""Check notebook phase wiring without fitting or extracting real features."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

import train_087910_cpu as trainer
from pipeline.training_config import plan


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / 'pipeline/full_cpu_stages.json'
    config = root / 'pipeline/hyperparameters.json'
    recipe = plan(config, base)
    expected = [s['id'] for s in recipe.manifest['stages']]
    calls = []
    fits = []

    with tempfile.TemporaryDirectory(prefix='notebook-check-') as temporary:
        folder = Path(temporary)
        data = folder / 'data'
        data.mkdir()
        for name in ('train_series.parquet', 'test_series.parquet', 'train_labels.csv', 'sample_submission.csv'):
            (data / name).write_text('phase wiring fixture, not competition data')

        def initialize(work, source):
            (work / 'stages').mkdir(parents=True)
            shutil.copytree(source, work / 'data')
            (work / 'CPU_REBUILD_WORKSPACE.json').write_text(json.dumps({
                'raw_input_sha256': {p.name: trainer.digest(p) for p in source.iterdir()}}))

        def build(work, part, workers, limit):
            calls.append(part)
            (work / f'stages/foundation_{part}.json').write_text('{}')

        def run(work, stage):
            fits.append(stage['id'])
            (work / 'stages' / (stage['id'] + '.json')).write_text(json.dumps({
                'fingerprint': {'argv': stage['argv'], 'env': stage['env']}, 'exit_code': 0}))

        def prepare(work, *args):
            (work / 'TRAINING_CONFIG.json').write_text(json.dumps({
                'config_sha256': recipe.config_sha256, 'base_manifest_sha256': trainer.digest(base)}))
            return recipe

        def validate(work, manifest):
            assert [s for s in expected if not (work / 'stages' / (s + '.json')).is_file()] == []
            # Deliberately synthetic: verifies record plumbing, never a model score.
            return dict(local_auc=.5, target_met_at_reported_precision=False,
                        submission=str(work / 'new weights/example/submission.csv'))

        def execute(work, phase):
            argv = ['train', '--data', str(data), '--work', str(work), '--config', str(config),
                    '--manifest', str(base), '--phase', phase]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                trainer.main()

        with patch.multiple(trainer, initialize=initialize, build=build, run=run,
                            prepare=prepare, validate=validate, sync=lambda work: None):
            new = folder / 'missing-features'
            try:
                execute(new, 'models')
            except ValueError as error:
                assert 'transformations cell' in str(error)
            else:
                raise AssertionError('Model phase must reject missing transformations')
            assert not new.exists()
            work = folder / 'split'
            execute(work, 'features')
            boundary = trainer.feature_boundary(recipe.manifest['stages'])
            assert fits == expected[:boundary] and calls == ['train', 'test']
            assert not (work / 'model.json').exists()
            execute(work, 'models')
            assert fits == expected and calls == ['train', 'test']
            record = json.loads((work / 'model.json').read_text())
            assert record['local_auc'] == .5 and record['device'] == 'cpu'
            assert json.loads((work / record['recipe']).read_text()) == recipe.manifest
            fits.clear()
            calls.clear()
            execute(folder / 'all', 'all')
            assert fits == expected and calls == ['train', 'test']
    notebook = json.loads((root / 'CPU_087910.ipynb').read_text())
    code = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    for source in code:
        compile(source, 'notebook cell', 'exec')
    assert sum("'--phase', 'features'" in source for source in code) == 1
    assert sum("'--phase', 'models'" in source for source in code) == 1
    assert notebook['metadata']['cpu_rebuild']['measured_local_auc'] is None
    print(f'PASS: split phases equal all {len(expected)} ordered recipe steps; no fits ran.')
    print('PASS: missing-feature refusal, model record, and notebook cell syntax.')


if __name__ == '__main__':
    main()
