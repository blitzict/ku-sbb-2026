"""Query-side feature dispatch for the fixed raw-series ensemble.

All mathematical feature functions come from the frozen CPU sources. Only
the split iteration of reviewed, label-free functions is restricted to the
test split. Learned population priors are loaded by priors.py instead.
This module is called in a fresh subprocess for each stage.
"""
import ast
import importlib
import inspect
import runpy
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

from . import priors, raw_features


def test_function(module, name):
    """Compile one reviewed feature function with ('train','test') -> ('test',).

    No script main, training routine, or other function is executed here.
    Lists and tuples are handled because both occur in the frozen sources.
    """
    source = ast.parse(textwrap.dedent(inspect.getsource(getattr(module, name))))

    class QuerySplit(ast.NodeTransformer):
        def visit_Tuple(self, node):
            return self._parts(node)

        def visit_List(self, node):
            return self._parts(node)

        def _parts(self, node):
            if len(node.elts) == 2 and all(isinstance(x, ast.Constant) for x in node.elts):
                if [x.value for x in node.elts] == ['train', 'test']:
                    node.elts = [ast.Constant('test')]
            return self.generic_visit(node)

    namespace = dict(module.__dict__)
    exec(compile(ast.fix_missing_locations(QuerySplit().visit(source)), module.__file__, 'exec'), namespace)
    return namespace[name]


def _run_script(stage, work):
    # Reviewed feature-only scripts. They receive only a test argument.
    sys.argv = list(stage['argv'])
    runpy.run_path(str(Path(work) / sys.argv[0]), run_name='__main__')


DIRECT = {
    'features_grid.py', 'features_grid2.py', 'features_dd.py', 'features_lr.py',
    'features_lrz.py', 'features_garch.py', 'features_lm.py', 'features_bf.py',
    'features_ps.py', 'features_sn.py', 'stream_numpy_features.py', 'harm_var.py',
    'late_features.py', 'component_moves_cpu.py', 'white_model_features.py', 'copy_table.py',
}
SPECIAL = {
    'generator_features.py', 'mechanism.py', 'white_outer_priors.py',
    'white_view_pipeline.py', 'white_view_states.py', 'white_view_extra.py',
    'white_crop_partition.py', 'white_population_transport.py', 'white_transition_priors.py',
    'white_filtered_steps.py', 'white_markov_copula.py', 'trend_clock.py', 'state_tests.py',
    'early_feature_extract.py',
}
TRAIN_ONLY = {'shift_augmentation.py', 'aug_grid.py', 'harm_aug.py', 'make_halves.py'}
VARIANTS = {'make_variants.py', 'make_crops.py'}


def mode(stage):
    """Return query, reference, variants, or None for a model stage."""
    name = Path(stage['argv'][0]).name
    if name in TRAIN_ONLY:
        return 'reference'
    if name in VARIANTS:
        return 'variants'
    if name == 'shape_bank.py' and stage['argv'][1:] == ['prepare']:
        return 'query'
    if name == 'train_var.py' and '--export' in stage['argv']:
        return 'query'
    if name in DIRECT:
        return 'query' if any('test' in x for x in stage['outputs']) else 'reference'
    if name in SPECIAL:
        return 'reference' if '--part' in stage['argv'] and stage['argv'][stage['argv'].index('--part') + 1] == 'train' else 'query'
    return None


def _view(work, model_root, view, kind):
    import white_view_pipeline as original
    folder = Path(work) / 'tune_out/weight_space' / f'view_{view}'
    folder.mkdir(parents=True, exist_ok=True)
    if kind == 'base':
        import bounded_prior
        original.materialize(folder, view, 'test')
        modules = ('periodic_components', 'innovation_transform', 'trend_components',
                   'curve_likelihood', 'ar_likelihood', 'phase_likelihood', 'mixture_law',
                   'partition_evidence', 'pulse_filter')
        for name in modules:
            module = importlib.import_module(name)
            with patch.multiple(module, ROOT=folder, OUT=folder):
                module.run('test')
        with patch.object(bounded_prior, 'main', test_function(bounded_prior, 'main')):
            test_function(original, 'bounds')(folder)
        transfer = importlib.import_module('transfer_member')
        with patch.multiple(transfer, ROOT=folder, OUT=folder):
            test_function(transfer, 'prepare')()
        laws = importlib.import_module('white_laws')
        with patch.multiple(laws, ROOT=folder, OUT=folder):
            laws.extract('test', 'physical', laws.dictionary())
        priors.outer(model_root, work, view=view)
        (folder / 'READY').write_text('Query mechanisms built using saved training priors.')
    elif kind == 'states':
        module = importlib.import_module('white_view_states')
        test_function(module, 'build')(view)
    elif kind == 'extra':
        module = importlib.import_module('white_view_extra')
        test_function(module, 'run')(view)


def predict_stage(stage, model_root, work):
    name, args = Path(stage['argv'][0]).name, stage['argv'][1:]
    sys.argv = ['query-features']  # Old v6 imports inspect argv.
    if mode(stage) in {'reference', 'variants'}:
        return
    if name in DIRECT or name == 'train_var.py':
        _run_script(stage, work)
    elif name == 'generator_features.py':
        raw_features.generator(model_root, work)
    elif name == 'mechanism.py':
        if args[0] in {'noise_tests', 'bounds'}:
            module = importlib.import_module('transfer_member' if args[0] == 'noise_tests' else 'bounded_prior')
            test_function(module, 'prepare' if args[0] == 'noise_tests' else 'main')()
        else:
            _run_script(stage, work)
    elif name == 'white_outer_priors.py':
        for seed in args[args.index('--seeds') + 1:]:
            priors.outer(model_root, work, int(seed))
    elif name in {'white_view_pipeline.py', 'white_view_states.py', 'white_view_extra.py'}:
        kind = {'white_view_pipeline.py': 'base', 'white_view_states.py': 'states', 'white_view_extra.py': 'extra'}[name]
        for view in args:
            _view(work, model_root, view, kind)
    elif name == 'white_crop_partition.py':
        for view in args[1:]:
            priors.outer(model_root, work, int(args[0]), view=view)
    elif name == 'white_population_transport.py':
        priors.transport(model_root, work, args[0])
    elif name == 'white_transition_priors.py':
        priors.transitions(model_root, work)
    elif name in {'white_filtered_steps.py', 'white_markov_copula.py'}:
        importlib.import_module(Path(name).stem).extract('test')
    elif name in {'trend_clock.py', 'state_tests.py'}:
        importlib.import_module(Path(name).stem).run('test')
    elif name == 'shape_bank.py':
        test_function(importlib.import_module('shape_bank'), 'prepare')()
    elif name == 'early_feature_extract.py':
        module, function, extra = {
            'shape_counts': ('white_shapes', 'data', []),
            'deconv_pulse_measure': ('white_deconv', 'measurements', ['pulse']),
            'conditional_scores': ('white_influence', 'extract', []),
            'trend_population': ('white_trend_prior', 'extract', []),
        }[args[0]]
        getattr(importlib.import_module(module), function)('test', *extra)
    else:
        raise NotImplementedError(stage['id'])
