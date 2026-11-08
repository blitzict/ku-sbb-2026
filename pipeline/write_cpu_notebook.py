"""Write the Colab notebook: load, visualize, transform, train/combine, save."""
import argparse
import json
from pathlib import Path
from pipeline.run_stages import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path)
    args = parser.parse_args()
    measured = json.loads(args.validation.read_text()) if args.validation else None
    status = (f"Measured local CPU AUC: **{measured['local_auc']:.8f}**."
              if measured else '**Full CPU score unverified.** Historical local target: 0.87910.')
    cells = []

    def markdown(text):
        cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True)))

    def code(text):
        compile(text, 'notebook cell', 'exec')
        cells.append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=text.splitlines(keepends=True)))

    markdown('# Train the recorded competition ensemble on CPU\n\n'
        '**Load data → visualize → apply existing transformations → train and combine → save.**\n\n'
        'Use the same four raw competition files and the formulas, selected feature lists, and model settings already recorded in this repository. '
        'No feature search, Optuna search, or model redesign runs here.\n\n' + status + '\n\n'
        'Select a Python 3.12+ runtime with hardware accelerator **None**. Training is a long job; '
        'the complete runtime and 8 GB memory limit have not been verified. '
        'Keep the work folder on persistent storage. Rerunning resumes completed steps with matching inputs and settings.')
    markdown('## Setup\n\nOpen this notebook from `blitzict/ku-sbb-2026` in Colab. '
        'For the private repository, add a Colab secret named `GITHUB_TOKEN` with read access, or enter it privately when prompted. '
        'This cell only installs code and dependencies; training starts in cells 3 and 4 below.')
    code('''from pathlib import Path
import os, sys, subprocess, json, base64, getpass

assert sys.version_info >= (3, 12), "Python 3.12 or newer is required."
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
CODE = Path('/content/AI') if IN_COLAB else Path.cwd()
RELEASE = 'cpu-source-v1'  # Source release; full CPU score is still unverified.
if not (CODE / 'train_087910_cpu.py').is_file():
    try:
        from google.colab import userdata
        token = userdata.get('GITHUB_TOKEN')
    except Exception:
        token = getpass.getpass('GitHub token with read access to blitzict/ku-sbb-2026: ')
    if not token:
        raise ValueError('A GitHub token is required for the private repository.')
    clone_env = os.environ.copy()
    clone_env.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
        GIT_CONFIG_VALUE_0='AUTHORIZATION: basic ' + base64.b64encode(('x-access-token:' + token).encode()).decode())
    try:
        subprocess.run(['git', 'clone', '--branch', RELEASE, '--depth', '1',
                        'https://github.com/blitzict/ku-sbb-2026.git', str(CODE)], env=clone_env, check=True)
    finally:
        del token, clone_env
os.chdir(CODE)
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
if IN_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.11.0',
                    '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-cpu-full.txt'], check=True)

# Every training subprocess receives CPU-only settings.
cpu_env = os.environ.copy()
cpu_env.update(CUDA_VISIBLE_DEVICES='', ONEAPI_DEVICE_SELECTOR='*:cpu')
''')
    markdown('## Settings\n\nOnly change DATA and WORK to use the recorded recipe. '
        'The empty override dictionaries keep all historical features and hyperparameters. '
        'If you deliberately change the recipe, choose a new WORK folder. '
        'The full settings reference is `pipeline/hyperparameters.template.json`.')
    code('''if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA = Path('/content/drive/MyDrive/AI-competition/data')
    WORK = Path('/content/drive/MyDrive/AI-competition/cpu_run')
else:
    DATA = CODE / 'data'
    WORK = CODE / 'cpu_run'

WORKERS = 2
FEATURE_SELECTIONS = {}  # Example: {'tune_out/selected_ns3.csv': {'remove': ['r_emb2_mmd'], 'add': []}}
TREE_SETTINGS = {}       # Example: {'iters': 487}; {} keeps the saved tree parameters.
MODEL_SETTINGS = {}      # Example: {'experiments/v6/nn_var.py': {'--epochs': 20}}
MEMBER_SETTINGS = {}     # Overrides for a specific member's stage ID, if wanted.

config = dict(workers=None, feature_selections=FEATURE_SELECTIONS,
              shared_settings={'tune_out/best_xgb5.json': TREE_SETTINGS} if TREE_SETTINGS else {},
              script_options=MODEL_SETTINGS, stage_options=MEMBER_SETTINGS)
CONFIG = CODE / 'notebook_settings.json'
CONFIG.write_text(json.dumps(config, indent=2))
from pipeline.training_config import plan
recipe = plan(CONFIG, CODE / 'pipeline/full_cpu_stages.json')
print('Using the unchanged historical recipe:', recipe.default)
print('Feature lists and formulas are already supplied; no feature selection search runs.')
reference = json.loads((CODE / 'pipeline/hyperparameters.template.json').read_text())
print('Recorded tree defaults:', reference['_best_xgb5_defaults'])
print('Recorded feature lists (number of columns):', reference['_feature_list_sizes'])
command = [sys.executable, '-u', str(CODE / 'train_087910_cpu.py'),
           '--data', str(DATA), '--work', str(WORK), '--workers', str(WORKERS), '--config', str(CONFIG)]
''')
    markdown('## 1. Load the data\n\nLoad the labels and a small preview. '
        'The Parquet readers leave the complete raw series on disk until the transformation step needs them.')
    code('''import pandas as pd
import pyarrow.parquet as pq

for name in ('train_series.parquet', 'test_series.parquet', 'train_labels.csv', 'sample_submission.csv'):
    assert (DATA / name).is_file(), f'Missing raw file: {DATA / name}'
labels = pd.read_csv(DATA / 'train_labels.csv', dtype={'id': str}).set_index('id')
sample = pd.read_csv(DATA / 'sample_submission.csv', dtype={'id': str})
train_data = pq.ParquetFile(DATA / 'train_series.parquet')
test_data = pq.ParquetFile(DATA / 'test_series.parquet')
preview = next(train_data.iter_batches(batch_size=8192, columns=['id', 'time', 'period', 'value'])).to_pandas()
print(f'{len(labels):,} training series; {len(sample):,} test series')
print(f'{train_data.metadata.num_rows:,} training points; {test_data.metadata.num_rows:,} test points')
display(labels.head())
''')
    markdown('## 2. Visualize the data\n\n'
        'Four example series, with the supplied before/after boundary marked. This display does not change the training data or recipe.')
    code('''import matplotlib.pyplot as plt

examples = list(preview.groupby('id', sort=False, observed=True))[:4]
fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
for ax, (ident, series) in zip(axes.flat, examples):
    series = series.sort_values('time')
    boundary = series.loc[series['period'] == 1, 'time'].min()
    ax.plot(series['time'], series['value'], linewidth=0.8)
    ax.axvline(boundary, color='tomato', linestyle='--', label='Before / after')
    ax.set_title(f'{str(ident)[:12]} | break label: {labels.loc[str(ident), "label"]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
plt.show()
''')
    markdown('## 3. Apply the existing transformations\n\n'
        'Apply the recorded statistical formulas, distribution and time-dependence tests, physical signal transforms, '
        'and boundary-preserving views. The feature lists already specify which columns each model uses. '
        'Where the recipe learns a transformation from labels, it retains the original training-fold rules. '
        'Later model-specific transforms remain with their original model fits to preserve dependency order.')
    code('''subprocess.run([*command, '--phase', 'features'], env=cpu_env, check=True)
print('Recorded transformations are saved in:', WORK)
''')
    markdown('## 4. Train and combine the specified models\n\n'
        'Fit the recorded trees, neural models, and specialists; then fit the recorded ensemble weights and corrections. '
        'This uses the fixed recipe and your settings above. Checkpoints are saved during fitting. '
        'The final local AUC is measured from the new CPU predictions, not copied from the historical result.')
    code('''subprocess.run([*command, '--phase', 'models'], env=cpu_env, check=True)
''')
    markdown('## 5. Save the model and read the result\n\n'
        'The fitted ensemble is saved in WORK during training. Keep **the whole WORK folder**, including its weights and preprocessing files; '
        '`model.json` is its description, not the weights themselves. '
        'When WORK is on Drive, the model persists after Colab disconnects. '
        'The competition CSV is saved alongside it. The separate new-series prediction interface is deferred.')
    code('''model = json.loads((WORK / 'model.json').read_text())
report = json.loads((WORK / model['validation']).read_text())
SUBMISSION = WORK / model['submission']
print('Saved model folder:', WORK)
print('Saved model description:', WORK / 'model.json')
print(f'Measured local CPU AUC: {report["local_auc"]:.8f}')
print('Competition CSV:', SUBMISSION)
display(pd.read_csv(SUBMISSION, dtype={'id': str}).head())
# Optional download after training:
# from google.colab import files
# files.download(str(SUBMISSION))
''')
    markdown('## AI-use account\n\nThe user will add the combined account across all sessions here.')
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
            language_info=dict(name='python', version='3.12'), colab=dict(name='CPU_087910.ipynb', provenance=[]),
            cpu_rebuild=dict(target_local_auc=.87910,
                measured_local_auc=measured['local_auc'] if measured else None,
                validation_status='measured' if measured else 'pending',
                colab_execution_verified=False, source_release='cpu-source-v1')))
    for i, cell in enumerate(cells):
        cell['id'] = f'cpu-cell-{i:02d}'
    path = ROOT / 'CPU_087910.ipynb'
    path.write_text(json.dumps(notebook, indent=2), encoding='utf-8')
    print(f'Wrote {path}: {len(cells)} cells, score {notebook["metadata"]["cpu_rebuild"]["validation_status"]}')


if __name__ == '__main__':
    main()
