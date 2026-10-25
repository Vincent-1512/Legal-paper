"""Offline integration check with a tiny ELECTRA; never downloads a model."""
import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from transformers import BertTokenizer, ElectraConfig, ElectraModel


def runner_source():
    path = Path(__file__).resolve().parents[1] / 'PAP_NER_ELECTRA_CRF_HierarchicalSupCon_dedup_20epoch_RTX3090_24GB.ipynb'
    notebook = json.loads(path.read_text())
    parts = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == 'RUNNER_PARTS'):
                parts.append(ast.literal_eval(node.value.args[0]))
    return ''.join(parts)


def main():
    with tempfile.TemporaryDirectory(prefix='papner-integration-') as tmp:
        root = Path(tmp)
        model_dir = root / 'model'
        model_dir.mkdir()
        vocab = model_dir / 'vocab.txt'
        vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nco\nquan\nvan\nban\nngay\n')
        BertTokenizer(vocab_file=str(vocab), do_lower_case=False).save_pretrained(model_dir)
        tiny = ElectraConfig(vocab_size=10, embedding_size=16, hidden_size=16,
                             num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
        ElectraModel(tiny).save_pretrained(model_dir)
        data_files = {}
        sentence = 'co N O B-CQ O\nquan N O I-CQ O\nvan N O B-VBPL O\nban N O I-VBPL O\nngay N O O O\n\n'
        for split, count in [('train', 4), ('dev', 3), ('test', 3)]:
            path = root / f'{split}.txt'
            path.write_text(sentence * count)
            data_files[split] = str(path)
        config = dict(model_name_or_path=str(model_dir), use_supcon=True,
                      lambda_supcon=.3, supcon_temp=.07, supcon_alpha=.3,
                      crf_reduction='mean', max_seq_length=32, train_batch_size=2,
                      eval_batch_size=2, gradient_accumulation_steps=1,
                      gradient_checkpointing=False, precision='fp32', learning_rate=2e-5,
                      classifier_learning_rate=1e-3, epochs=2, early_stop=3,
                      warmup_proportion=.1, weight_decay=.01, adam_epsilon=1e-6,
                      max_grad_norm=1., dropout=.1, seed=42, num_workers=0,
                      device='cpu', smoke_batches=2, train_limit=None, dev_limit=None,
                      cpu_threads=1, selection_metric='entity_strict_micro_f1', resume=False,
                      dataset_repo='offline-fixture', dataset_revision='fixture-v1',
                      data_files=data_files, output_dir=str(root / 'run'),
                      cache_dir=str(root / 'cache'))
        runner = root / 'runner.py'
        runner.write_text(runner_source())
        config_path = root / 'config.json'
        environment = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')

        def run(mode, overrides=None, extra=(), expected_failure=False):
            current = dict(config, **(overrides or {}))
            config_path.write_text(json.dumps(current))
            result = subprocess.run([sys.executable, str(runner), mode, '--config', str(config_path), *extra],
                                    env=environment, capture_output=True, text=True, timeout=90)
            if expected_failure:
                assert result.returncode != 0 and 'Cannot resume after changing supcon_alpha' in result.stderr
            elif result.returncode:
                raise AssertionError(result.stdout + result.stderr)

        run('smoke', dict(output_dir=str(root / 'smoke')))
        smoke = json.loads((root / 'smoke' / 'smoke_metrics.json').read_text())
        assert smoke['train_supcon_loss'] > 0
        run('train')
        history = json.loads((root / 'run' / 'history.json').read_text())
        assert len(history) == 2 and all(row['train_supcon_loss'] > 0 for row in history)
        run('test')
        predictions = json.loads((root / 'run' / 'test_predictions.json').read_text())
        assert len(predictions['gold']) == len(predictions['pred']) == 3
        run('predict', extra=('--text', 'co quan van ban ngay'))
        run('train', dict(resume=True))
        run('train', dict(resume=True, supcon_alpha=.4), expected_failure=True)
        print('PASS: offline smoke/train/test/predict/resume; checkpoint mismatch rejected; SupCon positive and finite.')


if __name__ == '__main__':
    main()
