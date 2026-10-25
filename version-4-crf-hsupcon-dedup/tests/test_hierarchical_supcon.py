"""CPU checks for the runnable notebook; run this file with Python."""
import ast
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import ElectraConfig, ElectraForTokenClassification


def load_notebook():
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
    tree = ast.parse(''.join(parts))
    definitions = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name in ('HierarchicalSupConLoss', 'ElectraCrf')]
    definitions += [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'optimizer_for']
    namespace = dict(globals(), LABELS=['O', 'B-CQ', 'I-CQ', 'B-ĐT', 'I-ĐT',
                                     'B-VBPL', 'I-VBPL', 'B-NG', 'I-NG', 'B-SL', 'I-SL'])
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<runner>', 'exec'), namespace)
    return namespace


def reference_loss(features, labels, temperature, alpha):
    z = F.normalize(features, dim=-1)
    terms = []
    for i in range(len(labels)):
        if labels[i] == 0:
            continue
        candidates = [j for j in range(len(labels)) if j != i and labels[j] != 0]
        weights = [1.0 if labels[i] == labels[j] else
                   alpha if (labels[i] - 1) // 2 == (labels[j] - 1) // 2 else 0.0
                   for j in candidates]
        if sum(weights) == 0:
            continue
        scores = torch.stack([z[i].dot(z[j]) / temperature for j in candidates])
        terms.append(-(torch.tensor(weights) * F.log_softmax(scores, dim=0)).sum() / sum(weights))
    return torch.stack(terms).mean() if terms else features.sum() * 0


def main():
    torch.manual_seed(42)
    ns = load_notebook()
    loss_fn = ns['HierarchicalSupConLoss'](temperature=0.07, alpha=0.3)
    x = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.tensor([[1, 2, 1, 3], [3, 0, 9, 2]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    actual = loss_fn(x, labels, mask)
    expected = reference_loss(x[mask], labels[mask].tolist(), 0.07, 0.3)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[1, 1].abs().sum() == 0 and x.grad[1, 3].abs().sum() == 0
    for label_list in ([0, 0], [1], [1, 3], [0, 1], [1, 1]):
        values = torch.randn(1, len(label_list), 8, requires_grad=True)
        y = torch.tensor([label_list])
        loss = loss_fn(values, y, torch.ones_like(y, dtype=torch.bool))
        assert torch.isfinite(loss)
        if label_list != [1, 1]:
            assert loss.item() == 0
        loss.backward()
        assert torch.isfinite(values.grad).all()
    cfg = ElectraConfig(vocab_size=32, embedding_size=16, hidden_size=16,
                       num_hidden_layers=1, num_attention_heads=2,
                       intermediate_size=32, num_labels=11, hidden_dropout_prob=0,
                       attention_probs_dropout_prob=0, classifier_dropout=0)
    cfg.use_supcon = True
    cfg.lambda_supcon = 0.3
    cfg.supcon_temp = 0.07
    cfg.supcon_alpha = 0.3
    model = ns['ElectraCrf'](cfg)
    batch = dict(input_ids=torch.randint(0, 32, (2, 6)),
                 attention_mask=torch.ones(2, 6, dtype=torch.long),
                 valid_ids=torch.tensor([[1, 2, 3], [1, 2, 3]]),
                 label_masks=torch.ones(2, 3, dtype=torch.bool),
                 labels=torch.tensor([[1, 2, 0], [1, 3, 4]]))
    model.train()
    out = model(**batch)
    hidden = model.electra(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])[0]
    words = hidden[torch.arange(2).unsqueeze(1), batch['valid_ids']]
    emissions = model.classifier(model.dropout(words)).float()
    direct_crf = -model.crf(emissions, batch['labels'], mask=batch['label_masks'], reduction='mean')
    torch.testing.assert_close(out.loss_crf, direct_crf)
    torch.testing.assert_close(out.loss, out.loss_crf + 0.3 * out.loss_supcon)
    assert out.tags is None
    out.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.projection_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.electra.parameters())
    optimizer = ns['optimizer_for'](model, dict(learning_rate=2e-5,
                          classifier_learning_rate=1e-3, weight_decay=.01, adam_epsilon=1e-6))
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters()}
    model.eval()
    def forbidden(*args, **kwargs):
        raise AssertionError('Projection head must not execute during evaluation')
    hook = model.projection_head.register_forward_pre_hook(forbidden)
    evaluated = model(**batch, decode=True)
    assert len(evaluated.tags) == 2 and evaluated.loss_supcon.item() == 0
    torch.testing.assert_close(evaluated.loss, evaluated.loss_crf)
    inferred = model(**{key: value for key, value in batch.items() if key != 'labels'}, decode=True)
    assert inferred.loss is None and inferred.tags == evaluated.tags
    hook.remove()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'checkpoint.pt'
        torch.save(model.state_dict(), path)
        restored = ns['ElectraCrf'](cfg)
        restored.load_state_dict(torch.load(path, weights_only=True))
        restored.eval()
        assert restored(**batch, decode=True).tags == evaluated.tags
    cfg.use_supcon = False
    plain = ns['ElectraCrf'](cfg)
    plain.train()
    plain_out = plain(**batch)
    torch.testing.assert_close(plain_out.loss, plain_out.loss_crf)
    assert plain_out.loss_supcon.item() == 0
    print('PASS: weighted loss reference, masks, degenerate batches, backward, optimizer, inference, checkpoint, disabled SupCon.')


if __name__ == '__main__':
    main()
