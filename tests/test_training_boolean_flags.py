import argparse
import ast
import io
import re
import runpy
import shlex
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "code" / "train.py"
DEFAULTS = {"bn_freeze": True, "l2_norm": True, "enableMemory": False, "visualize": False}


def compile_statements(statements):
    return compile(ast.Module(body=statements, type_ignores=[]), str(TRAIN), "exec")


def condition(statements, expression):
    return next(
        node for node in statements
        if isinstance(node, ast.If) and ast.unparse(node.test) == expression
    )


# train.py starts CUDA, W&B and dataset loading at import time. Select its
# parser and the production branches under test without those side effects.
body = ast.parse(TRAIN.read_bytes(), filename=str(TRAIN)).body
parser_start = next(i for i, node in enumerate(body) if ast.unparse(node).startswith(
    "parser = argparse.ArgumentParser("
))
parser_end = next(i for i, node in enumerate(body) if ast.unparse(node) == "args = parser.parse_args()")
helpers = [node for node in body if isinstance(node, ast.FunctionDef) and node.name == "str2bool"]
PARSER_CODE = compile_statements([*helpers, *body[parser_start:parser_end]])
BACKBONE_CODE = compile_statements([condition(body, "args.model.find('googlenet') + 1")])
MEMORY_CODE = compile_statements([condition(body, "args.enableMemory")])
epoch_body = next(node.body for node in body if isinstance(node, ast.For)
                  and ast.unparse(node.iter) == "range(args.num_epochs)")
EPOCH_MODE_CODE = compile_statements([
    next(node for node in epoch_body if ast.unparse(node) == "model.train()"),
    condition(epoch_body, "args.bn_freeze"),
])
VISUALIZATION_CODE = compile_statements([condition(epoch_body, "epoch == 0 and args.visualize")])


def parse(*arguments):
    scope = {"argparse": argparse}
    exec(PARSER_CODE, scope)  # noqa: S102
    return scope["parser"].parse_args(arguments)


class TrainingBooleanFlagTests(unittest.TestCase):
    def test_accepted_values(self):
        for flag in DEFAULTS:
            for expected, values in (
                (True, ("True", "true", "TRUE", "T", "yes", "Y", "1")),
                (False, ("False", "false", "FALSE", "F", "no", "N", "0")),
            ):
                for value in values:
                    with self.subTest(flag=flag, value=value):
                        self.assertIs(getattr(parse(f"--{flag}", value), flag), expected)

    def test_invalid_values(self):
        for flag in DEFAULTS:
            for value in ("maybe", "falsee", "offish", "none", ""):
                with self.subTest(flag=flag, value=value):
                    stderr = io.StringIO()
                    with self.assertRaises(SystemExit) as error, redirect_stderr(stderr):
                        parse(f"--{flag}", value)
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn(f"argument --{flag}:", stderr.getvalue())
                    self.assertIn("expected a boolean value", stderr.getvalue())

    def test_defaults_and_unrelated_args(self):
        args = parse("--dataset", "SOP", "--gpu_id", "-1", "--lr", "0.002", "--batch_size", "32")
        self.assertEqual({flag: getattr(args, flag) for flag in DEFAULTS}, DEFAULTS)
        self.assertEqual((args.dataset, args.gpu_id, args.lr, args.sz_batch), ("SOP", -1, 0.002, 32))

    def test_readme_commands_disable_bn_freeze(self):
        commands = (ROOT / "README.md").read_text(encoding="utf-8")
        commands = re.sub(r"\\[ \t]*\r?\n", " ", commands)
        checked = []
        for command in re.findall(r"^python train\.py (.+)$", commands, re.MULTILINE):
            argv = shlex.split(command)
            if "--bn_freeze" in argv:
                args = parse(*argv)
                with self.subTest(dataset=args.dataset):
                    self.assertIs(args.bn_freeze, False)
                checked.append(args.dataset)
        self.assertCountEqual(checked, ["SOP", "Inshop"])


class TrainingBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        implementation = runpy.run_path(str(ROOT / "code" / "net" / "bn_inception.py"))
        cls.backbone = implementation["bn_inception"]

    def build_backbone(self, args):
        def without_weights(**kwargs):
            self.assertIs(kwargs["pretrained"], True)
            return self.backbone(**{**kwargs, "pretrained": False})

        factory = Mock(side_effect=without_weights)
        scope = {"args": args, "bn_inception": factory}
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(67)
            exec(BACKBONE_CODE, scope)  # noqa: S102
        factory.assert_called_once_with(
            embedding_size=args.sz_embedding, pretrained=True,
            is_norm=args.l2_norm, bn_freeze=args.bn_freeze,
        )
        return scope["model"]

    def test_cpu_batchnorm_policy_and_optimizer_step(self):
        inputs = torch.linspace(-0.2, 0.8, 2 * 3 * 64 * 64).reshape(2, 3, 64, 64)
        for freeze in (True, False):
            for gpu_id in ("0", "-1"):
                with self.subTest(freeze=freeze, gpu_id=gpu_id):
                    args = parse("--bn_freeze", str(freeze), "--gpu_id", gpu_id, "--embedding_size", "2")
                    backbone = self.build_backbone(args)
                    layers = [module for module in backbone.modules()
                              if isinstance(module, torch.nn.BatchNorm2d)]
                    self.assertTrue(layers)
                    self.assertTrue(all(layer.weight.requires_grad != freeze for layer in layers))
                    self.assertTrue(all(layer.bias.requires_grad != freeze for layer in layers))
                    model = torch.nn.DataParallel(backbone) if gpu_id == "-1" else backbone
                    model.eval()
                    exec(EPOCH_MODE_CODE, {"args": args, "model": model, "torch": torch})  # noqa: S102
                    self.assertTrue(model.training)
                    self.assertTrue(all(layer.training != freeze for layer in layers))

                    before = backbone.embedding.weight.detach().clone()
                    steps = [layer.num_batches_tracked.item() for layer in layers]
                    running_mean = layers[0].running_mean.clone()
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                    outputs = model(inputs)
                    self.assertTrue(torch.isfinite(outputs).all())
                    outputs[:, 0].mean().backward()
                    self.assertEqual([layer.weight.grad is None for layer in layers], [freeze] * len(layers))
                    optimizer.step()
                    self.assertFalse(torch.equal(backbone.embedding.weight, before))
                    self.assertEqual(torch.equal(layers[0].running_mean, running_mean), freeze)
                    self.assertEqual([layer.num_batches_tracked.item() for layer in layers],
                                     [step + int(not freeze) for step in steps])

    def test_embedding_normalization_branch(self):
        for normalize in (True, False):
            with self.subTest(normalize=normalize), torch.no_grad():
                model = self.build_backbone(parse("--l2_norm", str(normalize), "--embedding_size", "2"))
                model.embedding.weight.zero_()
                model.embedding.bias.copy_(torch.tensor([3., 4.]))
                with patch.object(model.model, "forward", return_value=torch.ones(1, 1024, 1, 1)):
                    actual = model(torch.empty(1))
                expected = [[0.6, 0.8]] if normalize else [[3., 4.]]
                torch.testing.assert_close(actual, torch.tensor(expected), atol=1e-6, rtol=0)

    def test_memory_construction_branch(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), redirect_stdout(io.StringIO()):
                args = parse("--enableMemory", str(enabled), "--T", "3", "--embedding_size", "2")
                factory = Mock()
                scope = {"args": args, "CrossBatchMemory": factory}
                exec(MEMORY_CODE, scope)  # noqa: S102
                if enabled:
                    factory.assert_called_once_with(3, 2)
                    self.assertIs(scope["memory_q"], factory.return_value)
                else:
                    factory.assert_not_called()
                    self.assertNotIn("memory_q", scope)

    def test_initial_visualization_branch(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                args = parse("--visualize", str(enabled))
                figure = Mock()
                log = Mock()
                model, batch, proxies = object(), object(), object()
                scope = {"args": args, "epoch": 0, "torch": torch, "model": model,
                         "dl_tr": batch, "criterion": SimpleNamespace(proxies=proxies),
                         "utils": SimpleNamespace(tSNEPlot=figure), "wandb": SimpleNamespace(log=log)}
                exec(VISUALIZATION_CODE, scope)  # noqa: S102
                if enabled:
                    figure.assert_called_once_with(model, batch, n_components=2, proxy=proxies)
                    log.assert_called_once_with({"t-SNE": figure.return_value}, step=0)
                else:
                    figure.assert_not_called()
                    log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
