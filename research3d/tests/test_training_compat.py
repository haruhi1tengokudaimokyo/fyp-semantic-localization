"""The compatibility hook must count finite AMP updates and skip overflow steps."""

import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_ace_local import restore_step_counter


class TrainingCompatibility(unittest.TestCase):
    def test_real_step_and_amp_skip(self):
        import torch

        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        restore_step_counter(optimizer)
        scaler = torch.amp.GradScaler("cpu", init_scale=2.0)
        optimizer.zero_grad()
        scaler.scale(parameter.square().sum()).backward()
        scaler.step(optimizer)
        scaler.update()
        self.assertEqual(optimizer._step_count, 1)
        after = parameter.detach().clone()
        optimizer.zero_grad()
        scaler.scale((parameter * float("inf")).sum()).backward()
        scaler.step(optimizer)
        scaler.update()
        self.assertEqual(optimizer._step_count, 1)
        self.assertTrue(torch.equal(parameter, after))


if __name__ == "__main__":
    unittest.main()
