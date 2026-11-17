import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from configs import build_parser
import main_nus
from my_opt import PCIRH


class EvaluationTests(unittest.TestCase):
    def test_nus_standalone_loads_the_saved_pth_checkpoint(self):
        log = logging.getLogger('test.checkpoint')
        load = torch.load

        def init_fixture(trainer, trainer_log, config):
            trainer.log = trainer_log
            trainer.config = config
            trainer.imgnet = torch.nn.Linear(2, 1, bias=False)
            trainer.txtnet = torch.nn.Linear(2, 1, bias=False)

        def load_on_cpu(path, map_location):
            return load(path, map_location='cpu')

        with tempfile.TemporaryDirectory() as model_dir:
            config = build_parser('nuswide').parse_args(['--TRAIN', 'False', '--MODEL_DIR', model_dir])
            saved = PCIRH.__new__(PCIRH)
            init_fixture(saved, log, config)
            with torch.no_grad():
                saved.imgnet.weight.fill_(2.0)
                saved.txtnet.weight.fill_(3.0)
            saved.save_checkpoints()
            expected_path = Path(model_dir) / 'NUSWIDE_16bits.pth'
            self.assertTrue(expected_path.is_file())

            def verify_weights(trainer):
                self.assertTrue(torch.equal(trainer.imgnet.weight, saved.imgnet.weight))
                self.assertTrue(torch.equal(trainer.txtnet.weight, saved.txtnet.weight))

            # Use real serialization with CPU mapping so the filename regression can run without a GPU.
            with patch.object(PCIRH, '__init__', init_fixture), \
                 patch.object(PCIRH, 'performance_eval', autospec=True, side_effect=verify_weights) as evaluate, \
                 patch('my_opt.torch.load', side_effect=load_on_cpu) as read, \
                 patch('torch.cuda.set_device'), patch('utils.logger', return_value=log):
                main_nus.main(config)

            evaluate.assert_called_once()
            read.assert_called_once()
            self.assertEqual(Path(read.call_args[0][0]), expected_path)

    def test_active_evaluator_uses_and_logs_map_at_5000(self):
        trainer = PCIRH.__new__(PCIRH)
        trainer.log = logging.getLogger('test.retrieval')
        trainer.imgnet = Mock()
        trainer.txtnet = Mock()
        trainer.database_loader = object()
        trainer.test_loader = object()
        database_image, database_text, database_labels, query_image, query_text, query_labels = (
            object() for _ in range(6)
        )
        codes = (database_image, database_text, database_labels, query_image, query_text, query_labels)

        with patch('my_opt.compress', return_value=codes), \
             patch('my_opt.calculate_top_map', side_effect=(0.25, 0.75)) as calculate, \
             self.assertLogs(trainer.log, 'INFO') as captured:
            self.assertEqual(trainer.performance_eval(), (0.25, 0.75))

        self.assertEqual([call[1] for call in calculate.call_args_list], [
            dict(qu_B=query_image, re_B=database_text, qu_L=query_labels, re_L=database_labels, topk=5000),
            dict(qu_B=query_text, re_B=database_image, qu_L=query_labels, re_L=database_labels, topk=5000),
        ])
        self.assertEqual([record.getMessage() for record in captured.records], [
            'Evaluation: MAP@5000',
            'MAP@5000 I->T: 0.250, MAP@5000 T->I: 0.750',
        ])


if __name__ == '__main__':
    unittest.main()
