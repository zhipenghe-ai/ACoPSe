import contextlib
import io
import unittest

from configs import build_parser


class TrainArgumentTests(unittest.TestCase):
    def setUp(self):
        self.parsers = {preset: build_parser(preset) for preset in ('mirflickr', 'nuswide')}

    def test_training_is_enabled_by_default(self):
        for preset, parser in self.parsers.items():
            with self.subTest(preset=preset):
                self.assertIs(parser.parse_args([]).TRAIN, True)

    def test_parses_explicit_boolean_values(self):
        values = (
            ('true', True), ('TRUE', True), ('1', True), ('yes', True), ('on', True),
            ('false', False), ('False', False), ('0', False), ('no', False), ('off', False),
            (' true ', True), (' false ', False),
        )
        for preset, parser in self.parsers.items():
            for value, expected in values:
                with self.subTest(preset=preset, value=value):
                    self.assertIs(parser.parse_args(['--TRAIN', value]).TRAIN, expected)

    def test_preserves_explicit_empty_string_for_evaluation(self):
        for preset, parser in self.parsers.items():
            with self.subTest(preset=preset):
                self.assertIs(parser.parse_args(['--TRAIN=']).TRAIN, False)

    def test_rejects_unrecognized_values(self):
        for preset, parser in self.parsers.items():
            for value in ('maybe', '2', ' '):
                with self.subTest(preset=preset, value=value):
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
                        parser.parse_args(['--TRAIN', value])
                    self.assertEqual(exc.exception.code, 2)
                    self.assertIn('--TRAIN', stderr.getvalue())
                    self.assertIn('expected a Boolean value', stderr.getvalue())

    def test_requires_an_argument_when_the_option_is_present(self):
        for preset, parser in self.parsers.items():
            with self.subTest(preset=preset):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
                    parser.parse_args(['--TRAIN'])
                self.assertEqual(exc.exception.code, 2)
                self.assertIn('expected one argument', stderr.getvalue())
