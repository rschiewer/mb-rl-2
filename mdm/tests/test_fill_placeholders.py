import unittest


from mdm.utils.utils import fill_placeholders, load_yaml, here


class FillPlaceholdersTest(unittest.TestCase):

    def test_fill_placeholders(self):
        cfg = load_yaml(here() / 'test_cfg.yaml')
        fill_placeholders(cfg)

        self.assertEqual(cfg['main_1'], f'{cfg["main_0"]} world')
        self.assertEqual(cfg['nested_0']['param_0'], 42)
        self.assertEqual(cfg['nested_0']['param_1'], 125.45)
        self.assertEqual(cfg['nested_0']['param_2'], 10e-3)
        self.assertEqual(cfg['nested_0']['param_3'], 4E2)
        self.assertEqual(cfg['nested_0']['param_4'], 42)
        self.assertEqual(cfg['nested_0']['param_5'], 125.45)
        self.assertEqual(cfg['nested_0']['param_6'], 10e-3)
        self.assertEqual(cfg['nested_0']['param_7'], 42125.45)
        self.assertEqual(cfg['nested_1']['param_0'], cfg['nested_0']['param_4'])
        self.assertEqual(cfg['nested_1']['param_0'], cfg['nested_0']['param_0'])
        self.assertEqual(cfg['nested_1']['param_1'], cfg['main_0'])
        self.assertEqual(cfg['nested_1']['param_2'], 'hello hello')
        self.assertEqual(cfg['nested_1']['param_3'], 'hellohello')


if __name__ == '__main__':
    unittest.main()
