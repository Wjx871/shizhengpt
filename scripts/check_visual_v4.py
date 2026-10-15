"""Scope regressions: preserve visible basics instead of whole-image refusals."""
import json
import pathlib
import tempfile
import unittest
from PIL import Image
from visual_observation_v4 import observe, basic_clauses


class ScopeChecks(unittest.TestCase):
    def test_clause_filter_retains_basic_observation(self):
        self.assertEqual(basic_clauses('衣服为橙色，带有可辨识型号，材质为金属。'),'衣服为橙色。')
        self.assertEqual(basic_clauses('人物局部特写，仅能看到额头。'),'人物局部特写。')
    def test_clinical_absence_claim_is_removed(self):
        self.assertEqual(basic_clauses('可见舌头，无明显裂纹或瘀斑，湿润度适中。'),'可见舌头。')
    def test_same_inference_count_and_useful_basics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)/'test.png';Image.new('RGB',(150,150),'grey').save(path)
            calls=[]
            def infer(*args):
                calls.append(args)
                return {'raw_output':json.dumps({'observations':['衣服为橙色，材质为金属。','背景可见旗帜。'],'limitations':['仅能看到额头。']},ensure_ascii=False),'truncated':False}
            answer,status,trace,count=observe(infer,path)
            self.assertEqual(status,'image_observed');self.assertEqual(count,1);self.assertEqual(len(calls),1)
            self.assertIn('衣服为橙色',answer);self.assertIn('背景可见旗帜',answer)
            self.assertNotIn('金属',answer);self.assertNotIn('仅能看到',answer)


if __name__=='__main__':unittest.main()
