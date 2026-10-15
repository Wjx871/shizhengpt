"""Focused routing and item-filter regressions; not a visual accuracy test."""
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from cautious_chat_v3 import run


class VisualChecks(unittest.TestCase):
    def test_observations_for_colour_grey_and_local_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            for colour in [(120, 80, 60), (120, 120, 120)]:
                path=Path(tmp)/'sample.png'
                Image.new('RGB',(160,160),colour).save(path)
                calls=[]
                def infer(prompt,image=None):
                    calls.append(image)
                    return {'raw_output':json.dumps({'observations':['图中可见一个局部区域。'], 'limitations':['其他部位未显示。']},ensure_ascii=False),'truncated':False}
                result=run(infer,['请描述图片'],str(path))
                self.assertEqual(result['status'],'image_observed')
                self.assertEqual(result['model_invocations'],1)
                self.assertEqual(calls,[str(path)])
                self.assertIn('局部区域',result['answer'])
                self.assertNotIn('彩色像素',result['answer'])

    def test_black_abstains_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'black.png';Image.new('RGB',(160,160)).save(path)
            def infer(*args):raise AssertionError('blank image should not invoke model')
            result=run(infer,['描述图片'],str(path))
            self.assertEqual(result['status'],'image_abstained')

    def test_drop_clinical_item_keep_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.png';Image.new('RGB',(160,160),'grey').save(path)
            result=run(lambda *args:json.dumps({'observations':['可见舌头。','提示脾虚。'],'limitations':[]},ensure_ascii=False),['描述图片'],str(path))
            self.assertIn('可见舌头',result['answer']);self.assertNotIn('提示脾虚',result['answer'])

    def test_invalid_json_is_not_displayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.png';Image.new('RGB',(160,160),'grey').save(path)
            result=run(lambda *args:'确诊为某种疾病',['描述图片'],str(path))
            self.assertEqual(result['status'],'image_uncertain')
            self.assertNotIn('确诊为',result['answer'])

    def test_text_route_preserved(self):
        result=run(lambda *args:None,['问诊前需要提供什么信息？'])
        self.assertEqual(result['status'],'informational')
        self.assertEqual(result['model_invocations'],0)

    def test_trailing_comma_keeps_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.png';Image.new('RGB',(160,160),'grey').save(path)
            raw='```json\n{"observations":["可见灰色衣物。",],"limitations":[],}\n```'
            result=run(lambda *args:{'raw_output':raw,'truncated':False},['描述图片'],str(path))
            self.assertEqual(result['status'],'image_observed')
            self.assertIn('灰色衣物',result['answer'])


if __name__=='__main__':unittest.main()
