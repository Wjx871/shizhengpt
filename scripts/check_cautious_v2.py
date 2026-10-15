"""Regression checks for routing changes, without model or GPU dependencies."""
import json
import unittest
from cautious_chat_v2 import run, extract_sources, question_checks
from consultation_alignment_v4 import fact_ledger


def no_inference(*args):
    raise AssertionError('This request should not invoke the model')


class RoutingChecks(unittest.TestCase):
    def test_polite_self_report_keeps_exact_substring(self):
        original='请问，我最近三天轻微疲劳，休息后减轻。'
        sources=extract_sources([original])
        self.assertEqual(sources,['我最近三天轻微疲劳，休息后减轻'])
        self.assertIn(sources[0],original)

    def test_request_for_help_with_real_complaint(self):
        self.assertEqual(extract_sources(['请帮我分析，我今天有点腹胀。']),['我今天有点腹胀'])

    def test_hypothetical_is_not_a_report(self):
        for text in ['请问，如果我发热会是什么原因？','我如果发热会是什么原因？']:
            with self.subTest(text=text):self.assertEqual(extract_sources([text]),[])

    def test_medication_question_cannot_reuse_old_symptoms(self):
        result=run(no_inference,['我最近轻微疲劳。','那这个药的标准剂量是多少？'])
        self.assertEqual(result['status'],'knowledge_abstained')
        self.assertNotIn('待核实的可能方向',result['answer'])

    def test_correction_keeps_latest_source(self):
        self.assertEqual(extract_sources(['我有发热。','更正，我没有发热。']),['更正，我没有发热'])

    def test_definite_statement_cannot_hide_in_question(self):
        self.assertTrue(question_checks('你已经确诊气虚，肯定是长期失眠吗？',fact_ledger(['睡眠良好'])))
        self.assertFalse(question_checks('近期是否有入睡困难？',fact_ledger(['睡眠良好'])))

    def test_valid_report_still_gets_a_checked_answer(self):
        response={'possible_explanations':[{'hypothesis':'需核实近期活动与疲劳的可能关系',
                  'evidence':[{'source_id':'S1','quote':'轻微疲劳'}],'missing_info':['近期活动量']}],
                  'questions':['最近活动量是否增加？'],'limitations':[]}
        def infer(*args):return {'raw_output':json.dumps(response,ensure_ascii=False),'truncated':False,'latency_sec':0}
        result=run(infer,['请问，我最近三天轻微疲劳，休息后减轻。'])
        self.assertEqual(result['status'],'limited_answer')
        self.assertIn('最近活动量是否增加',result['answer'])


if __name__=='__main__':unittest.main()
