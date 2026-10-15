"""Validate the external request contract without loading Torch or the model."""
import unittest
from chat_api import normalize_request,format_response,openapi_spec


class ContractChecks(unittest.TestCase):
    def test_default_is_original_model(self):
        request=normalize_request({'text':'你好'})
        self.assertEqual(request['mode'],'raw');self.assertEqual(request['max_new_tokens'],768)

    def test_image_only_and_data_url(self):
        self.assertEqual(normalize_request({'image_base64':'data:image/png;base64,YWJj'})['image'],'YWJj')

    def test_missing_or_invalid_inputs(self):
        for value in [[],{}, {'text':' '},{'text':12},{'text':'a'*6001},{'text':'你好','session_id':[]},
                      {'text':'你好','image_base64':{}},{'text':'你好','image':'misnamed'},
                      {'text':'你好','image_base64':'data:image/tiff;base64,YQ=='}]:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(ValueError):normalize_request(value)

    def test_token_limit_and_guarded_mode(self):
        for cap in [0,1025,True,1.5,'16']:
            with self.subTest(cap=cap):
                with self.assertRaises(ValueError):normalize_request({'text':'你好','max_new_tokens':cap})
        self.assertEqual(normalize_request({'text':'你好','max_new_tokens':32})['max_new_tokens'],32)
        with self.assertRaises(ValueError):normalize_request({'text':'你好','mode':'guarded','max_new_tokens':32})

    def test_response_preserves_answer_and_labels(self):
        reply={'session_id':'example','mode':'raw','answer':'原始回答','latency_sec':1.2,'truncated':True,
               'guard_status':'raw_unchecked','usage':{'input_tokens':8,'output_tokens':32}}
        obj=format_response(reply,'request-example')
        self.assertEqual(obj['answer'],reply['answer']);self.assertEqual(obj['finish_reason'],'length')
        self.assertEqual(obj['workflow_version'],'raw_base');self.assertEqual(obj['request_id'],'request-example')

    def test_openapi_contains_live_routes(self):
        self.assertEqual(set(openapi_spec()['paths']),{'/healthz','/v1/models','/v1/chat','/v1/sessions/reset'})


if __name__=='__main__':unittest.main()
