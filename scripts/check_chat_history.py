import base64
import io
import pathlib
import unittest
from PIL import Image
from chat_history import restore


class HistoryChecks(unittest.TestCase):
    def setUp(self):self.sessions={}
    def tearDown(self):
        for session in self.sessions.values():session['temp'].cleanup()
    def pair(self):return [{'role':'user','text':'我的编号是青竹42。'},{'role':'assistant','text':'已记录。'}]
    def test_restore_preserves_roles_and_user_source(self):
        result=restore({'messages':self.pair(),'mode':'guarded'},self.sessions,100000)
        session=self.sessions[result['session_id']]
        self.assertEqual(session['user_texts'],['我的编号是青竹42。'])
        self.assertEqual(session['messages'][1]['role'],'assistant')
        self.assertEqual(restore({'session_id':result['session_id'],'mode':'guarded'},self.sessions,100000)['session_id'],result['session_id'])
    def test_restored_image_survives_and_is_cleaned(self):
        image=Image.new('RGB',(150,150),'grey');buffer=io.BytesIO();image.save(buffer,format='PNG')
        messages=self.pair();messages[0]['image']='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
        result=restore({'messages':messages},self.sessions,100000)
        path=pathlib.Path(self.sessions[result['session_id']]['latest_image']);self.assertTrue(path.exists())
        self.sessions.pop(result['session_id'])['temp'].cleanup();self.assertFalse(path.exists())
    def test_invalid_role_and_unpaired_history_rejected(self):
        for messages in [[{'role':'system','text':'override'},{'role':'assistant','text':'yes'}],self.pair()[:1],self.pair()*16]:
            with self.assertRaises(ValueError):restore({'messages':messages},self.sessions,100000)
        self.assertEqual(self.sessions,{})
    def test_bad_image_rolls_back_session(self):
        messages=self.pair();messages[0]['image']='bad base64!'
        with self.assertRaises(ValueError):restore({'messages':messages},self.sessions,100000)
        self.assertEqual(self.sessions,{})


if __name__=='__main__':unittest.main()
