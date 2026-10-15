"""Pure-stdlib public API contract; inference stays in chat_server.py."""
import re

API_VERSION='1.0.0'
MODEL_ID='ShizhenGPT-7B-VL'
WEIGHTS_VERSION='base-bf16'
MODES=('raw','guarded','guarded_v1')


def normalize_request(payload):
    if not isinstance(payload,dict):raise ValueError('请求必须是 JSON 对象。')
    extra=set(payload)-{'text','image_base64','session_id','mode','max_new_tokens'}
    if extra:raise ValueError('未知字段：'+', '.join(sorted(extra)))
    text=payload.get('text','')
    if not isinstance(text,str) or len(text)>6000:raise ValueError('text 必须是最多 6000 字的字符串。')
    mode=payload.get('mode','raw')
    if mode not in MODES:raise ValueError('mode 必须为 raw、guarded 或 guarded_v1。')
    sid=payload.get('session_id')
    if sid is not None and (not isinstance(sid,str) or not sid or len(sid)>80):raise ValueError('session_id 必须是有效的会话字符串。')
    image=payload.get('image_base64')
    if image is not None and not isinstance(image,str):raise ValueError('image_base64 必须是 Base64 字符串。')
    if image and image.startswith('data:'):
        header,sep,image=image.partition(',')
        if not sep or not re.fullmatch(r'data:image/(?:jpeg|png|webp);base64',header):
            raise ValueError('图片 Data URL 仅支持 JPEG、PNG 或 WebP 的 Base64。')
    if not text.strip() and not image:raise ValueError('text 和 image_base64 至少提供一项。')
    cap=payload.get('max_new_tokens',768)
    if isinstance(cap,bool) or not isinstance(cap,int) or not 1<=cap<=1024:
        raise ValueError('max_new_tokens 必须是 1 到 1024 的整数。')
    if mode!='raw' and 'max_new_tokens' in payload:
        raise ValueError('max_new_tokens 仅用于 raw；谨慎模式使用固定的阶段上限。')
    return {'text':text,'image':image,'session_id':sid,'mode':mode,'max_new_tokens':cap}


def format_response(reply,request_id):
    return {'request_id':request_id,'api_version':API_VERSION,'model':MODEL_ID,'weights_version':WEIGHTS_VERSION,
            'session_id':reply['session_id'],'mode':reply['mode'],
            'workflow_version':reply.get('workflow_version','raw_base'),
            'answer':reply['answer'],'latency_sec':reply['latency_sec'],
            'finish_reason':'length' if reply.get('truncated') else 'stop',
            'truncated':reply.get('truncated',False),'guard_status':reply['guard_status'],
            'usage':reply.get('usage'),'model_invocations':reply.get('model_invocations',1)}


def model_info(workflow_version):
    return {'api_version':API_VERSION,'model':MODEL_ID,'weights_version':WEIGHTS_VERSION,
            'default_mode':'raw','available_modes':list(MODES),'cautious_workflow_version':workflow_version,
            'input_types':['text','image','text_and_image'],'streaming':False,
            'limits':{'text_characters':6000,'image_bytes':8*1024*1024,'image_pixels':24000000,
                      'request_bytes':12*1024*1024,'input_tokens_raw':12000,'raw_max_new_tokens':1024,
                      'concurrent_inferences':1,'session_idle_seconds':7200,'max_replies':15},
            'image_preprocessing':'EXIF transpose, RGB, longest edge <=1600, JPEG quality 95',
            'guarded_usage_note':'usage is null for guarded workflows; model_invocations counts stages/retries'}


def openapi_spec():
    error={'description':'失败响应；查看 error.code 和 error.message，request_id 可用于定位请求。',
           'content':{'application/json':{'schema':{'$ref':'#/components/schemas/Error'}}}}
    properties={
        'text':{'type':'string','maxLength':6000,'default':'','description':'与图片至少提供一项。'},
        'image_base64':{'type':'string','description':'JPEG/PNG/WebP 纯 Base64 或 data:image/...;base64,...；解码后最多8 MB。'},
        'session_id':{'type':'string','description':'首次省略，续聊传回上次返回值；模式必须一致。'},
        'mode':{'type':'string','enum':list(MODES),'default':'raw'},
        'max_new_tokens':{'type':'integer','minimum':1,'maximum':1024,'default':768,'description':'仅 raw 支持。'},
    }
    return {'openapi':'3.0.3','info':{'title':'ShizhenGPT 图文问答接口','version':API_VERSION,
            'description':'自定义 REST JSON 接口，非 OpenAI 协议。默认基础权重 raw 模式。'},
            'servers':[{'url':'http://127.0.0.1:7860'}],
            'paths':{
                '/healthz':{'get':{'summary':'服务就绪与版本','responses':{'200':{'description':'模型已加载'}}}},
                '/v1/models':{'get':{'summary':'模型、模式与输入限制','responses':{'200':{'description':'模型元数据'}}}},
                '/v1/chat':{'post':{'summary':'文字、图片或图文联合问答',
                    'requestBody':{'required':True,'content':{'application/json':{'schema':{'$ref':'#/components/schemas/ChatRequest'}}}},
                    'responses':{'200':{'description':'模型回答','content':{'application/json':{'schema':{'$ref':'#/components/schemas/ChatResponse'}}}},
                                 '400':error,'404':error,'413':error,'500':error,'503':error}}},
                '/v1/sessions/reset':{'post':{'summary':'释放一个会话及其临时图片',
                    'requestBody':{'required':True,'content':{'application/json':{'schema':{'type':'object','required':['session_id'],
                        'additionalProperties':False,'properties':{'session_id':{'type':'string','minLength':1,'maxLength':80}}}}}},
                    'responses':{'200':{'description':'幂等清理成功'},'400':error,'503':error}}},
            },'components':{'schemas':{
                'ChatRequest':{'type':'object','additionalProperties':False,'properties':properties,
                    'anyOf':[{'required':['text'],'properties':{'text':{'minLength':1}}},
                             {'required':['image_base64'],'properties':{'image_base64':{'minLength':1}}}]},
                'ChatResponse':{'type':'object','required':['request_id','session_id','answer','model','mode','finish_reason'],
                    'properties':{'request_id':{'type':'string'},'api_version':{'type':'string'},'model':{'type':'string'},
                        'weights_version':{'type':'string'},'session_id':{'type':'string'},'mode':{'type':'string'},
                        'workflow_version':{'type':'string'},'answer':{'type':'string'},'latency_sec':{'type':'number'},
                        'finish_reason':{'type':'string','enum':['stop','length']},'truncated':{'type':'boolean'},
                        'guard_status':{'type':'string'},'model_invocations':{'type':'integer'},
                        'usage':{'type':'object','nullable':True,'properties':{'input_tokens':{'type':'integer'},'output_tokens':{'type':'integer'}}}}},
                'Error':{'type':'object','properties':{'request_id':{'type':'string'},'error':{'type':'object',
                    'properties':{'code':{'type':'string'},'message':{'type':'string'}}}}},
            }}}
