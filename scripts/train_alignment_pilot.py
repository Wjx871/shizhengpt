"""One frozen LoRA pilot with assistant-only labels. Leaves original weights intact."""
import hashlib
import json
import pathlib
import random
import time

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/alignment_lora_pilot_20261009'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    import torch
    from peft import LoraConfig,TaskType,get_peft_model
    from qwen_vl_utils import process_vision_info
    from infer_image import Engine
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for name,expected in protocol['source_sha256'].items():assert sha(ROOT/'scripts'/name)==expected,name
    assert sha(OUT/'train.jsonl')==protocol['train_jsonl_sha256']
    if (OUT/'adapter').exists():raise RuntimeError('Do not overwrite adapter')
    cfg=protocol['training'];torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    engine=Engine();torch.manual_seed(cfg['seed'])
    base=engine.model
    targets=[name for name,module in base.named_modules() if isinstance(module,torch.nn.Linear)
             and any(name.endswith(suffix) for suffix in protocol['lora']['target_suffixes'])]
    assert targets and all('visual' not in name for name in targets)
    adapter=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,bias='none',
        target_modules=targets,task_type=TaskType.CAUSAL_LM))
    trainable=[(name,p) for name,p in adapter.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in name for name,_ in trainable)
    assert all(not p.requires_grad for name,p in adapter.named_parameters() if 'lora_' not in name)
    # Spot checks accompany the complete gradient-freeze invariant; original files
    # are never a save target and optimizer only receives LoRA parameters.
    sentinels={name:p.detach().reshape(-1)[:16].clone() for name,p in adapter.named_parameters()
               if 'lora_' not in name and any(term in name for term in ['embed_tokens.weight','lm_head.weight'])}
    adapter.config.use_cache=False
    adapter.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    adapter.enable_input_require_grads()
    adapter.train()
    optimizer=torch.optim.AdamW([p for _,p in trainable],lr=cfg['learning_rate'],weight_decay=0)
    rows=[json.loads(line) for line in (OUT/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    random.Random(cfg['seed']).shuffle(rows)
    assert len(rows)==cfg['optimizer_steps']==64
    log=[];started=time.time();torch.cuda.reset_peak_memory_stats()
    for step,row in enumerate(rows,1):
        content=[]
        if row['image']:
            image=ROOT/row['image'];assert sha(image)==row['image_sha256']
            content.append(dict(type='image',image=str(image),max_pixels=engine.config['max_pixels']))
        content.append(dict(type='text',text=row['prompt']))
        user=[dict(role='user',content=content)]
        answer=json.dumps(row['answer'],ensure_ascii=False,separators=(',',':'))
        messages=user+[dict(role='assistant',content=answer)]
        prefix=engine.processor.apply_chat_template(user,tokenize=False,add_generation_prompt=True)
        full=engine.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        images,videos=process_vision_info(user)
        inputs=engine.processor(text=[full],images=images,videos=videos,padding=True,return_tensors='pt').to('cuda:0')
        prefix_inputs=engine.processor(text=[prefix],images=images,videos=videos,padding=True,return_tensors='pt')
        prefix_ids=prefix_inputs.input_ids[0]
        prefix_len=len(prefix_ids)
        assert torch.equal(inputs.input_ids[0,:prefix_len].cpu(),prefix_ids),'assistant boundary mismatch'
        assert inputs.input_ids.shape[1]<=cfg['max_sequence_tokens'],'Do not silently truncate training'
        labels=inputs.input_ids.clone();labels[:,:prefix_len]=-100
        assert (labels!=-100).sum()>0
        optimizer.zero_grad(set_to_none=True)
        result=adapter(**inputs,labels=labels,use_cache=False)
        loss=result.loss
        assert torch.isfinite(loss),'nonfinite loss'
        loss.backward()
        grads=[p.grad for _,p in trainable if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads),'nonfinite or missing adapter gradients'
        assert all(p.grad is None for name,p in adapter.named_parameters() if 'lora_' not in name),'base gradient leak'
        norm=torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.0)
        optimizer.step();torch.cuda.synchronize()
        item=dict(step=step,id=row['id'],task=row['task'],loss=float(loss.detach()),gradient_norm=float(norm),
                  input_tokens=inputs.input_ids.shape[1],supervised_tokens=int((labels!=-100).sum()),
                  elapsed_sec=time.time()-started)
        log.append(item)
        with (OUT/'training_steps.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(item)+'\n')
        if step==1 or step%4==0:print('TRAIN',step,'of',len(rows),'loss',round(item['loss'],4),'seconds',round(item['elapsed_sec'],1),flush=True)
        del inputs,prefix_inputs,result,loss,labels
    named=dict(adapter.named_parameters())
    for name,before in sentinels.items():assert torch.equal(named[name].detach().reshape(-1)[:16],before),'base sentinel changed'
    path=OUT/'adapter';adapter.save_pretrained(path,safe_serialization=True)
    save(OUT/'training_run.json',dict(optimizer_steps=len(log),elapsed_sec=time.time()-started,
        trainable_parameters=sum(p.numel() for _,p in trainable),target_modules=targets,
        trainable_dtype=sorted({str(p.dtype) for _,p in trainable}),base_dtype=str(next(base.parameters()).dtype),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        first_8_mean_loss=sum(r['loss'] for r in log[:8])/8,last_8_mean_loss=sum(r['loss'] for r in log[-8:])/8,
        original_weights_saved=False,base_gradients_none=True,base_sentinels_unchanged=True,
        mask='Only assistant completion tokens supervised; prompt/image tokens masked -100',
        protocol_sha256=sha(OUT/'protocol.json'),adapter_sha256=sha(path/'adapter_model.safetensors'),
        clinical_training=False))
    print('TRAIN_COMPLETE',len(log),'ADAPTER',str(path),flush=True)


if __name__=='__main__':main()
