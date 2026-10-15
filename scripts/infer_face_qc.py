"""Face-photo prototype: deterministic QC gate, with unverified model description."""
import argparse
import json
import pathlib

from face_quality import analyze, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--symptoms-json', help='UTF-8 JSON file containing a list of exact symptom strings')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    symptoms = json.loads(pathlib.Path(args.symptoms_json).read_text(encoding='utf-8')) if args.symptoms_json else []
    if not isinstance(symptoms, list) or not all(isinstance(s, str) for s in symptoms):
        raise ValueError('symptoms must be a list of strings')
    facts = analyze(args.image)
    result = {'cv_facts': facts, 'user_symptoms': symptoms, 'medical_status': 'unknown',
              'model_description_verified': False}
    if not facts['usable_for_technical_description']:
        result.update(action='reject_before_model_load', model_invoked=False)
    else:
        from route_b_compare import compact_cv, parse_response
        from infer_image import Engine
        config = json.loads((ROOT/'configs/route_b_prompts.json').read_text(encoding='utf-8'))
        prompt = config['base']+'\n'+json.dumps({'image_present': True, 'user_symptoms': symptoms,
                  'cv_measurements': compact_cv(facts)}, ensure_ascii=False)
        raw = Engine().infer(prompt, args.image, max_new_tokens=512)
        parsed, valid = parse_response(raw['raw_output'])
        fidelity = valid and parsed['user_symptoms'] == symptoms
        result.update(action='generated_unverified_description', model_invoked=True,
                      schema_valid=valid, symptoms_exact=fidelity,
                      requires_review=True, model_result=raw)
        # Even valid JSON is not proof that visual or measurement claims are accurate.
        # Authoritative facts and symptoms remain program-supplied fields above.
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'action': result['action'], 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
