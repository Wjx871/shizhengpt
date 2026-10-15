"""Bounded Chinese lexical facts with exact source spans; not a medical parser."""
import re

TERMS = {
    'fatigue': r'疲劳|疲乏|乏力', 'fever': r'发热|发烧',
    'chest_pain': r'胸痛|胸部疼痛', 'headache': r'头痛|头疼',
    'dizziness': r'头晕|眩晕', 'palpitations': r'心悸|心慌',
    'cough': r'咳嗽', 'nausea': r'恶心', 'vomiting': r'呕吐',
    'abdominal_pain': r'腹痛|肚子痛', 'bloating': r'腹胀|肚子胀',
    'diarrhea': r'腹泻|拉肚子', 'constipation': r'便秘',
    'cold_limbs': r'手脚(?:较)?凉|手足凉|四肢不温',
    'poor_appetite': r'食欲不佳|食欲差|食欲减退|食欲下降',
    'insomnia': r'失眠|入睡困难', 'dreaming': r'多梦',
    'sleep_quality': r'睡眠(?:质量)?(?:一般|尚可|良好|正常|较差|差|不好|不佳|浅)',
}
QUALIFIERS = {
    'severity': r'轻微|较轻|有点|严重|剧烈|明显|一般|尚可|良好|正常|较差|不好|不佳',
    'frequency': r'偶尔|偶有|有时|经常|每天|持续|间歇|反复',
    'time': r'(?:近|最近|已有|持续)?(?:[一二三四五六七八九十两\d]+)(?:个)?(?:小时|天|周|个月|月|年)|昨天|今天|上周|近日',
}


def mentions(text):
    return [(key, m) for key, pattern in TERMS.items() for m in re.finditer(pattern, text)]


def span(text, match, offset=0):
    return dict(text=match.group(), start=offset+match.start(), end=offset+match.end())


def extract_facts(symptoms):
    records = []
    for index, text in enumerate(symptoms):
        clauses = []
        for clause in re.finditer(r'[^，,；;。\n]+', text):
            content = clause.group()
            findings = []
            for term, match in mentions(content):
                prefix = re.split(r'但是|但|不过', content[:match.start()])[-1]
                # Local clause only: do not spread negation across punctuation.
                polarity = 'unspecified'
                if re.search(r'不再|已(?:经)?缓解|已(?:经)?消失', prefix) or re.search(r'已(?:经)?缓解|已(?:经)?消失', content[match.end():]):
                    polarity = 'resolved'
                elif re.search(r'没有|未出现|无(?:明显)?|否认|不伴|不$', prefix):
                    polarity = 'negative'
                else:
                    polarity = 'reported'
                findings.append(dict(term=term, polarity=polarity, evidence=span(text, match, clause.start())))
            qualifiers = {key: [span(text, m, clause.start()) for m in re.finditer(pattern, content)]
                          for key, pattern in QUALIFIERS.items()}
            clauses.append(dict(text=content, start=clause.start(), end=clause.end(),
                                findings=findings, qualifiers=qualifiers))
        records.append(dict(id=f'S{index+1}', raw=text, clauses=clauses,
                            coverage='bounded_lexicon_requires_review'))
    return records


def evidence_terms(records, ids=None):
    result = {}
    for record in records:
        if ids is not None and record['id'] not in ids:
            continue
        for clause in record['clauses']:
            for finding in clause['findings']:
                result.setdefault(finding['term'], set()).add(finding['polarity'])
    return result


def question_risks(question, records):
    """Checks each mentioned symptom's local prefix, not a global 'whether' escape."""
    known = evidence_terms(records)
    reasons = []
    for term, match in mentions(question):
        states = known.get(term, set())
        if 'reported' in states:
            prefix = re.split(r'[，,；;。？?]', question[:match.start()])[-1]
            source = ''.join(r['raw'] for r in records if any(f['term']==term for c in r['clauses'] for f in c['findings']))
            if not re.search(r'是否|有无|有没有|如果|若', prefix):
                for label, pattern in QUALIFIERS.items():
                    if label == 'time':
                        continue  # Time questions may ask for unreported onset.
                    for qualifier in re.finditer(pattern, prefix):
                        if qualifier.group() not in source:
                            reasons.append('question_presupposes_added_'+label+'_'+term)
            continue
        # A leading explicit inquiry/condition licenses an unknown symptom.
        # '胸痛是否持续' does not: 'whether' appears after the presupposed symptom.
        prefix = re.split(r'[，,；;。？?]', question[:match.start()])[-1]
        neutral = bool(re.search(r'是否(?:有|出现|存在|伴随|伴有)?[^，,；;。？?]*$|有无[^，,；;。？?]*$|有没有[^，,；;。？?]*$|如果[^，,；;。？?]*$|若[^，,；;。？?]*$', prefix))
        if neutral:
            continue
        if re.search(r'有|出现|持续|多久|几天|何时|频率|程度|加重|缓解|改善|部位|原因', question[match.start():]):
            reasons.append('question_presupposes_unconfirmed_'+term)
    return reasons


def explanation_risks(item, records):
    known = evidence_terms(records, item['basis_ids'])
    risks = []
    if known and all(states <= {'negative', 'resolved'} for states in known.values()):
        risks.append('negative_or_resolved_only_basis')
    for term, match in mentions(item['explanation']):
        if 'reported' not in known.get(term, set()):
            risks.append('explanation_unsupported_symptom_'+term)
    # Added severity/time/frequency should not be dressed up as a possible explanation.
    source = ''.join(r['raw'] for r in records if r['id'] in item['basis_ids'])
    for key, pattern in QUALIFIERS.items():
        for match in re.finditer(pattern, item['explanation']):
            if match.group() not in source and mentions(item['explanation']):
                risks.append('explanation_added_'+key)
    return risks
