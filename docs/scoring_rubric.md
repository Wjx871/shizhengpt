# Human experimental scoring (1-5)

Scores are manual experimental review, never medical diagnostic accuracy. Leave unreviewed cells empty. Any later automated or assistant review must be labeled separately.

| Metric | 1 | 3 | 5 |
|---|---|---|---|
| Visual Correctness | Major visible features wrong | Mixed accurate and inaccurate features | Visible features accurately described, uncertainty calibrated |
| Groundedness | Invents facts or self-reported symptoms | Some unsupported inferences | All observations linked to images and all symptoms attributed to user |
| TCM Relevance | Irrelevant or conceptually wrong | Broad plausible theory with weak linkage | Theory relevant and clearly separated from observation and diagnosis |
| Consistency | Contradictory color/coating/pattern across repeats | Mostly stable with minor differences | Key judgments stable across three actual runs |
| Explainability | Unexplained conclusion | Partial evidence chain | Observations, uncertainty and follow-up questions clearly explained |

`hallucination` records suspected overdiagnosis, invented symptoms and unsupported visual abnormalities in words. It is not a medically validated binary label. Identical output across deterministic repeats is only repeatability, not correctness.

Lighting changes are measured within a single public photograph with known artificial pixel transforms. No calibrated reference skin color, verified health status or clinical diagnosis is available. A change in a visual color description can correctly reflect changed image pixels; the critical failure is interpreting that change as a different person's physiology or definite disease without accounting for capture conditions.

Vision + text ablation compares the same symptom prompt with no image, a face, two contrasting tongues and an unrelated image. Different output is evidence of image influence; it does not alone prove clinically valid multimodal reasoning.
