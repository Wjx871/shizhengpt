# Chat experience and guarded v4 — 2026-10-15

The browser now has a responsive history sidebar, search, rename/delete,
conversation switching, image history, copy-answer action, animated waiting
indicator with elapsed time, and progressive display of the final answer.
History is stored in this browser's IndexedDB; there is no account or device sync.
Storage failures are surfaced. New conversations preserve previous records.

Progressive display is a browser animation after inference/checks finish, not
token streaming from the model. `/v1/models.streaming` remains `false`.
Reduced-motion preferences skip animation; “立即显示全文” also finishes it.
Waiting text is process status, not a transcript of model reasoning.

`POST /api/restore` resumes a browser-owned record, reusing a live server session
or reconstructing an expired one. It accepts `{session_id?, mode, messages}`;
messages alternate user/assistant with `text` and optional user `image` Base64
or image Data URL. It validates roles, length, image format/size, and at most
15 complete turns. It cannot introduce system messages. Reconstruction uses
temporary image files and rolls back on validation failure. To accommodate
multiple image turns, this route allows up to 170 MiB; normal chat requests
retain the existing 12 MiB limit. Conversations still support 15 replies.

## Model-side scope

`guarded` is now `cautious_chat_v4_20261015`, using unchanged BF16 base weights.
`raw` remains the original path and default for a fresh browser and the v1 API.
Existing v1/v2/v3 files are preserved.

V4 keeps the v3 visual inference prompt and removes risky detail clauses from
individual observations. It retains other basic observations from the same
item, rather than rejecting the whole image. Restricted details include material,
small writing/labels/accessories, blanket absence/normality claims, and assertions
that only one region is visible. Unchecked visibility-limit prose is omitted.
This is a rule-based restriction on output detail, not verification against the
image, retraining, or proof of improved diagnosis.

Exploratory same-model review retained known wrong claims in four development
cases; a more restrictive prompt also produced new unsupported details. Neither
approach was deployed. The final path adds no extra visual model invocation.

## Evidence and remaining limitations

- Local checks: 3 clause/scope, 4 restoration, 6 existing API contract, and
  7 existing text-routing checks passed using v4.
- Simulated UI lifecycle: history persistence/reload/restore, search/delete,
  mode switching, waiting animation, progressive text, HTML sanitization and
  mobile layout passed. These tests do not measure model quality.
- Real deployed browser: restored an explicitly deleted server session and
  recalled its original test identifier; sent a real image; persisted its image
  and answer across reload; no JavaScript errors observed.
- Existing live API regression: 22 HTTP checks passed, including raw text/image/fusion, sessions, errors and concurrency.
- Six live visual/fusion development cases returned HTTP 200: clear, grayscale,
  partial crop, tongue, black, and synthetic symptoms plus public image.
  The four usable image-only inputs retained concrete observations; black
  abstained; fusion retained the symptom wording and source quotes.
- On the public photo, v3's unsupported white faceplate/fixing-strap detail was
  removed while clothing colour, background objects and helmet colour remained.
  On the partial crop, the wrong “only forehead and eyebrows” restriction was
  removed while the local face description remained.

These are development examples, not an independent clinical benchmark. The
rules can remove correct details and miss other hallucinations. The visual
model can still omit visible features or misidentify objects/colours; it cannot
be described as reliably knowing when it is wrong. No clinical-accuracy gain
or zero-hallucination claim is supported. Local image assets and full evaluation
outputs remain outside Git.
