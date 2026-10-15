# Guarded visual v3 (2026-10-15)

`guarded` now uses `cautious_chat_v3_20261015`. Same base BF16 weights;
no weight training. `raw` still calls the original model directly and is the
webpage and v1 API default. Historical v1/v2 source files remain unchanged.

## Cause and repair

V2 did not invoke vision inference: it returned resolution/chromatic-pixel
measurements only. Its face detector also rejected tongues and partial faces.
Chromatic pixels were not a rejection reason, but the response looked like one.

V3 invokes the image model for short observations and local limitations.
General near-black/near-white/tiny-image checks replace the frontal-face gate.
Colour, grayscale, absence of a face, and partial views do not themselves block
the image. A narrow item filter removes selected explicit medical conclusions;
it is not a complete semantic or medical validator. Invalid/truncated output
remains uncertain. A syntax-only repair accepts trailing JSON commas without
changing quoted text. Visual observations are not passed as diagnostic evidence
to the existing text alignment flow.

## Validation and limits

- Six focused local tests: colour/grayscale routing, black-image rejection,
  item-level medical filtering, invalid output, trailing-comma syntax repair,
  and text help routing. Seven existing text routing checks also pass with v3.
- Six live service cases: clear public photo, grayscale copy, partial crop,
  tongue image, black fixture, and synthetic symptoms plus the public photo.
  All returned HTTP 200. Four usable image-only cases retained observations;
  the black fixture abstained without inference; fusion retained the original
  symptom wording and source quotes. This measures response coverage only.
- Browser check: original mode remains selected by default; guarded v3 is
  selectable; image preview works; no JavaScript errors observed.

Manual review found remaining incorrect visual details: the public astronaut
photo response describes a white helmet faceplate/fixing strap that is not
clearly supported, and the partial crop response understates visible facial
features. No claim of hallucination elimination, reliable uncertainty detection,
clinical accuracy, or improved learned visual/diagnostic ability is warranted.
The change repairs excessive refusal while exposing the model's remaining limits.

Run `scripts/check_visual_v3.py` for focused regressions and
`scripts/check_visual_v3_live.py` against the running local tunnel for recorded
live checks. Image fixtures and detailed evaluation outputs stay outside Git.
