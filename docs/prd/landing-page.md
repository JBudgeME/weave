# Issue: public landing page (prd:landing-page, T1)

## SOP exception

Local issue file instead of GH issue — standing no-remote-for-workflow
authorization (the public repo excludes workflow artifacts; PRDs stay
in-repo but this one predates its own merge).

## Brief (Gate 1, approved 2026-07-14 "sounds amazing")

Now that weave is public (github.com/JBudgeME/weave, weave-kohl.vercel.app),
the page should present the proof of concept, not just the demo. Layout A
from grilling: hero → app centerpiece (reduced width) → info sections.

- Hero: name, tagline, GitHub link button, badges (MIT / CI / WebGPU).
- App demo at ~2/3 width, still the interactive star.
- Sections: What is this · How it works (pipeline) · By the numbers
  (26M params, 100%/100%, ~350ms median, 52MB, no server) · Quick start ·
  footer with license + Needle/json-render credits.
- Standard OSS-landing conventions; dark/light themes respected.
- Out of scope: model/logic changes, analytics, heavy animation.
- Success: lint/tsc/tests/build green; browser check desktop + narrow
  viewport; user approves preview; redeployed to Vercel and verified live.

## Slices

- [x] S1: layout + hero + sections on `app/page.tsx` (+ small components),
      preview approved locally
- [x] S2: redeploy + live verify + repo/README touch-ups if needed
