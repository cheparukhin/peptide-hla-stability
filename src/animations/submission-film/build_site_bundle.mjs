// Builds two static pages into site/: the submission film (film/) and the
// original 30-second pMHC interface animation (intro/).
//
// The dev page (index.html) fetches each .jsx and transforms it in the browser
// with Babel standalone. That costs a 3 MB download and a visible transform
// pause on every load, which is the wrong trade for a published page. Here we
// run the same vendored Babel in Node at build time and ship plain JS.
//
// The sources share one scope (no import/export), and `const` declared inside
// an eval is scoped to that eval — so the browser must evaluate ONE combined
// script, not one per file. We concatenate in dependency order and emit a
// single bundle.
//
//   node src/animations/submission-film/build_site_bundle.mjs

import { readFileSync, writeFileSync, mkdirSync, cpSync, rmSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..', '..', '..');
const OUT = join(REPO, 'site', 'film');

// Dependency order: runtime first, then the intro scene whose helpers every act
// reuses (C, FONT, MOTION, tw, lerp, Frame, Beat*), then the acts, then the shell.
const SOURCES = [
  join(REPO, 'src/animations/pmhc-intro/animations-v3.jsx'),
  join(REPO, 'src/animations/pmhc-intro/pmhc-scene-v2.jsx'),
  join(HERE, 'act0-coldopen.jsx'),
  join(HERE, 'act2-approach.jsx'),
  join(HERE, 'act5-verdict.jsx'),
  join(HERE, 'act6-transfer.jsx'),
  join(HERE, 'act7-next.jsx'),
  join(HERE, 'film-scene.jsx'),
];

// Load the vendored Babel standalone into this Node process.
// Babel standalone is a UMD bundle: with no `exports` and no AMD `define` in
// scope it attaches itself to globalThis, so that is where we read it back.
const babelSrc = readFileSync(join(HERE, 'vendor', 'babel.js'), 'utf8');
new Function(babelSrc)();
const Babel = globalThis.Babel;
if (!Babel) throw new Error('could not load vendored Babel');

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

const chunks = [];
for (const src of SOURCES) {
  if (!existsSync(src)) throw new Error('missing source: ' + src);
  const code = Babel.transform(readFileSync(src, 'utf8'), { presets: ['react'] }).code;
  chunks.push(`/* ${src.replace(REPO + '/', '')} */\n${code}`);
}
writeFileSync(join(OUT, 'film.js'), chunks.join('\n;\n'), 'utf8');

// Assets the page needs at runtime.
cpSync(join(REPO, 'src/animations/pmhc-intro/shots'), join(OUT, 'shots'), { recursive: true });
cpSync(join(HERE, 'film_data.js'), join(OUT, 'film_data.js'));
mkdirSync(join(OUT, 'vendor'), { recursive: true });
for (const f of ['react.js', 'react-dom.js']) {
  cpSync(join(HERE, 'vendor', f), join(OUT, 'vendor', f));
}

// The scene list, copied from index.html so the published page and the dev page
// stay one source of truth for timing.
const OM_SCENES = readFileSync(join(HERE, 'index.html'), 'utf8')
  .match(/window\.OM_SCENES = JSON\.stringify\((\[[\s\S]*?\])\);/);
if (!OM_SCENES) throw new Error('could not read OM_SCENES out of index.html');

writeFileSync(join(OUT, 'index.html'), `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Sequence Is All You Need?</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
<style>
  html, body { margin: 0; background: #14151A; overflow: hidden; }
</style>
</head>
<body>
<div id="root"></div>
<script src="vendor/react.js"></script>
<script src="vendor/react-dom.js"></script>
<script src="film_data.js"></script>
<script>
  window.OM_SCENES = JSON.stringify(${OM_SCENES[1]});
  window.OM_PLAYBACK = '{"mode":"loop"}';
  window.TWEAK_DEFAULTS = { labels: true, captions: true, motionEditor: false };

  // The published page has no tweaks harness, so the scene's panel is stubbed
  // out rather than rendered.
  window.useTweaks = function (d) { return [d || {}, function () {}]; };
  window.TweaksPanel = function () { return null; };
  window.TweakSection = function () { return null; };
  window.TweakToggle = function () { return null; };

  var q = new URLSearchParams(location.search);
  var t = parseFloat(q.get('t'));
  window.__FILM_T__ = Number.isFinite(t) ? t : null;
  window.__FILM_PANEL__ = false;
</script>
<script src="film.js"></script>
<script>
  ReactDOM.createRoot(document.getElementById('root'))
    .render(React.createElement(window.SubmissionFilm));
</script>
</body>
</html>
`, 'utf8');

console.log('built ' + OUT);

// ---------------------------------------------------------------------------
// The original pMHC interface animation, as a live page rather than a video.
//
// Its scene root is window.PmhcVideoV2 and it reads its timing from CUES, so
// it needs its own OM_SCENES. The original cue sheet is NOT in the repo — it
// lived in the artifact host page — so these durations are derived from the
// beat code's own lower bounds (the frame at which each beat's last callout,
// pop or ripple finishes). They sum to 29.9s, which matches site/intro.mp4
// exactly, so the pacing reproduces the original render rather than guessing.
const INTRO_OUT = join(REPO, 'site', 'intro');
const INTRO_CUES = [
  ['Groove', 1.3, 'HLA surface, zoom toward the groove'],
  ['Anatomy', 3.8, 'Groove anatomy: helices, floor, the nine residues'],
  ['Candidates', 2.1, 'Four candidate peptides cycle through the groove'],
  ['Lock', 1.0, 'One peptide forms a stable pair'],
  ['Approach', 3.2, 'The T-cell receptor descends onto the complex'],
  ['Interface', 2.1, 'V-domain ribbons; CDR loops read the peptide'],
  ['Wobble', 4.6, 'A loose peptide escapes, then a stable one settles'],
  ['Anchors', 2.1, 'Anchor residues seat in their pockets'],
  ['Contact', 3.6, 'CDR3 contacts the peptide and the receptor signals'],
  ['Activation', 1.5, 'The T cell activates'],
  ['Expansion', 3.0, 'One clone expands out of the naive repertoire'],
  ['Body', 1.6, 'The whole-body response'],
];

rmSync(INTRO_OUT, { recursive: true, force: true });
mkdirSync(INTRO_OUT, { recursive: true });

// Only the runtime and the scene itself — none of the film's acts or shell.
const introChunks = SOURCES.slice(0, 2).map((src) =>
  `/* ${src.replace(REPO + '/', '')} */\n` +
  Babel.transform(readFileSync(src, 'utf8'), { presets: ['react'] }).code);
writeFileSync(join(INTRO_OUT, 'intro.js'), introChunks.join('\n;\n'), 'utf8');

cpSync(join(REPO, 'src/animations/pmhc-intro/shots'), join(INTRO_OUT, 'shots'), { recursive: true });
mkdirSync(join(INTRO_OUT, 'vendor'), { recursive: true });
for (const f of ['react.js', 'react-dom.js']) {
  cpSync(join(HERE, 'vendor', f), join(INTRO_OUT, 'vendor', f));
}

writeFileSync(join(INTRO_OUT, 'index.html'), `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Peptide-MHC interface</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
<style>
  html, body { margin: 0; background: #14151A; overflow: hidden; }
</style>
</head>
<body>
<div id="root"></div>
<script src="vendor/react.js"></script>
<script src="vendor/react-dom.js"></script>
<script>
  window.OM_SCENES = JSON.stringify(${JSON.stringify(
    INTRO_CUES.map(([name, dur, desc]) => ({ name, dur, desc })), null, 2)});
  window.OM_PLAYBACK = '{"mode":"loop"}';
  window.TWEAK_DEFAULTS = { labels: true, captions: true, motionEditor: false };

  // Supplied by the artifact harness in development; stubbed here.
  window.useTweaks = function (d) { return [d || {}, function () {}]; };
  window.TweaksPanel = function () { return null; };
  window.TweakSection = function () { return null; };
  window.TweakToggle = function () { return null; };
</script>
<script src="intro.js"></script>
<script>
  ReactDOM.createRoot(document.getElementById('root'))
    .render(React.createElement(window.PmhcVideoV2));
</script>
</body>
</html>
`, 'utf8');

console.log('built ' + INTRO_OUT);
