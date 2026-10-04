// Builds the submission film into site/film/ as a self-contained static page.
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
