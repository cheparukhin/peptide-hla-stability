// Build the illustrated peptide–HLA biology animation from its source.
// Run from the repository root: node src/animations/pmhc-intro/build_site_bundle.mjs
import { readFileSync, writeFileSync, mkdirSync, cpSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..', '..', '..');
const SOURCES = [join(HERE, 'animations-v3.jsx'), join(HERE, 'pmhc-scene-v2.jsx')];
const babelSrc = readFileSync(join(HERE, 'vendor', 'babel.js'), 'utf8');
new Function(babelSrc)();
const Babel = globalThis.Babel;
if (!Babel) throw new Error('could not load vendored Babel');

// Illustrative sequence timings, in seconds; not kinetic measurements.
const INTRO_OUT = join(REPO, 'site', 'intro');
const INTRO_CUES = [
  ['Groove', 1.3, 'HLA surface, zoom toward the groove'],
  ['Anatomy', 3.8, 'Groove anatomy: helices, floor, the nine residues'],
  ['Candidates', 2.1, 'Four candidate peptides cycle through the groove'],
  ['Lock', 1.0, 'A peptide binds in the HLA groove'],
  ['Approach', 3.2, 'The T-cell receptor descends onto the complex'],
  ['Interface', 2.1, 'V-domain ribbons; CDR loops read the peptide'],
  ['Wobble', 4.6, 'A loose peptide escapes, then a stable one settles'],
  ['Anchors', 2.1, 'Anchor residues seat in their pockets'],
  ['Contact', 3.6, 'TCR engagement can initiate signalling'],
  ['Activation', 1.5, 'Activation also depends on cellular context'],
  ['Expansion', 3.0, 'One clone expands out of the naive repertoire'],
  ['Body', 1.6, 'The whole-body response'],
];

rmSync(INTRO_OUT, { recursive: true, force: true });
mkdirSync(INTRO_OUT, { recursive: true });

// Only the runtime and the scene itself — none of the film's acts or shell.
const introChunks = SOURCES.map((src) =>
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
<title>Peptide–HLA recognition · Biological illustration</title>
<style>
  html, body { margin: 0; background: #14151A; overflow: hidden; }
</style>
</head>
<body>
<div class="context" style="position:fixed;top:12px;left:12px;z-index:999;background:white;padding:8px 12px;font:12px/1.5 system-ui;border-radius:4px"><a href="../" style="color:#0a6b6c">Results</a> · Biological illustration, not a molecular simulation</div>
<div id="root"></div>
<noscript><p style="color:white;margin:70px 20px">This animation requires JavaScript. The <a href="../" style="color:#b7e4dc">scientific results</a> do not.</p></noscript>
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
