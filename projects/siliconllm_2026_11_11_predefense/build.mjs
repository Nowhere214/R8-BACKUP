import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const root = path.dirname(fileURLToPath(import.meta.url));
const suffix = process.argv[2] ?? '';
const build = path.join(root, '.build');
const preview = path.join(build, 'previews');
const skill = '/Users/laiqixiao/.codex/plugins/cache/openai-primary-runtime/presentations/26.1110.11258/skills/presentations';
const python = '/Users/laiqixiao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
process.env.RUNTIME_NODE_MODULES = '/Users/laiqixiao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
await fs.mkdir(preview, { recursive: true });
const data = JSON.parse(await fs.readFile(path.join(build, 'slide_payload.json'), 'utf8'));
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const n = (a, key, fallback = 0) => Number(a[key] ?? fallback);

for (const page of data) {
  const slide = presentation.slides.add();
  slide.background.fill = '#F4EFFA';
  let index = 0;
  for (const item of page.elements) {
    const a = item.attrs;
    const name = `${item.type}-${++index}`;
    if (item.type === 'image') {
      slide.images.add({ blob: new Uint8Array(await fs.readFile(item.path)), contentType: 'image/png',
        alt: `Reported aggregate values: ${page.title}`, fit: 'contain',
        position: { left: n(a, 'x'), top: n(a, 'y'), width: n(a, 'width'), height: n(a, 'height') } });
    } else if (item.type === 'text') {
      const size = n(a, 'font-size', 24);
      const shape = slide.shapes.add({ geometry: 'textbox', name,
        position: { left: n(a, 'x'), top: n(a, 'y') - size, width: item.width + 6, height: size * 1.45 },
        fill: 'none', line: { fill: 'none', width: 0 } });
      shape.text = item.text;
      shape.text.style = { typeface: item.font, fontSize: size, bold: a['font-weight'] === '700',
        color: a.fill, insets: 0, wrap: 'none', autoFit: 'none', verticalAlignment: 'top' };
    } else if (item.type === 'line') {
      const x1 = n(a, 'x1'), x2 = n(a, 'x2'), y1 = n(a, 'y1'), y2 = n(a, 'y2');
      // Flips preserve arrowhead orientation without negative shape extents.
      slide.shapes.add({ geometry: 'line', name,
        position: { left: Math.min(x1, x2), top: Math.min(y1, y2),
          width: Math.abs(x2 - x1) || 0.1, height: Math.abs(y2 - y1) || 0.1,
          horizontalFlip: x2 < x1, verticalFlip: y2 < y1 },
        fill: 'none', line: { fill: a.stroke, width: n(a, 'stroke-width', 1) } });
    } else {
      slide.shapes.add({ geometry: 'rect', name,
        position: { left: n(a, 'x'), top: n(a, 'y'), width: n(a, 'width'), height: n(a, 'height') },
        fill: a.fill, line: { fill: a.stroke ?? 'none', width: a.stroke ? 1 : 0 } });
    }
  }
  slide.speakerNotes.text = page.notes;
  console.log(`Authored ${page.number}: ${page.title}`);
}

const candidate = path.join(build, 'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidate);
for (let i = 0; i < presentation.slides.items.length; i++) {
  const png = await presentation.export({ slide: presentation.slides.items[i], format: 'png', scale: 1.25 });
  await fs.writeFile(path.join(preview, `slide-${String(i + 1).padStart(2, '0')}.png`), new Uint8Array(await png.arrayBuffer()));
  console.log(`Rendered slide ${i + 1}`);
}
const { finalizePresentation } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const result = await finalizePresentation({ workspaceDir: root, candidatePath: candidate,
  finalPath: path.join(root, `exports/Local_LLM_Agentic_Workloads_PreDefense_Draft_2026_11_11${suffix}.pptx`),
  explicitTotalSlideCount: data.length, requiredNativeTableOwnerSlides: [], requiredNativeChartOwnerSlides: [],
  pythonExecutable: python,
  integrityValidatorPath: path.join(skill, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skill, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-bullet-geometry', '--validate-heading-fit'],
  fontPolicy: { basis: 'design', families: ['Arial', 'Courier New'] }, verifyArtifactToolImport: true,
  receiptPath: path.join(build, `deck${suffix}.validation.json`) });
console.log(JSON.stringify(result));
