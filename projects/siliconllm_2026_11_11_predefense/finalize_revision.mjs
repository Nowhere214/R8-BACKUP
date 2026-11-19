import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';

const root=path.dirname(fileURLToPath(import.meta.url));
const build=path.join(root,'.build/revision_20261116');
const skill='/Users/laiqixiao/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11011/skills/presentations';
const python='/Users/laiqixiao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
process.env.RUNTIME_NODE_MODULES='/Users/laiqixiao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const presentation=Presentation.load(JSON.parse(await fs.readFile(path.join(build,'reviewed_state.json'),'utf8')));
if(presentation.slides.items.length!==31) throw new Error('Expected 31 reviewed slides');
// Session image resources need their original bytes for a standalone export.
for(const [number,key] of [[20,'overall'],[21,'repair'],[22,'agent'],[23,'native'],[28,'fixed_lengths'],[29,'feedback']]) {
  const record=await presentation.inspect({select:`$.slides[${number-1}].elements[?(@.kind == 'image')]`,include:['id']});
  if(record.recordCount!==1) throw new Error(`Expected one retained chart image on slide ${number}`);
  const image=presentation.resolve(record.records[0].id);
  const frame=image.frame,crop=image.crop;
  image.replace({blob:new Uint8Array(await fs.readFile(path.join(root,`images/${key}.png`))),
    contentType:'image/png',alt:image.alt,fit:image.fit??'contain'});
  image.frame=frame; image.crop=crop;
}
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidate);
for(const n of [7,15,19,20]) {
  const blob=await presentation.export({slide:presentation.slides.getItem(n-1),format:'png',scale:1});
  await fs.writeFile(path.join(build,`export-check-${n}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,
  finalPath:path.join(root,'exports/Local_LLM_PreDefense_Revised_2026_11_16_Complete.pptx'),
  explicitTotalSlideCount:31,requiredNativeTableOwnerSlides:[9,12,17,18,19],
  requiredNativeChartOwnerSlides:[],pythonExecutable:python,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
    ...[9,12,17,18,19].flatMap(n=>['--require-native-table-slide',String(n)])],
  fontPolicy:{basis:'reference',families:['Arial','Courier New'],
    referencePath:path.join(root,'exports/Local_LLM_Agentic_Workloads_PreDefense_Draft_2026_11_11_Reviewed.pptx'),
    referenceSha256:'a2c69c7596146d2519abdb04843c9e7a94fb48c4561e3f2b8b984d2b8ceecf1d'},
  verifyArtifactToolImport:true,receiptPath:path.join(build,'deck_complete.validation.json')});
console.log(JSON.stringify(result));
