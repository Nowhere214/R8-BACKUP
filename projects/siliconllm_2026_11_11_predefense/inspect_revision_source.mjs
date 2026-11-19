import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';

const root=path.dirname(fileURLToPath(import.meta.url));
const dir=path.join(root,'.build/revision_20261116');
await fs.mkdir(dir,{recursive:true});
const deck=await PresentationFile.importPptx(await FileBlob.load(path.join(root,'exports/Local_LLM_Agentic_Workloads_PreDefense_Draft_2026_11_11_Reviewed.pptx'))).catch(e=>{console.error(e.message);process.exit(1);});
const slides=await deck.inspect({select:'$.slides[*]',include:['id']});
const elements=await deck.inspect({select:'$.slides[*].elements[*]',include:['id','name','text','position','kind']});
await fs.writeFile(path.join(dir,'source_slides.json'),JSON.stringify(slides.records,null,2));
await fs.writeFile(path.join(dir,'source_elements.json'),JSON.stringify(elements.records,null,2));
console.log(JSON.stringify({slides:slides.recordCount,elements:elements.recordCount,first:elements.records.slice(0,7)},null,2));
