import {FileBlob,PresentationFile} from '@oai/artifact-tool';

const path='/Users/laiqixiao/Documents/SiliconLLM/projects/siliconllm_2026_11_11_predefense/exports/Local_LLM_Agentic_Workloads_PreDefense_Draft_2026_11_11_Reviewed.pptx';
export async function copySourceBatch({presentation,input}) {
  const source=await PresentationFile.importPptx(await FileBlob.load(path));
  if(input.start===0 && presentation.slides.items.length===1) presentation.slides.delete(0);
  for(let i=input.start;i<input.end;i++) presentation.slides.addCopy(source.slides.getItem(i).copy());
  await presentation.render({slide:[input.start]});
  return {slides:presentation.slides.items.length};
}
