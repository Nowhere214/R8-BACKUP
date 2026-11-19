import {drawGeneration,drawRecovery} from './procedure_supplement.mjs';
import {pages} from './revision_notes_data.mjs';

const BG='#F4EFFA', PURPLE='#3E1485', INK='#1F1633', GRAY='#655875';

function text(slide,name,label,x,y,width,size=26,bold=false,color=INK) {
  const s=slide.shapes.add({geometry:'textbox',name,
    position:{left:x,top:y,width,height:size*1.4},fill:'none',line:{fill:'none',width:0}});
  s.text=label;
  s.text.style={typeface:'Arial',fontSize:size,bold,color,insets:0,wrap:'none',autoFit:'none',verticalAlignment:'top'};
  return s;
}

function rect(slide,name,x,y,width,height,fill) {
  return slide.shapes.add({geometry:'rect',name,position:{left:x,top:y,width,height},fill,line:{fill:'none',width:0}});
}

async function clearBody(presentation,index) {
  const result=await presentation.inspect({select:`$.slides[${index}].elements[*]`,include:['id','position']});
  const ids=result.records.filter(r=>r.position.top>=175&&r.position.top<657).map(r=>r.id);
  if(ids.length) presentation.delete(ids);
  return presentation.slides.getItem(index);
}

function table(slide,name,values,y,height,labelWidth=340,size=24) {
  const t=slide.tables.add({rows:values.length,columns:2,left:56,top:y,width:1170,height,
    columnWidths:[labelWidth,1170-labelWidth],values});
  t.borders.assign({fill:'#D6CBE8',width:.7,style:'solid'});
  for(let r=0;r<values.length;r++) for(let c=0;c<2;c++) {
    const cell=t.getCell(r,c);
    cell.fill=r===0?'#E5DDEE':BG;
    cell.text.style={typeface:'Arial',fontSize:size,bold:r===0,color:INK,
      insets:{left:13,right:13,top:8,bottom:8},verticalAlignment:'middle',autoFit:'none'};
  }
  return t;
}

function frame(slide,title,section) {
  slide.background.fill=BG;
  rect(slide,'header',0,0,1280,118,PURPLE);
  rect(slide,'page-tab',1178,0,102,118,'#9E89C2');
  text(slide,'title',title,52,36,1090,34,true,'#FFFFFF');
  text(slide,'page','19',1200,37,65,38,true,'#FFFFFF');
  text(slide,'section',section,56,136,1100,17,true,'#6B4CA3');
  rect(slide,'footer-rule',54,657,1172,1,'#D6CBE8');
}

export async function methodRevision({presentation}) {
  if(presentation.slides.items.length!==28) throw new Error('Expected the unchanged 28-slide source');
  let s=await clearBody(presentation,3);
  text(s,'problem-label','Problem',56,195,1140,25,true,PURPLE);
  text(s,'problem-1','Repair can regenerate code that already exists in the history.',56,238,1165,29);
  text(s,'objective-label','Objective',56,326,1140,25,true,PURPLE);
  text(s,'objective-1','Reduce local inference time while preserving the ordinary greedy output.',56,369,1165,28);
  text(s,'question-label','Research question',56,456,1140,25,true,PURPLE);
  text(s,'question-1','Can a time-based choice of whether and how much to reuse',56,499,1165,29);
  text(s,'question-2','outperform a fixed historical draft length?',56,542,1165,29);
  text(s,'objective-check','Check model latency and agent latency. Check output and task results separately.',56,615,1165,23);

  s=await clearBody(presentation,7);
  table(s,'related-methods',[
    ['Earlier work','Related idea'],
    ['Prompt Lookup [2]','Retrieve continuations from existing text'],
    ['EfficientEdit [3]','Code reuse with edit-oriented drafting'],
    ['AgSpec [4]','Agent history and adaptive draft length'],
    ['DISCO [5]','Dynamic speculation lookahead']
  ],205,285,330,24);
  text(s,'specific-rule','Our proposed part: a concrete time-based rule for local history drafts.',56,523,1165,26,true);
  text(s,'specific-input','Step and verification timings, recent acceptance, and an ordinary-decoding option.',56,568,1165,24);
  text(s,'novelty-scope','History reuse and adaptive length are existing ideas. We evaluate this selection rule.',56,615,1165,23);

  s=await clearBody(presentation,8);
  text(s,'target-label','Target model',56,195,1140,25,true,PURPLE);
  text(s,'target-model','Qwen2-1.5B-Instruct, 4-bit. We keep the model weights unchanged.',56,238,1165,28);
  text(s,'draft-label','Draft source',56,323,1140,25,true,PURPLE);
  text(s,'draft-source','Earlier generated code provides candidate tokens through history lookup.',56,366,1165,28);
  text(s,'controller-label','Controller design',56,451,1140,25,true,PURPLE);
  text(s,'controller-1','Score ordinary decoding and feasible lengths k = 2, 4, 8, 16 by time per token.',56,494,1165,27);
  text(s,'controller-2','Choose reuse only when the estimated saving exceeds the policy margin.',56,538,1165,27);
  text(s,'model-scope','We design the decoding controller. We do not train a new language model.',56,615,1165,24);

  s=await clearBody(presentation,10);
  text(s,'score-rationale','One round may cost more time but produce several output tokens.',56,195,1165,28);
  text(s,'score-equation','C(k) = [ V(k) + L + R(k) ] / G(k)',68,266,1145,37,true,PURPLE);
  table(s,'cost-terms',[
    ['Term','Meaning'],
    ['V(k)','Verification time, including KV-cache use'],
    ['L','History lookup time'],
    ['R(k)','Expected extra replay time in the current local path'],
    ['G(k)','Expected accepted prefix length plus one target token']
  ],332,212,180,22);
  text(s,'cost-reason','Compare time per round with expected output progress.',56,569,1165,25);
  text(s,'replay-scope','R(k) is path-specific. mlx-lm can trim the cache instead. See the next recovery pages.',56,615,1165,23);

  s=await clearBody(presentation,12);
  text(s,'snapshot-label','Before verification',56,195,1140,25,true,PURPLE);
  text(s,'snapshot-1','Keep a snapshot of the valid KV cache.',56,238,1165,29);
  text(s,'replay-label','After a rejected draft',56,323,1140,25,true,PURPLE);
  text(s,'replay-1','Restore the snapshot and replay the retained prefix with ordinary steps.',56,366,1165,28);
  text(s,'replay-reason-label','Why this path uses replay',56,451,1140,25,true,PURPLE);
  text(s,'replay-reason','Earlier checks found numerical cache differences after batch verification.',56,494,1165,28);
  text(s,'replay-status','Replay addresses that implementation issue. Its necessity still needs checking.',56,538,1165,27);
  text(s,'replay-bridge','The next slide shows the different cache-trim procedure in mlx-lm.',56,615,1165,24);

  let added=presentation.slides.add({width:1280,height:720});
  drawGeneration(added); added.moveTo(6);
  added=presentation.slides.add({width:1280,height:720});
  drawRecovery(added); added.moveTo(14);
  await presentation.render({slide:[3,6,8,9,11,13,14]});
  return {slides:presentation.slides.items.length};
}

export async function evaluationRevision({presentation}) {
  if(presentation.slides.items.length!==30) throw new Error('Expected 30 slides after method revision');
  let s=await clearBody(presentation,16);
  table(s,'experiment-settings',[
    ['Item','Configuration'],
    ['Target model','Qwen2-1.5B-Instruct, 4-bit, group size 64'],
    ['Hardware','Apple Silicon, 18 GiB unified memory'],
    ['Main engine','SiliconLLM synchronous loop [9]'],
    ['Tasks','MBPP sanitized [7] and HumanEval [8]'],
    ['Replay / agent timing','30 tasks, 64 requests / 20 tasks. Three repeats.'],
    ['Generation / repair','Greedy. 384 tokens per call. Up to two repairs.']
  ],195,350,325,23);
  text(s,'test-visibility','Repairs use visible tests. Final scoring tests never enter the prompt.',56,575,1165,24);
  text(s,'engine-scope','Native mlx-lm components and the full stock API are separate comparisons.',56,615,1165,23);

  s=await clearBody(presentation,17);
  table(s,'controlled-baselines',[
    ['Method','Next decoding action'],
    ['Ordinary KV','One ordinary greedy step'],
    ['Fixed history','Use available history, up to k = 16'],
    ['Adaptive history','Choose k = 0, 2, 4, 8 or 16']
  ],205,260,345,25);
  text(s,'matched-controls','Same model, prompts, history source, verifier and recovery path.',56,501,1165,26);
  text(s,'baseline-scope','These are in-project controls, not full reproductions of AgSpec or other systems.',56,550,1165,24);
  text(s,'stock-pending','A full stock mlx-lm API baseline is planned. Its timing is still pending.',56,610,1165,24);

  const metrics=presentation.slides.add({width:1280,height:720});
  frame(metrics,'Evaluation Metrics','EVALUATION');
  table(metrics,'evaluation-metrics',[
    ['Metric','What we compare'],
    ['Model wall time','Timed model requests, including decoding overhead'],
    ['Agent wall time','Generation, visible tests, feedback and logging'],
    ['Speedup','Baseline time / method time for the same workload'],
    ['Output match','Full token sequence compared with ordinary decoding'],
    ['Task success','Passed tasks / evaluated tasks on final scoring tests']
  ],205,342,315,23);
  text(metrics,'timing-stages','Report overall and repair-only timings separately.',56,578,1165,24);
  text(metrics,'agent-exclusions','Agent timing excludes model loading, warmup and final held-out scoring.',56,620,1165,22);
  metrics.speakerNotes.text='Metric definitions. Speedup uses the same task set and timing boundary. Task success is a check on retained outcomes, not a claimed improvement. Source aggregates are external reported values. Exact model-request timing boundaries require the original run logs before final thesis claims.';
  metrics.moveTo(18);

  const sourceNotes=[];
  for(let i=0;i<31;i++) {
    const slide=presentation.slides.getItem(i);
    sourceNotes.push(slide.speakerNotes.text);
    const r=await presentation.inspect({select:`$.slides[${i}].elements[*]`,include:['id','text','position','name']});
    for(const e of r.records) {
      if(!e.text) continue;
      const shape=presentation.resolve(e.id);
      if(e.position.top<118 && /^\d+$/.test(e.text)) {
        shape.text=String(i+1);
        shape.position={left:i+1<10?1208:1200,top:37,width:65,height:55.1};
      } else if(/^\d\d \/ 28$/.test(e.text)) {
        shape.text=`${String(i+1).padStart(2,'0')} / 31`;
      } else if(e.text.startsWith('Working draft - numerical placeholders')) {
        shape.text='Working draft - numerical placeholders on slides 13, 16 and 24';
        shape.position={...e.position,width:1020};
      } else if(e.name==='source') {
        presentation.delete(e.id);
      } else if(e.text==='November 11, 2026') {
        shape.text='November 16, 2026';
      }
    }
    if(pages[i].new) {
      text(slide,'working-footer','Working draft - numerical placeholders on slides 13, 16 and 24',56,674,1020,14,false,'#6F6280');
      text(slide,'number-footer',`${String(i+1).padStart(2,'0')} / 31`,1137,671,88,17,false,'#6F6280');
    }
    const p=pages[i];
    slide.speakerNotes.text=`English\n${p.en}\n\n中文\n${p.zh}\n\nTechnical notes and sources\n${sourceNotes[i]}`;
  }
  await presentation.render({slide:[0,16,17,18,23]});
  await presentation.inspect({select:'$.slides[*]',include:['id','index','elements.id','elements.name','elements.text','elements.kind','elements.position','notes.text'],fileName:'revised-content.ndjson'});
  return {slides:presentation.slides.items.length,main:26,placeholders:[13,16,24]};
}

export async function renderAll({presentation}) {
  await presentation.render();
  return {slides:presentation.slides.items.length};
}
