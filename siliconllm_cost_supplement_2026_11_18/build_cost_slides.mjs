import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {data} from './cost_data.mjs';

const root=path.dirname(fileURLToPath(import.meta.url));
const build=path.join(root,'.build');
const skill='/Users/laiqixiao/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11011/skills/presentations';
const python='/Users/laiqixiao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const reference=path.join(root,'../siliconllm_2026_11_11_predefense/exports/Local_LLM_PreDefense_Revised_2026_11_16_Complete.pptx');
process.env.RUNTIME_NODE_MODULES='/Users/laiqixiao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const BG='#F4EFFA',PURPLE='#3E1485',INK='#29252F',GRAY='#696270',RULE='#D6CBE8';
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
const f=x=>x.toFixed(2);

function rect(s,name,x,y,w,h,fill) {
  return s.shapes.add({geometry:'rect',name,position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});
}
function text(s,name,label,x,y,w,h=42,size=26,bold=false,color=INK) {
  const q=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  q.text=label;
  q.text.style={typeface:'Arial',fontSize:size,bold,color,insets:0,wrap:'none',autoFit:'none',verticalAlignment:'middle'};
  return q;
}
function slide(title,status,footnote) {
  const s=presentation.slides.add({width:1280,height:720});s.background.fill=BG;
  rect(s,'title-band',0,0,1280,118,PURPLE);
  text(s,'title',title,52,31,1170,55,34,true,'#FFFFFF');
  text(s,'status',status,56,135,1170,30,17,true,PURPLE);
  rect(s,'footer-rule',54,657,1172,1.2,RULE);
  text(s,'footnote',footnote,56,670,1170,30,16,false,GRAY);
  return s;
}
function table(s,name,values,y,rowH,widths,font=26,highlight=-1) {
  const headerH=values[0].some(v=>String(v).includes('\n'))?68:48;
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:56,top:y,width:1170,height:headerH+rowH*(values.length-1),columnWidths:widths,values});
  t.name=name;
  t.borders.assign({fill:'none',width:0});
  for(let r=0;r<values.length;r++) {
    t.rows[r].height=r===0?headerH:rowH;
    for(let c=0;c<values[r].length;c++) {
      const cell=t.getCell(r,c);cell.fill=r===highlight?'#E7DDF2':BG;
      cell.text.style={typeface:'Arial',fontSize:r===0?font-2:font,bold:r===0||r===highlight,color:r===highlight?PURPLE:INK,
        alignment:'center',verticalAlignment:'middle',insets:{left:8,right:8,top:2,bottom:2},wrap:'none',autoFit:'none'};
    }
    if(r<values.length-1) rect(s,`${name}-rule-${r}`,56,y+headerH+r*rowH,1170,1.2,r===0?'#B9AEC9':'#DED7E7');
  }
  return t;
}
const source='Baseline scale selected by the user: 18.90 ms. The archived calibration gives 18.900229 ms and belongs to the withdrawn October local study. It is a historical illustrative reference, not reinstated benchmark evidence. Ratios 1.10/1.22/1.48/2.05 and lookup ratio .02 are from the existing illustrative_placeholders.json, not measured V(k). No new inference, API calibration or controller experiment was run. Full-precision inputs are retained in cost_example.json.';

let s=slide('Verification Time V(k)','ILLUSTRATIVE SCALING, NOT NEW MEASUREMENTS',
  'Historical reference scale only. The k > 0 values are illustrative, not full mlx-lm API measurements.');
text(s,'baseline','Ordinary-step reference: T_step = 18.90 ms',56,174,1170,43,28,true,PURPLE);
table(s,'verification-table',[
  ['Draft length k','Input tokens\nper forward','Forward time\n(ms)','Time / ordinary\nstep'],
  ...data.rows.map(r=>[r.k===0?'0 (ordinary)':String(r.k),String(r.input_tokens),r.v_display,r.ratio_display.replace('x','\u00d7')]),
],231,59,[245,340,260,325],26);
text(s,'k-plus-one','A draft of k tokens adds one pending input token, so the target sees k + 1.',56,602,1170,37,24);
text(s,'ratio-meaning','The last column is a forward-cost ratio, not a decoding speedup.',56,631,1170,24,20,false,GRAY);
s.speakerNotes.text=`English\nAn ordinary step takes 18.90 milliseconds on our chosen historical scale. With k draft tokens, one target forward receives k plus one input tokens. The other rows use assumed cost ratios. They explain the cost model and are not new measurements. The last column compares forward costs, not decoding speed.\n\n中文\n这里以历史记录中的普通单步18.90毫秒作为说明标尺。草稿有k个token时，一次目标前向处理k加1个输入token。其余行由假设成本比例换算，用来解释成本模型，不是新测量。最后一列比较的是前向成本，不是解码加速比。\n\nSources and status\n${source}\nArchived record: benches/agent_study_20261021_main/calibration.json. Assumed ratios: projects/siliconllm_2026_11_11_predefense/sources/illustrative_placeholders.json. Historical measured V values in that archive are 19.53/24.60/36.78/43.08 ms, not the illustrated values. They are not substituted into current benchmark claims.`;

s=slide('Expected Recovery Cost R(k)','REPLAY PATH EXAMPLE',
  'Assumed replay cost. A native trim path requires its own cache-handling cost.');
text(s,'rejection-example','Draft A B C D: accept A and B, reject C.',56,176,1170,43,29,true,PURPLE);
text(s,'restore','Restore the pre-verification KV cache, then process x, A and B.',56,221,1170,37,25);
const rt=table(s,'replay-steps',[
  ['Step','Ordinary input','Predicted token'],
  ['1','x','A'],['2','A','B'],['3','B','Z (replaces C)'],
],274,45,[170,450,550],26);
rt.getCell(3,2).text.style={typeface:'Arial',fontSize:26,color:'#A33636',alignment:'center',
  verticalAlignment:'middle',insets:{left:8,right:8,top:7,bottom:7},wrap:'none',autoFit:'none'};
text(s,'single-rejection-cost','This rejection: 3 \u00d7 18.90 = 56.70 ms of replay forwards.',56,478,1170,43,28,true,PURPLE);
text(s,'expected-cost','With assumed p = 0.98 and k = 8: R(8) \u2248 12.39 ms per round.',56,534,1170,42,27);
text(s,'expectation-explained','R(k) averages all rejection positions, with zero replay for full acceptance.',56,580,1170,37,24);
text(s,'next-input','Z is the next normal input. Its next forward is not another replay step.',56,620,1170,29,22,false,GRAY);
s.speakerNotes.text=`English\nThis example uses the replay path. After rejection, we restore the earlier cache. Input x predicts A, input A predicts B, and input B predicts Z. That is three ordinary forwards, estimated at 56.70 milliseconds. R(k) is different: it averages the replay cost across possible rejection positions. For assumed p of 0.98 and k of eight, it is about 12.39 milliseconds per round. Full acceptance adds no replay in this example. A trim path needs a different recovery model.\n\n中文\n这个例子对应replay路径。发生拒绝后，先恢复验证前的缓存。输入x预测A，输入A预测B，输入B预测Z，共三次普通前向，按56.70毫秒估算。R(k)不是这一次拒绝的耗时，而是对各个拒绝位置的重放开销加权后的每轮期望值。假设p为0.98、k为8时，约为每轮12.39毫秒。这个算例中，全接受不增加重放。如果使用trim，就需要改用实际缓存裁剪的成本模型。\n\nAssumptions and source\n${source}\nR(k) = T_step * sum((j+1)*p**j*(1-p), j=0..k-1). This assumes constant independent match probability, no EOS/budget truncation, j+1 replay forwards on rejection and no replay on full acceptance. Snapshot/restore overhead and other missing costs need separate accounting. The four symbolic tokens illustrate j=2; the separate expected-value calculation uses k=8. This is not a claim that mlx-lm native trimming replays an accepted prefix.`;

s=slide('Time per Output Token C(k)','ILLUSTRATIVE REPLAY SCORE, NOT A MEASURED CONTROLLER RESULT',
  'The table uses unrounded inputs. Displayed values are rounded. The minimum is specific to these assumptions.');
text(s,'assumptions','Assume p = 0.98 and L = 0.02 \u00d7 18.90 = 0.378 ms.',56,173,1170,42,27);
table(s,'score-table',[
  ['k','V(k)\n(ms)','L\n(ms)','R(k)\n(ms)','G(k)\n(tokens)','C(k)\n(ms/token)'],
  ...data.rows.map(r=>[String(r.k),r.v_display,f(r.l_ms),f(r.r_ms),f(r.g_tokens),f(r.c_ms_per_token)]),
],218,46,[100,200,170,200,200,300],25,4);
text(s,'formula','C(k) = [ V(k) + L + R(k) ] / G(k)',56,536,1170,43,30,true,PURPLE);
text(s,'worked-example','C(8) = (27.97 + 0.38 + 12.39) / 8.31 \u2248 4.90 ms/token',56,580,1170,41,27);
text(s,'winner','k = 8 has the lowest score in this example.',56,621,1170,30,25,true,PURPLE);
s.speakerNotes.text=`English\nWe add verification time, lookup time and expected replay time. We divide this by the expected number of output tokens. For k of eight, the score is about 4.90 milliseconds per token. For sixteen, it is about 5.59. Eight has the lowest score in this example and passes the five-percent margin against the ordinary reference. This explains the decision rule. It is not a measured speedup or a decision observed in the API pilot.\n\n中文\n先把验证、查找和预期重放耗时相加，再除以预计输出token数。k为8时，分数约为每token 4.90毫秒；k为16时约为5.59。这个算例中，8的分数最低，也满足相对普通基线预留5%收益的条件。这里说明的是选择规则，不是测出来的加速比，也不是API pilot中实际观察到的动态决策。\n\nCalculation and status\n${source}\nG(k)=1+sum(p**j,j=1..k). L=.378 ms. k=0 is the separate ordinary reference without drafting lookup/replay. R(8)=${data.rows[3].r_ms}; G(8)=${data.rows[3].g_tokens}; C(8)=${data.rows[3].c_ms_per_token}; C(16)=${data.rows[4].c_ms_per_token}. C(8) < .95*18.90 ms. The score does not include every possible policy/cache overhead and is not a calibrated native trim model.`;

await fs.mkdir(build,{recursive:true});
for(let i=0;i<3;i++) {
  const image=await presentation.export({slide:presentation.slides.getItem(i),format:'png',scale:1});
  await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await image.arrayBuffer()));
}
const inspected=await presentation.inspect({select:'$.slides[*]',include:['index','title','elements.kind','elements.text','notes.text']});
await fs.writeFile(path.join(build,'content.ndjson'),inspected.ndjson);
await (await PresentationFile.exportPptx(presentation)).save(path.join(build,'candidate.pptx'));
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({workspaceDir:root,candidatePath:path.join(build,'candidate.pptx'),
  finalPath:path.join(root,'exports/Local_LLM_Cost_Explanation_18p90ms_2026_11_18.pptx'),
  explicitTotalSlideCount:3,requiredNativeTableOwnerSlides:[1,2,3],requiredNativeChartOwnerSlides:[],
  pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',
    ...[1,2,3].flatMap(n=>['--require-native-table-slide',String(n)])],
  fontPolicy:{basis:'reference',families:['Arial'],referencePath:reference,
    referenceSha256:crypto.createHash('sha256').update(await fs.readFile(reference)).digest('hex')},
  verifyArtifactToolImport:true,receiptPath:path.join(build,'cost_slides_reviewed.validation.json')});
console.log(JSON.stringify(result));
