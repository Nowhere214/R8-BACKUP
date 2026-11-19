const BG = '#F4EFFA', PURPLE = '#3E1485', INK = '#29252F';
const GRAY = '#696270', BORDER = '#918999', PALE = '#E4DEE9';

function text(slide, name, label, x, y, w, size = 24, color = INK, bold = false, align = 'left', mono = false) {
  const s = slide.shapes.add({geometry:'textbox', name,
    position:{left:x,top:y,width:w,height:size*1.5}, fill:'none',line:{fill:'none',width:0}});
  s.text = label;
  s.text.style = {typeface:mono?'Courier New':'Arial',fontSize:size,color,bold,
    alignment:align,insets:0,wrap:'none',autoFit:'none',verticalAlignment:'middle'};
  return s;
}

function rect(slide, name, x, y, w, h, fill='none', edge=BORDER, width=1) {
  return slide.shapes.add({geometry:'rect',name,position:{left:x,top:y,width:w,height:h},
    fill,line:{fill:edge,width}});
}

function segment(slide, name, a, b, arrow=false, color=GRAY) {
  return slide.shapes.add({geometry:arrow?'connector':'line',name,
    ...(arrow?{kind:'straight',tail:{type:'arrow',width:'sm',length:'sm'}}:{}),
    position:{left:Math.min(a[0],b[0]),top:Math.min(a[1],b[1]),
      width:Math.abs(a[0]-b[0])||.1,height:Math.abs(a[1]-b[1])||.1,
      horizontalFlip:b[0]<a[0],verticalFlip:b[1]<a[1]},
    line:{fill:color,width:1.5},fill:'none'});
}

function arrow(slide, name, points, color=GRAY) {
  points.slice(1).forEach((point,i)=>segment(slide,`${name}-${i}`,points[i],point,i===points.length-2,color));
}

function chrome(slide,title,page) {
  slide.size={width:1280,height:720};
  slide.background.fill=BG;
  rect(slide,'header',0,0,1280,118,PURPLE,'none',0);
  rect(slide,'page-tab',1178,0,102,118,'#9E89C2','none',0);
  text(slide,'title',title,52,33,1090,34,'#FFFFFF',true);
  text(slide,'page',String(page),1200,32,62,38,'#FFFFFF',true,'center');
  segment(slide,'footer-rule',[54,658],[1226,658],false,'#D6CBE8');
  text(slide,'source','Source: mlx-lm generate.py and models/cache.py',56,674,1030,15,GRAY);
}

export function drawGeneration(slide) {
  chrome(slide,'mlx-lm Generation Procedure',1);
  const steps=[['Prompt',60,160,'Task and code'],['Tokenize',270,170,'Token IDs'],
    ['Prefill',505,160,'Prompt prefix'],['Decode',770,180,'Forward + argmax'],
    ['Output',1040,170,'Generated code']];
  for(const [label,x,w,detail] of steps) {
    text(slide,`step-${label}`,label,x,294,w,30,INK,false,'center');
    text(slide,`detail-${label}`,detail,x-12,344,w+24,20,GRAY,false,'center');
  }
  for(const [a,b] of [[220,265],[440,500],[665,765],[950,1035]])
    arrow(slide,`flow-${a}`,[[a,317],[b,317]],INK);
  arrow(slide,'token-loop',[[904,285],[904,237],[813,237],[813,285]],INK);
  text(slide,'next-token','next token',782,195,155,19,GRAY,false,'center');
  text(slide,'stop','EOS / limit',959,274,96,17,GRAY,false,'center');
  for(const [key,y] of [['K',477],['V',521]]) {
    text(slide,`cache-${key}`,key,465,y-2,28,23,INK,false,'center',true);
    rect(slide,`${key}-prefix`,505,y,290,35,PALE);
    rect(slide,`${key}-generated`,795,y,205,35,'none');
    for(let i=1;i<5;i++) segment(slide,`${key}-token-${i}`,[795+i*41,y],[795+i*41,y+35]);
  }
  arrow(slide,'prefill-cache',[[584,387],[584,465]]);
  text(slide,'initialize','initialize',602,413,125,19,GRAY);
  arrow(slide,'cache-read',[[853,465],[853,387]]);
  text(slide,'read','read',787,413,58,19,GRAY);
  arrow(slide,'cache-append',[[923,387],[923,465]]);
  text(slide,'append','append',937,413,110,19,GRAY);
  text(slide,'prefix-cache-label','prompt cache',531,572,240,20,GRAY,false,'center');
  text(slide,'generated-cache-label','generated tokens',800,572,220,20,GRAY,false,'center');
  slide.speakerNotes.text=`English\nmlx-lm first turns the prompt into tokens. Prefill builds the cache for the prompt prefix. The model then generates tokens one at a time. Each step reads and updates the cache. Generation stops at an end token or the output limit.\n\n中文\nmlx-lm 先把输入转成 token。Prefill 为输入前缀建立缓存。随后模型逐个生成 token。每一步都会读取并更新缓存。遇到结束 token 或达到长度上限时停止。\n\nScope: ordinary greedy decoding, simplified. No timing experiment.\nSources inspected November 16, 2026:\nhttps://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py\nhttps://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py`;
}

export function drawRecovery(slide) {
  chrome(slide,'mlx-lm Cache Recovery',2);
  text(slide,'example','Draft: A B C D     Accepted: A B     Correction: Z',56,160,1165,25,INK,false,'left',true);
  const base=500,cell=73,ys=[256,338,420,502];
  const labels=['Before verification','After verification','After trim','Next forward'];
  for(let row=0;row<4;row++) {
    const y=ys[row];
    text(slide,`state-${row}`,labels[row],56,y+5,295,24);
    rect(slide,`prefix-${row}`,365,y,135,44,PALE);
    text(slide,`prefix-label-${row}`,'prefix',369,y+3,127,22,INK,false,'center');
  }
  text(slide,'pending-x','x',base,ys[0]+3,cell,25,INK,false,'center',true);
  text(slide,'pending-label','pending input',600,ys[0]+8,230,21,GRAY);
  const rows=[['x','A','B','C','D'],['x','A','B'],['x','A','B','Z']];
  rows.forEach((tokens,index)=>tokens.forEach((token,j)=>{
    const y=ys[index+1],x=base+j*cell,rejected=index===0&&j>=3;
    rect(slide,`cell-${index}-${j}`,x,y,cell,44,rejected?'#D5CDDD':'none');
    text(slide,`token-${index}-${j}`,token,x+2,y+3,cell-4,25,INK,false,'center',true);
    if(rejected) segment(slide,`reject-mark-${j}`,[x+10,y+36],[x+63,y+8],false,GRAY);
  }));
  text(slide,'trim-note','Trim C and D',922,ys[1]+8,280,22,GRAY);
  text(slide,'pending-z','Z is the next input',922,ys[2]+8,300,22,GRAY);
  text(slide,'process-z','Process Z',922,ys[3]+8,280,22,GRAY);
  text(slide,'cache-range','The target cache keeps the accepted prefix.',56,593,1165,23,INK);
  text(slide,'cache-scope','Target cache only. Standard trimmable KV cache.',56,625,1165,17,GRAY);
  slide.speakerNotes.text=`English\nThe draft contains four tokens. The target accepts A and B, but predicts Z instead of C. The cache keeps the prefix, x, A and B. It trims the entries for C and D. Z becomes the next input. This mlx-lm path does not replay the accepted prefix.\n\n中文\n草稿有四个 token。目标模型接受 A 和 B，但在 C 的位置预测了 Z。缓存保留原前缀以及 x、A、B，并裁掉 C 和 D 对应的条目。Z 成为下一轮输入。这个 mlx-lm 流程不需要重放已经接受的前缀。\n\nScope: schematic example of the target cache in the official speculative path. A separate draft-model cache is omitted. Applies to a standard trimmable cache, not all cache types. Trimming does not establish unconditional numerical or output equivalence. This is not a newly completed full stock-API experiment. Local restore/replay is a different implementation path.\nSources inspected November 16, 2026, not the unavailable external run's exact version:\nhttps://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py\nhttps://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py`;
}

export async function firstSlide({presentation}) {
  drawGeneration(presentation.slides.getItem(0));
  await presentation.render();
  return {slides:presentation.slides.items.length};
}

export async function secondSlide({presentation}) {
  drawRecovery(presentation.slides.add({width:1280,height:720}));
  await presentation.render();
  return {slides:presentation.slides.items.length};
}
