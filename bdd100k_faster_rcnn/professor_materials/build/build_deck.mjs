import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = path.resolve(import.meta.dirname, "../..");
const SKILL_DIR = "C:/Users/Yash/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations";
const PYTHON = "C:/Users/Yash/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const OUTPUT = path.join(ROOT, "professor_materials/out/BDD100K_professor_presentation_final.pptx");
const STAGING = path.join(ROOT, "professor_materials/.codex-finalizer");
const PREVIEW = path.join(ROOT, "professor_materials/tmp/deck_preview");
const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href
);
await fs.mkdir(STAGING, { recursive: true });
await fs.mkdir(PREVIEW, { recursive: true });
await fs.mkdir(path.dirname(OUTPUT), { recursive: true });

const FONT = resolvePresentationFont({ fontFamily: "Arial" });
const C = { navy: "#15324B", teal: "#167D83", pale: "#F7FAFA", ink: "#233947", muted: "#5C6974", orange: "#B9672B", white: "#FFFFFF", grey: "#DDE8EA" };
const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const noteSources = "Local sources: RESEARCH_REPORT.md; FINAL_EVALUATION_RESULTS.md; NIGHT_COUNT_CONTROL_RESULTS.md. External references: Yu et al., CVPR 2020; Nemhauser et al., Mathematical Programming 1978; Ren et al., NeurIPS 2015.";

function textbox(slide, text, x, y, w, h, fontSize=24, color=C.ink, bold=false, align="left") {
  const s = slide.shapes.add({geometry:"textbox", position:{left:x,top:y,width:w,height:h}, fill:"none", line:{fill:"none",width:0}});
  s.text = text;
  s.text.style = {typeface:FONT,fontSize,bold,color,alignment:align,autoFit:"none"};
  return s;
}
function base(title, notes, dark=false) {
  const s=p.slides.add();
  s.background.fill=dark?C.navy:C.pale;
  if(title) textbox(s,title,64,43,1140,75,42,dark?C.white:C.navy,true);
  textbox(s,String(p.slides.items.length).padStart(2,"0"),1180,669,45,30,15,dark?C.grey:C.muted,false,"right");
  s.speakerNotes.textFrame.setText(`${notes}\n\n${noteSources}`);
  return s;
}
function body(slide, lines, x=70, y=165, w=1120, font=24, gap=66) {
  lines.forEach((line,i)=>textbox(slide,line,x,y+i*gap,w,gap-5,font,C.ink));
}
function equation(slide, value, y=275, size=37) {
  textbox(slide,value,80,y,1120,90,size,C.teal,true,"center");
}
function lineNote(slide, text, y=610) {
  textbox(slide,text,70,y,1130,58,18,C.muted);
}

// 1. Cover
{
 const s=base("", "Open by saying that the project changed from a day-night detector comparison into a fixed-budget training-image selection question. Say you will prove a property of the selector and then separately test the detector.", true);
 textbox(s,"Target-aware combinatorial\ndata selection",70,188,1110,160,55,C.white,true);
 textbox(s,"Faster R-CNN on BDD100K under a fixed compute budget",74,402,1070,56,27,"#C9E5E6");
 textbox(s,"Research project presentation  |  Yash",74,568,1100,36,21,C.white);
}
// 2. Research question
{
 const s=base("Research question", "Our mathematical variable is a subset of training images. We choose exactly 7,500 of 70,000. The target is nighttime object detection. Training is held to 7,500 optimizer updates for every method. We cannot directly optimize AP over all subsets, so F is a tractable surrogate.");
 textbox(s,"Choose k images from a pool D for target T",75,170,1110,68,30,C.ink);
 equation(s,"S* = arg max F_T(S),   S ⊆ D,   |S| ≤ k",280,40);
 body(s,["D: 70,000 BDD100K training images", "T: night; k: 7,500 images; B: 7,500 training updates"],80,422,1100,25,64);
 lineNote(s,"The theorem concerns F_T. Detector AP is measured after training.");
}
// 3. Existing work and audit
{
 const s=base("Data audit and experimental split", "We first preserved the existing COCO-pretrained Faster R-CNN pipeline and audited all labels. There are 1,288,010 valid training boxes. The original validation set has 10,000 images. We partitioned it before the primary study. The query images were used only as pixels for visual variants; the final adaptive method uses metadata and development AP gaps.");
 body(s,["70,000 train images, 1,288,010 valid boxes, 10 object classes", "400 night query and 100 rain query images", "1,000 development images for selection feedback and checkpoint loss", "8,500 held-out final images, including 3,126 night images"],70,168,1135,24,88);
 lineNote(s,"The four validation partitions are disjoint; final labels did not guide primary selection.");
}
// 4. Set system
{
 const s=base("Images as a set system", "An image carries several attributes at once. That motivates an image-token incidence system. The tokens include time, weather, scene, pairwise interactions, class presence, and box size. Selecting images covers multiple conditions simultaneously. This is the applied combinatorics part.");
 textbox(s,"Image i",72,172,220,50,30,C.teal,true);
 textbox(s,"{night, city, car, person, small box}",320,172,860,50,28,C.ink);
 textbox(s,"Token universe U",72,297,340,50,30,C.teal,true);
 textbox(s,"time, weather, scene, pairwise combinations,\nclasses, box sizes",430,297,760,92,27,C.ink);
 equation(s,"n_t(S) = Σ_{i∈S} 1[t∈A_i]",457,36);
 lineNote(s,"A_i is the token set carried by image i; n_t counts selected images covering token t.");
}
// 5. Generic objective
{
 const s=base("Generic metadata coverage", "The baseline rewards coverage of every token. Token frequency f_t is measured in the 70,000-image pool. Weighting by f to the minus one-quarter power limits domination by ubiquitous attributes. The square root term gives diminishing returns.");
 equation(s,"F(S) = Σ_{t∈U} w_t √n_t(S)",245,43);
 textbox(s,"w_t = f_t^(-1/4)",93,380,520,65,31,C.teal,true);
 textbox(s,"First example of a token matters more\nthan the hundredth",657,374,532,100,27,C.ink);
 lineNote(s,"7,500 images are selected by lazy greedy marginal gains.");
}
// 6. Adaptive objective
{
 const s=base("Adaptive night-class weighting", "Use the generic models' development AP gaps for seeds 42 to 44. For each class, calculate daytime AP minus night AP, average it, and keep the positive part. Add a target_class token to night images carrying that class. The factor 12 was chosen during development. Weights remain nonnegative, so the proof still holds. This shifts the night-image count from 1,732 to 3,851 in seed 42.");
 equation(s,"g_c = max(0, mean(AP_day,c − AP_night,c))",226,35);
 equation(s,"w_target,c = f_target,c^(-1/4)(1 + 12g_c)",353,34);
 textbox(s,"Generic: 1,732 night images",88,492,520,62,27,C.ink);
 textbox(s,"Adaptive: 3,851 night images",652,492,525,62,27,C.ink);
 lineNote(s,"Gaps came from the development set; the held-out final set was not used for these weights.");
}
// 7. Proof
{
 const s=base("Why the objective is submodular", "Give the actual argument. If A is a subset of B, every token count under A is at most its count under B. The marginal square-root increment decreases with count. This holds token by token, and all weights are nonnegative. Therefore the whole sum has diminishing returns. It is also normalized and monotone.");
 equation(s,"Δ_t(i | S) = w_t[√(n_t(S)+1) − √n_t(S)]",208,35);
 body(s,["For A ⊆ B: n_t(A) ≤ n_t(B)", "The increment √(n+1) − √n decreases as n grows", "So Δ(i | A) ≥ Δ(i | B) for every candidate image i"],86,355,1080,25,72);
 lineNote(s,"Nonnegative sums preserve submodularity; adding night-class tokens preserves the proof.");
}
// 8. Greedy
{
 const s=base("Greedy selection and its guarantee", "At each iteration, choose the image with largest marginal increase in F. Lazy greedy stores previous marginal gains as upper bounds and recomputes only when needed. The classical Nemhauser bound applies under a simple cardinality budget. Be precise: the bound concerns F, not detection AP. The later quota control has a different feasible set, so do not apply this bound to it.");
 equation(s,"i* = arg max_{i∉S} [F(S∪{i}) − F(S)]",205,36);
 equation(s,"F(S_greedy) ≥ (1 − 1/e)F(S_opt)",339,39);
 textbox(s,"1 − 1/e ≈ 0.632",80,481,500,62,30,C.teal,true);
 textbox(s,"Guarantee for the surrogate objective, not mAP",580,480,610,76,25,C.ink);
 lineNote(s,"Lazy greedy returns exact greedy choices up to ties for this objective.");
}
// 9. Experimental protocol
{
 const s=base("Controlled detector comparison", "Every method uses COCO-pretrained Faster R-CNN with a ResNet-50/FPN backbone and the same BDD100K head. All runs use the same batch, update count, learning-rate schedule, and checkpoint-selection rule. Seeds 42 to 46 give paired comparisons. COCO mAP is averaged across IoU thresholds 0.50 through 0.95.");
 body(s,["Faster R-CNN ResNet-50/FPN, COCO initialization", "7,500 selected images and 7,500 optimizer updates", "Batch 2; SGD; BF16; LR decay at 3,000 and 6,000", "Five paired seeds: 42, 43, 44, 45, 46", "Checkpoint chosen by development loss only", "Screened random, target-only, manual, and visual variants"],73,154,1110,24,77);
 lineNote(s,"Primary endpoint was night COCO mAP@0.50:0.95; overall mAP was retention.");
}
// 10. Primary result chart
{
 const s=base("Night detection improved in all five pairs", "The adaptive-minus-generic held-out night mAP differences are 0.0147, 0.0096, 0.0093, 0.0209, and 0.0240. Their mean is 0.0157. The exploratory paired t interval is 0.0075 to 0.0239. This compares seed-matched training runs on the same 3,126 night final images. Overall mean difference is 0.0032, and its interval spans zero.");
 const chart=s.charts.add("bar",{position:{left:85,top:165,width:1060,height:355},categories:["42","43","44","45","46"],series:[{name:"Night mAP gain",values:[0.0147,0.0096,0.0093,0.0209,0.0240],fill:C.teal,valuesFormatCode:"0.0000"}],barOptions:{direction:"column",grouping:"clustered"},hasLegend:false,yAxis:{min:0,max:0.03,majorUnit:0.01,numberFormatCode:"0.000",title:"Adaptive minus generic mAP"},xAxis:{title:"Training seed"},dataLabels:{showValue:true,position:"outEnd",textStyle:{typeface:FONT,fontSize:19,fill:C.navy}}});
 applyPresentationChartFont(chart,{fontFamily:FONT});
 textbox(s,"Mean: +0.0157 mAP",86,548,508,58,29,C.teal,true);
 textbox(s,"95% paired interval: [+0.0075, +0.0239]",598,549,600,52,24,C.ink);
 lineNote(s,"Overall mAP: +0.0032 mean; uncertainty interval crosses zero.",636);
}
// 11. Control
{
 const s=base("Night count explains much of the gain", "The primary gain might result from simply selecting more night images. After seeing primary final results, we made a control with exactly 3,851 night images but generic coverage weights. It gained 0.0094 over generic. Adaptive added another 0.0063 over the control, with paired interval minus 0.0019 to plus 0.0145. Since this is post-final, it is exploratory. This weakens a claim that class-gap weighting itself is proved beneficial.");
 const chart=s.charts.add("bar",{position:{left:90,top:165,width:1070,height:345},categories:["Generic","Same night count","Adaptive"],series:[{name:"Held-out night mAP",values:[0.2001,0.2095,0.2158],fill:C.teal,valuesFormatCode:"0.0000"}],barOptions:{direction:"column",grouping:"clustered"},hasLegend:false,yAxis:{min:0,max:0.24,majorUnit:0.05,numberFormatCode:"0.00",title:"Night mAP"},dataLabels:{showValue:true,position:"outEnd",textStyle:{typeface:FONT,fontSize:19,fill:C.navy}}});
 applyPresentationChartFont(chart,{fontFamily:FONT});
 textbox(s,"Control − generic: +0.0094",90,531,530,56,25,C.ink);
 textbox(s,"Adaptive − control: +0.0063",670,531,520,56,25,C.ink);
 lineNote(s,"Adaptive minus control interval: [−0.0019, +0.0145]; post-final exploratory ablation.",626);
}
// 12. Conclusions
{
 const s=base("What the evidence supports", "Separate the mathematical theorem from the empirical result. We proved the surrogate is monotone submodular under nonnegative weights. The primary frozen comparison showed a repeatable night gain. The count-matched ablation shows much of it is explained by more night images, and the additional weighting benefit remains uncertain. Do not claim a general target-conditioned method yet, because rain has not been trained as a target.");
 body(s,["Mathematics: valid monotone submodular coverage objective", "Experiment: +0.0157 night mAP over generic coverage", "Mechanism: +0.0094 recovered by matching the night count", "Class-gap weighting beyond count: +0.0063, uncertain"],76,174,1110,27,88);
 lineNote(s,"The optimization bound is about F(S); the detector result is empirical.");
}
// 13. Next steps
{
 const s=base("Research work still open", "The original project vision included a second target, more data budgets, and stronger baselines. We have completed a careful night study and one post-final mechanism check. The next decisive step is to test the same target-conditioned algorithm on rain or another target with a fresh frozen protocol. A second subset budget and stronger k-center or detection coreset comparison would test robustness. A full paper requires these broader results if it claims general target-aware selection.");
 body(s,["Test rain or another target with a new frozen protocol", "Repeat at a second subset or update budget", "Compare against a stronger feature-space coreset baseline", "Write the final paper around the evidence actually obtained"],74,185,1110,27,95);
 lineNote(s,"Current status: the nighttime case study is complete; general target transfer is untested.");
}
// 14. References
{
 const s=base("References and questions", "Close by offering to derive the submodularity result or explain the ablation. Sources: BDD100K by Yu et al. CVPR 2020; Nemhauser, Wolsey, and Fisher 1978 for the greedy theorem; Faster R-CNN by Ren et al. 2015; Chen et al. CVPR 2018 for domain adaptive detection context. Exact local runs are documented in the protocol and results files.");
 body(s,["Yu et al. BDD100K, CVPR 2020", "Nemhauser, Wolsey, Fisher. Submodular maximization, 1978", "Ren et al. Faster R-CNN, NeurIPS 2015", "Chen et al. Domain Adaptive Faster R-CNN, CVPR 2018"],76,180,1110,23,77);
 textbox(s,"Questions?",76,544,1050,70,37,C.teal,true);
}

const candidate=path.join(STAGING,"candidate.pptx");
await (await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<p.slides.items.length;i++){
  const blob=await p.export({slide:p.slides.items[i],format:"png",scale:1});
  await fs.writeFile(path.join(PREVIEW,`slide-${String(i+1).padStart(2,"0")}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const result=await finalizePresentation({
  workspaceDir:ROOT,candidatePath:candidate,finalPath:OUTPUT,pythonExecutable:PYTHON,
  integrityValidatorPath:path.join(SKILL_DIR,"container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath:path.join(SKILL_DIR,"container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu","12192000,6858000","--validate-heading-fit"],
  requiredNativeChartOwnerSlides:[10,11],materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:"design",families:[FONT]},verifyArtifactToolImport:true,
  receiptPath:path.join(STAGING,"deck_final.validation.json")
});
console.log(JSON.stringify({output:OUTPUT,slides:p.slides.items.length,result}));
