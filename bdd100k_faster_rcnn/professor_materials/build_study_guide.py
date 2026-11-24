"""Build a professor-facing mathematical study guide from verified results."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                               PageBreak, PageTemplate, Paragraph, Spacer, Table,
                               TableStyle)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "professor_materials" / "out" / "BDD100K_mathematical_study_guide.pdf"
FIGURE = ROOT / "figures" / "final_paired_comparison.png"
NAVY = colors.HexColor("#15324B")
TEAL = colors.HexColor("#167D83")
LIGHT = colors.HexColor("#EAF2F4")
GREY = colors.HexColor("#5C6974")


def register_fonts():
    pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Italic", r"C:\Windows\Fonts\ariali.ttf"))
    pdfmetrics.registerFont(TTFont("Consolas", r"C:\Windows\Fonts\consola.ttf"))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleCustom", parent=base["Title"], fontName="Arial-Bold",
                                fontSize=23, leading=28, textColor=NAVY, spaceAfter=16),
        "subtitle": ParagraphStyle("SubtitleCustom", parent=base["Normal"], fontName="Arial",
                                   fontSize=11, leading=16, textColor=GREY, alignment=TA_CENTER),
        "h1": ParagraphStyle("H1Custom", parent=base["Heading1"], fontName="Arial-Bold",
                             fontSize=15, leading=19, textColor=NAVY, spaceBefore=18, spaceAfter=8),
        "h2": ParagraphStyle("H2Custom", parent=base["Heading2"], fontName="Arial-Bold",
                             fontSize=11, leading=15, textColor=TEAL, spaceBefore=12, spaceAfter=5),
        "body": ParagraphStyle("BodyCustom", parent=base["BodyText"], fontName="Arial",
                               fontSize=9.4, leading=14.2, textColor=NAVY, spaceAfter=7),
        "small": ParagraphStyle("SmallCustom", parent=base["BodyText"], fontName="Arial",
                                fontSize=8.1, leading=11.7, textColor=NAVY, spaceAfter=5),
        "eq": ParagraphStyle("EquationCustom", parent=base["BodyText"], fontName="Consolas",
                             fontSize=9.1, leading=13.5, textColor=NAVY, leftIndent=12,
                             spaceBefore=7, spaceAfter=9),
        "caption": ParagraphStyle("CaptionCustom", parent=base["BodyText"], fontName="Arial-Italic",
                                  fontSize=8.1, leading=11, textColor=GREY, spaceAfter=8),
        "table": ParagraphStyle("TableCustom", parent=base["BodyText"], fontName="Arial",
                                fontSize=8.2, leading=10.5, textColor=NAVY),
    }


class Guide(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=A4, leftMargin=48, rightMargin=48,
                         topMargin=54, bottomMargin=48, title="BDD100K target-aware selection: mathematical study guide",
                         author="Yash - research project")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(1)
        canvas.line(48, h - 42, w - 48, h - 42)
        canvas.setFont("Arial", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(48, 31, "Target-aware combinatorial selection | BDD100K")
        canvas.drawRightString(w - 48, 31, str(doc.page))
        canvas.restoreState()


def make_table(rows, widths, style):
    data = [[Paragraph(str(cell), style["table"]) for cell in row] for row in rows]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, NAVY),
    ]))
    # Paragraph text has its own color; manually white header text instead.
    header = [Paragraph(f'<font color="white"><b>{cell}</b></font>', style["table"]) for cell in rows[0]]
    table._cellvalues[0] = header
    return table


def build():
    register_fonts()
    st = styles()
    P = lambda text: Paragraph(text, st["body"])
    S = lambda text: Paragraph(text, st["small"])
    H = lambda text: Paragraph(text, st["h1"])
    H2 = lambda text: Paragraph(text, st["h2"])
    E = lambda text: Paragraph(text, st["eq"])
    story = []

    story += [Spacer(1, 75), Paragraph("Target-aware combinatorial<br/>training-image selection", st["title"]),
              Paragraph("BDD100K object detection under a fixed compute budget", st["subtitle"]),
              Spacer(1, 34),
              P("<b>Purpose.</b> This guide reconstructs the complete research workflow for a mathematics presentation. It separates the proved property of the <i>selection objective</i> from empirical detector performance."),
              P("<b>Primary finding.</b> Across five paired seeds, adaptive selection improved held-out night COCO mAP@0.50:0.95 by 0.0157 over generic coverage. A later night-count control recovered 0.0094 of that gain. The incremental adaptive advantage over the control was 0.0063 and remains uncertain."),
              Spacer(1, 18),
              make_table([
                  ["Experiment", "Status", "What it establishes"],
                  ["Five-seed night comparison", "Complete", "Adaptive subset improved night detection relative to generic coverage under the chosen budget."],
                  ["Night-count ablation", "Complete, post-final", "More night images account for much of the observed gain."],
                  ["Second target domain", "Pending", "General target conditioning is not yet established."],
              ], [140, 105, 250], st),
              Spacer(1, 20),
              P("<b>Presentation claim to use:</b> We implemented a monotone submodular metadata-coverage selector and found a repeatable night-domain gain under matched compute. The current evidence does not isolate a reliable benefit from class-gap weighting beyond selecting more night images."),
              PageBreak()]

    story += [H("1. Research question and complete workflow"),
              P("Let D be the 70,000-image BDD100K training pool, T a requested deployment condition (night in the completed study), k = 7,500 the image budget, and B = 7,500 the optimizer-update budget. We want a subset S of D with |S| = k that yields high target-domain detection accuracy after training for B updates while keeping general accuracy useful."),
              E("S* = arg max { F_T(S) : S subset D, |S| <= k }"),
              P("The true downstream objective would involve retraining a detector for each candidate subset and measuring target AP. That is computationally infeasible. We therefore maximize a tractable combinatorial surrogate F_T and test whether its selected images improve downstream AP."),
              H2("Procedure"),
              P("1. Audit BDD100K images, annotations, classes, boxes, and metadata. 2. Freeze disjoint query, development, and final partitions. 3. Convert each training image into a set of metadata and object-presence tokens. 4. Generate equal-size subsets by several selection methods. 5. Train the same detector for equal optimizer updates and seeds. 6. Choose checkpoints only by development loss. 7. Score the held-out final set. 8. Compute paired differences and an exploratory uncertainty interval. 9. Run a post-final night-count control to probe mechanism."),
              H2("Why this is a combinatorial problem"),
              P("An image may simultaneously contain night, city, car, pedestrian, and small-object attributes. The image is therefore a set of properties rather than a single class label. Selecting k images to cover many target-relevant combinations is a cardinality-constrained set optimization problem. The token-image incidence relation can be viewed as a hypergraph."),
              H("2. Dataset audit and leakage boundaries"),
              make_table([
                  ["Partition", "Images", "Use"],
                  ["Training pool", "70,000", "Candidate images and labels for selection and training"],
                  ["Night query", "400", "Pixels only for exploratory visual variants"],
                  ["Rain query", "100", "Pixels only; no completed rain training study"],
                  ["Development", "1,000", "Class AP gaps, method screening, checkpoint loss"],
                  ["Final", "8,500", "Held-out scoring of frozen primary comparison"],
              ], [140, 70, 285], st),
              S("The training annotations contain 1,288,010 valid boxes in ten detection classes. The final set has 3,126 night, 4,650 daytime, and 532 rainy images; rainy and time-of-day slices overlap. The final adaptive selector did not use query-image pixels or labels. It did use development AP gaps, so development data are not an independent test set."),
              PageBreak()]

    story += [H("3. Generic coverage objective"),
              P("For each training image i, let A_i be a set of tokens. Tokens include individual weather, scene and time values; valid pairwise metadata interactions; presence of each of ten object classes; and presence of small, medium or large boxes. Let U be the token universe and n_t(S) = sum over i in S of 1[t in A_i]."),
              E("F_generic(S) = sum_{t in U} w_t sqrt(n_t(S))"),
              E("w_t = f_t^(-1/4),  f_t = number of pool images carrying token t"),
              P("The frequency weight reduces domination by ubiquitous tokens. Square root gives diminishing returns: the first selected image carrying a token contributes much more than the hundredth. The 7,500-image subset is found with lazy greedy marginal-gain updates."),
              H("4. Adaptive target weighting"),
              P("From three generic development runs (seeds 42-44), calculate for class c the average day-minus-night AP gap and keep only its positive part:"),
              E("g_c = max(0, mean_seed[ AP_day,c - AP_night,c ])"),
              P("For a night image containing class c, add token target_class=c. Its weight is increased to:"),
              E("w_target,c = f_target,c^(-1/4) * (1 + 12 g_c)"),
              P("The multiplier 12 and the token design were development choices. This construction weights classes that the reference detector handled worse at night. It does not directly optimize AP and it does not use final-set labels. Since night-class tokens occur only on night images, this objective also increases the selected night-image count."),
              make_table([
                  ["Seed-42 subset", "Night images", "Day images", "Dawn/dusk"],
                  ["Generic coverage", "1,732", "3,250", "2,452"],
                  ["Adaptive", "3,851", "1,865", "1,749"],
              ], [170, 100, 100, 125], st),
              S("The selected-image counts themselves are important evidence: a target gain could arise simply from using more night images. The later matched-night-count control tests this explanation."),
              PageBreak()]

    story += [H("5. Mathematical properties and greedy algorithm"),
              H2("Proposition: weighted square-root coverage is normalized, monotone, and submodular"),
              P("Assume every token weight w_t is nonnegative. Normalization follows because n_t(empty set) = 0 for every t, so F(empty set) = 0. Monotonicity follows because adding an image only increases token counts and square root is nondecreasing."),
              P("For submodularity, take A subset B and an image i outside B. For token t carried by i, its marginal contribution is"),
              E("Delta_t(i | S) = w_t [ sqrt(n_t(S)+1) - sqrt(n_t(S)) ]."),
              P("Since n_t(A) <= n_t(B) and x -> sqrt(x+1)-sqrt(x) decreases for x >= 0, Delta_t(i|A) >= Delta_t(i|B). Tokens absent from i contribute zero. Summing over nonnegative weights proves diminishing returns for F. The adaptive method merely adds further positive-weight tokens, so the same proof applies."),
              H2("Greedy bound and its precise meaning"),
              P("For the cardinality constraint |S| <= k, exact greedy achieves the classical bound [3]:"),
              E("F(S_greedy) >= (1 - 1/e) F(S_opt) approximately 0.632 F(S_opt)."),
              P("Lazy greedy uses cached upper bounds on marginal gains. Because this objective is submodular, old gains remain upper bounds after the selected set grows. A candidate is accepted once its recomputed gain exceeds every competing cached bound, yielding the same selection as exact greedy up to tie handling."),
              P("<b>Critical distinction:</b> the 0.632 guarantee concerns the engineered coverage objective F. It is not a lower bound on detector AP, night AP, or transfer performance. Those are empirical questions."),
              H2("Night-count control"),
              P("After seeing the primary result, we ran a generic coverage selector with exactly 3,851 night images and 3,649 non-night images per seed. The exact-quota feasible family differs from the simple cardinality constraint; the 1 - 1/e statement above should not be cited as a guarantee for this constrained implementation."),
              PageBreak()]

    story += [H("6. Detector and controlled experiment"),
              P("The detector is COCO-pretrained TorchVision Faster R-CNN with a ResNet-50/FPN backbone and a new predictor for ten BDD100K object classes plus background [4]. The baseline audit retained prior full-dataset results as historical context, but those old runs used a different schedule and validation set."),
              make_table([
                  ["Quantity", "Fixed research value"],
                  ["Subset size", "7,500 images"],
                  ["Optimizer", "SGD, learning rate 0.005, momentum 0.9, weight decay 0.0005"],
                  ["Batch and augmentation", "Batch 2; random horizontal flip; 640/1024 image size bounds"],
                  ["Compute", "7,500 optimizer updates; BF16 autocast on RTX 5050"],
                  ["Learning-rate decay", "At updates 3,000 and 6,000"],
                  ["Checkpoint choice", "Lower development loss at updates 3,750 or 7,500"],
                  ["Paired seeds", "42, 43, 44, 45, 46"],
                  ["Final metric", "COCO mAP averaged over IoU thresholds 0.50:0.95"],
              ], [150, 345], st),
              P("A fixed number of optimizer updates controls training compute better than equal epochs when subset sizes differ. The same model initialization family, training code, batch size, and final images were used for every primary paired run. Manifests, hashes, run configurations, histories, checkpoints, and scoring CSVs were saved."),
              H2("Methods screened during development"),
              P("We examined random selection, night-only sampling, an 80% night manual mix, generic coverage, metadata target weighting, visual target similarity, visual-cluster coverage, and the adaptive class-time objective. A 1,000-update pilot was diagnostic. Full-budget visual variants did not reproduce a gain against generic coverage across three development seeds. The five-seed adaptive-versus-generic comparison was then frozen before final scoring."),
              H2("Detection metric"),
              P("For each class and IoU threshold, average precision summarizes the precision-recall curve. COCO mAP@0.50:0.95 averages AP across ten IoU thresholds from 0.50 to 0.95 and across classes. Higher thresholds demand more accurate box localization, so this is more informative than a single IoU=0.50 F1 score."),
              PageBreak()]

    story += [H("7. Primary held-out result"),
              P("The protocol declared night mAP as the primary endpoint and overall mAP as the retention endpoint before final scoring. The final set contains 8,500 images, including 3,126 night images."),
              make_table([
                  ["Domain", "Generic mean", "Adaptive mean", "Paired difference", "95% paired interval"],
                  ["Night", "0.2001", "0.2158", "+0.0157", "[+0.0075, +0.0239]"],
                  ["Overall", "0.2198", "0.2231", "+0.0032", "[-0.0049, +0.0113]"],
                  ["Day", "0.2294", "0.2283", "-0.0010", "[-0.0090, +0.0070]"],
                  ["Rain", "0.2505", "0.2566", "+0.0061", "[-0.0075, +0.0197]"],
              ], [70, 95, 95, 110, 125], st),
              Spacer(1, 10),
              Image(str(FIGURE), width=495, height=189),
              Paragraph("Figure 1. Paired held-out mAP. Lines connect the same seed; diamonds mark mean differences.", st["caption"]),
              P("The five night differences (adaptive minus generic) were +0.0147, +0.0096, +0.0093, +0.0209, and +0.0240. Overall mAP improved in only two of five pairs, so an overall gain is not established."),
              H2("Uncertainty calculation"),
              E("d_s = AP_adaptive,s - AP_generic,s;  CI = mean(d) +/- t_(0.975,4) sd(d)/sqrt(5)"),
              P("Here t_(0.975,4) is about 2.776. This exploratory interval summarizes variation across five training seeds on a fixed image set. It does not account for sampling a new dataset, and five seeds are too few for strong distributional claims."),
              PageBreak()]

    story += [H("8. Post-final mechanism check"),
              P("The adaptive subset contained 3,851 night images, whereas generic coverage contained 1,732 for seed 42. We therefore built a control with the same night count as adaptive but generic token weights. This ablation was designed after inspecting the primary final result, so it is exploratory and is not a second confirmatory test."),
              make_table([
                  ["Held-out night mAP", "Generic", "Count-matched control", "Adaptive"],
                  ["Five-seed mean", "0.2001", "0.2095", "0.2158"],
                  ["Gain over generic", "0", "+0.0094", "+0.0157"],
              ], [145, 95, 150, 105], st),
              P("The paired adaptive-minus-control night differences were +0.0113, +0.0034, +0.0025, -0.0008, and +0.0151. Their mean was +0.0063, with exploratory 95% paired interval [-0.0019, +0.0145]. Four pairs favored adaptive, but the interval crosses zero."),
              P("<b>Interpretation:</b> increased night representation explains a substantial part of the primary result. The experiment does not clearly establish that development-gap weighting adds a reliable improvement beyond that count. The control also changed which individual night and non-night images were selected, so it is a useful empirical comparison rather than a perfect causal isolation of every selection component."),
              H("9. Limitations and honest conclusion"),
              P("The study uses one architecture, one subset size, one training budget, one target domain, and five seeds. Rain appears only as an evaluation slice, not as a completed rain-target training experiment. The final adaptive method is metadata-driven; the visual query variants were screened out. The class-gap weights came from the 1,000-image development set. The post-final control was designed after the final result was known."),
              P("<b>Conclusion for the professor:</b> The mathematical objective is a valid monotone submodular selection surrogate with a greedy approximation guarantee. Under matched detector compute, its adaptive version gave a repeatable night gain over generic coverage. The later control shows that much of the gain is attributable to selecting more night images, leaving the extra value of class-gap weighting unresolved."),
              PageBreak()]

    story += [H("10. Presentation guide and likely questions"),
              H2("A 60-second opening"),
              P("'My project asks which 7,500 BDD100K images a detector should train on when nighttime performance matters. I encode each image as a set of metadata and object-coverage tokens. A weighted square-root objective gives diminishing returns, so I can select a subset efficiently with lazy greedy. I then test whether this mathematical surrogate actually improves Faster R-CNN detection under the same 7,500-update budget. Across five seeds, night mAP rose by 0.0157 over generic coverage. A later count-matched control recovered much of that gain, so I separate the optimization guarantee from the empirical mechanism claim.'"),
              H2("If asked: Why is the objective submodular?"),
              P("For a token already covered n times, the next image contributes w[sqrt(n+1)-sqrt(n)]. This difference decreases as n increases. Larger selected sets have at least as many copies of every token, so every candidate's marginal gain can only decrease. Sum over positive weights."),
              H2("If asked: Does the 1 - 1/e theorem guarantee better AP?"),
              P("No. It guarantees a fraction of the optimum of the hand-designed selection objective. The link from objective to AP was tested empirically, and the ablation suggests that night count explains much of the gain."),
              H2("If asked: Is this a general night/rain selection algorithm yet?"),
              P("The implementation accepts target-related metadata, but only night has a completed controlled training study. A rain-target study is still needed before claiming generalization to another target."),
              H2("If asked: Was the evaluation set used to tune the method?"),
              P("The primary comparison and endpoint were frozen before the 8,500-image final set was scored. The later night-count control was explicitly designed after viewing primary final results and is labeled exploratory."),
              H2("If asked: What would you do next?"),
              P("Run a second target with the same selector and a fresh frozen analysis plan. Then test a second subset or compute budget and a stronger coreset baseline. Those would support broader research-paper claims."),
              H("References and reproducibility"),
              S("[1] Yu et al. BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning. CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.html"),
              S("[2] Chen et al. Domain Adaptive Faster R-CNN for Object Detection in the Wild. CVPR 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.html"),
              S("[3] Nemhauser, Wolsey, and Fisher. An Analysis of Approximations for Maximizing Submodular Set Functions-I. Mathematical Programming, 1978."),
              S("[4] Ren et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS 2015. https://arxiv.org/abs/1506.01497"),
              S("Local source files: FINAL_EVALUATION_PROTOCOL.md, FINAL_EVALUATION_RESULTS.md, NIGHT_COUNT_CONTROL_PROTOCOL.md, NIGHT_COUNT_CONTROL_RESULTS.md, select_subsets.py, select_night_count_control.py, train.py, evaluate_domains.py.")]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    Guide(OUTPUT).build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
