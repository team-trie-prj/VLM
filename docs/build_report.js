// VLM 부문 개발 보고서(.docx) 생성 — docx-js
// 실행: NODE_PATH=$(npm root -g) node docs/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  TableOfContents, PageBreak, PageNumber, Header, Footer,
} = require("docx");

const CW = 9360; // content width (US Letter, 1" margins)
const GREY = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const BORDERS = { top: GREY, bottom: GREY, left: GREY, right: GREY };

const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const P = (t, opts = {}) => new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: t, ...opts })] });
const B = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, children: [new TextRun(t)] });
const CODE = (t) => new Paragraph({
  spacing: { after: 40 }, shading: { type: ShadingType.CLEAR, fill: "F0F0F0" },
  children: [new TextRun({ text: t, font: "Consolas", size: 18 })],
});

function table(headers, rows, widths) {
  const cell = (txt, i, head) => new TableCell({
    borders: BORDERS, width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    shading: head ? { type: ShadingType.CLEAR, fill: "D5E8F0" } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text: String(txt), bold: !!head })] })],
  });
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) })),
    ],
  });
}

const children = [];

// ── 표지 ──
children.push(
  new Paragraph({ spacing: { before: 2400, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "도로 손상 자동 탐지·라벨링 파이프라인", bold: true, size: 44 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: "VLM 이미지 이해 부문 개발 보고서", size: 30, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "지엔소프트(주) 프로젝트형 일경험", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "작성일: 2026-06-21", size: 24, color: "555555" })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ── 목차 ──
children.push(H1("목차"),
  new TableOfContents("목차", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }));

// ── 1. 개요 ──
children.push(
  H1("1. 개요"),
  P("본 보고서는 지엔소프트 제안서의 'VLM·SAM·YOLOe를 활용한 이미지 분석·라벨링·검수 자동화' 부문을 구현한 결과를 정리한 것이다. 자연어 한 문장으로 도로 이미지에서 손상(포트홀 등)을 탐지하고, 라벨을 자동 생성하며, 사람이 검수하고, 도메인 모델을 파인튜닝하는 반자동 데이터 구축 파이프라인을 프로토타입으로 완성했다."),
  H2("1.1 한 줄 목표"),
  P("이미지 + \"포트홀 찾아줘\" → 캡션·속성(메타데이터) + 탐지(bbox) → COCO/YOLO 라벨 → 검수 지표 → 파인튜닝 전/후 성능 비교."),
  H2("1.2 구현 범위"),
  B("파이프라인 ①~⑥ 전 단계 + 통합 데모 UI"),
  B("도메인: 도로 손상(포트홀/균열). 데이터: Roboflow Pothole(brad-dwyer/pothole-voxrl v1, 665장, ODbL)"),
  B("교체 가능한 백엔드 추상화로 모델 조합 비교·검증 지원"),
);

// ── 2. 개발 환경 및 제약 ──
children.push(
  H1("2. 개발 환경 및 제약"),
  P("개발 PC에 NVIDIA GPU가 없고(Intel 내장만), 유료 API 비용을 피해야 하는 제약이 있었다. 이에 따라 다음 전략을 택했다."),
  table(
    ["항목", "선택", "이유"],
    [
      ["VLM/탐지 추론", "Google Gemini 2.5 Flash (무료 티어)", "GPU 불필요, 무료. Claude 구독과 API는 별개 과금이라 무료 경로 채택"],
      ["파인튜닝", "무료 Colab GPU + Ultralytics YOLO", "Gemini는 닫힌 API라 학습 불가 → 오픈 모델을 Colab에서 학습"],
      ["로컬 추론", "파인튜닝 YOLO를 CPU로 실행", "학습된 best.pt를 GPU 없이 로컬 추론(ultralytics)"],
      ["키 없이 실행", "mock 백엔드", "키/GPU 없이 파이프라인·테스트 검증"],
    ],
    [1900, 2800, 4660],
  ),
  P("무료 Gemini 한도: 분당 5회(5 RPM) + 일일 20회(20/day). 상시 시연·배치는 로컬 YOLO 권장.", { italics: true, size: 20 }),
);

// ── 3. 아키텍처 ──
children.push(
  H1("3. 시스템 아키텍처"),
  P("핵심 설계는 '교체 가능한 백엔드 추상화'다. 공통 인터페이스 아래 여러 구현을 갈아끼워 같은 파이프라인에서 모델을 비교한다."),
  H2("3.1 백엔드 추상화"),
  B("VLMBackend (이미지 이해): mock / gemini / anthropic / openai / qwen — analyze_image·vqa·parse_query"),
  B("DetectorBackend (탐지·분할): mock / gemini / yolo — detect(이미지, 개념, mask_dir)"),
  H2("3.2 파이프라인 흐름"),
  CODE("[이미지 + 질의] → ① VLM 이해(메타데이터) + ② 질의 파싱(개념)"),
  CODE("            → ③ 탐지·분할(bbox/mask) → ④ COCO/YOLO 라벨"),
  CODE("            → ⑤ 검수(자동확정률·수정비율) → ⑥ 파인튜닝(전/후 mAP)"),
  CODE("            → 데모 UI / 모델 비교·평가"),
);

// ── 4. 기능별 상세 ──
children.push(H1("4. 기능별 상세 (기능 / 구현 방법 / 사용법)"));

const feature = (title, desc, method, usage) => {
  children.push(H2(title), P(desc));
  children.push(H3("구현 방법"));
  method.forEach((m) => children.push(B(m)));
  children.push(H3("사용법"));
  usage.forEach((u) => children.push(CODE(u)));
};

feature("4.1 VLM 이미지 이해 (①)",
  "도로 이미지의 캡션·속성(도로종류·노면·날씨·시간대)과 손상 유형을 추정해 구조화 메타데이터(JSON)를 생성한다.",
  ["google-genai/Anthropic의 구조화 출력(response_schema = Pydantic ImageAnalysis)으로 스키마를 강제",
   "호출별 토큰 사용량·예상 비용 기록(pricing.py), prompt_log로 재현성 확보",
   "배치 처리 시 처리시간 측정(라벨링 시간 단축률 근거)"],
  ["python -m vlm --backend gemini analyze data/images/real/Pothole_000.jpg",
   "python -m vlm --backend gemini batch data/images/real --delay 13"]);

feature("4.2 자연어 질의 파싱 (②)",
  "\"포트홀 찾아줘\" 같은 자연어를 탐지 단계용 개념 프롬프트(예: pothole)로 변환한다.",
  ["LLM 구조화 출력(ConceptPrompt) + 한글/영문 키워드 규칙기반 폴백(keywords.py)",
   "탐지 백엔드(SAM/YOLOE/Gemini)의 입력으로 연결되는 인터페이스"],
  ["python -m vlm query \"도로 균열이랑 패임 표시해줘\"  # -> ['crack','rutting']"]);

feature("4.3 탐지·분할 (③)",
  "개념 프롬프트를 받아 bbox(+분할 mask)를 산출하고 이미지에 오버레이한다.",
  ["Gemini 2.5의 box_2d([ymin,xmin,ymax,xmax] 0~1000 정규화)를 픽셀 좌표로 변환",
   "분할 mask는 base64 PNG 디코드→이진화; 무료 flash는 mask가 불안정해 실패 시 박스로 폴백",
   "무료 한도 대응 429 자동 재시도(backoff), PIL 오버레이 렌더링"],
  ["python -m vlm detect <img> \"포트홀 찾아줘\" --detector gemini",
   "python -m vlm detect <img> --detector yolo --masks"]);

feature("4.4 자동 라벨 생성 (④)",
  "탐지 결과를 COCO/YOLO 포맷으로 변환하고 confidence로 필터링한다(검수 도구 import용).",
  ["to_coco(bbox=[x,y,w,h]+score) / to_yolo(정규화 class cx cy w h) + classes.txt",
   "--min-conf 임계값으로 자동확정 후보/검수 대상 분리 근거 제공"],
  ["python -m vlm label data/images/real \"포트홀 찾아줘\" --detector yolo --min-conf 0.3"]);

feature("4.5 검수 (⑤)",
  "자동 라벨을 사람이 검수하는 루프. confidence 트리아지와 수정 전/후 비교 지표를 산출한다.",
  ["triage: confidence로 자동확정/검수대상 분리 → 자동확정률",
   "compare_coco: auto와 사람-수정 COCO를 IoU 매칭 → 수정 비율·precision·recall",
   "CVAT/Label Studio는 COCO 라운드트립으로 연결(둘 다 COCO import/export)"],
  ["python -m vlm review prep --coco data/outputs/labels_export/annotations_coco.json --confirm-thr 0.5",
   "python -m vlm review report --auto auto.json --reviewed corrected.json"]);

feature("4.6 파인튜닝 (⑥)",
  "도메인(포트홀) 데이터로 YOLO를 무료 Colab GPU에서 파인튜닝하고, 학습된 모델을 로컬 파이프라인에 끼운다.",
  ["notebooks/finetune_yolo_colab.ipynb: 데이터 업로드 → ultralytics YOLO 학습 → 전/후 mAP → best.pt",
   "YoloDetector(ultralytics): best.pt를 GPU 없이 로컬 CPU 추론으로 DetectorBackend에 통합"],
  ["# Colab에서 학습 후 best.pt -> models/pothole_yolo.pt 로 저장",
   "python -m vlm detect <img> --detector yolo"]);

feature("4.7 모델 비교·평가 (⑦)",
  "백엔드/모델 간 성능을 정량 비교한다(제안서 '모델 비교·검증' 산출물).",
  ["eval: 정답(GT) 라벨 대비 IoU 매칭으로 precision/recall 산출(탐지기 비교)",
   "compare: 같은 이미지를 여러 VLM 백엔드로 분석해 일치율·속도·비용 비교"],
  ["python -m vlm eval data/roboflow/Pothole.v1-raw.yolov8/test --detector yolo",
   "python -m vlm eval ... --detector gemini --limit 15 --delay 13"]);

feature("4.8 데모 UI (⑧)",
  "업로드 → 자연어 질의 → 탐지 → 박스 오버레이 + COCO 라벨 다운로드를 한 화면에서 시연한다(앵커 시나리오 완성형).",
  ["Gradio(6.19) Blocks UI, DetectorBackend 추상화 재사용",
   "탐지 모델 선택(yolo/gemini), 신뢰도 임계값 슬라이더로 회수율 조절"],
  ["pip install gradio", "python -m vlm demo   # http://127.0.0.1:7860"]);

// ── 5. 정량 결과 ──
children.push(
  H1("5. 정량 결과"),
  H2("5.1 파인튜닝 전/후 (제안서 핵심 결과물)"),
  table(["모델", "mAP@0.5", "비고"],
    [["COCO 사전학습 YOLO", "≈ 0", "pothole 클래스 없음(전)"],
     ["파인튜닝 YOLO(yolov8n)", "0.79", "test 67장·154 인스턴스(후), P 0.82 / R 0.73"]],
    [4000, 1860, 3500]),
  H2("5.2 탐지 모델 비교 (동일 test 15장, IoU≥0.5)"),
  table(["탐지기", "precision", "recall", "TP/FP/FN"],
    [["파인튜닝 YOLO", "59%", "74%", "23 / 16 / 8"],
     ["Gemini zero-shot", "26%", "23%", "7 / 20 / 24"]],
    [3000, 2120, 2120, 2120]),
  P("→ 도메인 파인튜닝이 zero-shot을 크게 능가(회수율 74% vs 23%). 파인튜닝의 효과가 수치로 증명됨.", { bold: true }),
  H2("5.3 검수 지표 / 비용"),
  B("검수: 트리아지 자동확정률 + 수정 전/후 비교(수정 비율·precision·recall) 산출"),
  B("비용: Gemini 무료 티어로 추론 비용 $0(일일 20회 한도 내). YOLO 추론은 로컬·무료·무제한"),
);

// ── 6. 데이터 구조 ──
children.push(
  H1("6. 데이터 구조 (스키마)"),
  P("제안서 '데이터 구조 설계안'에 대응하는 Pydantic 스키마로 구현했다."),
  table(["스키마", "용도"],
    [["ImageAnalysis / VLMResult", "캡션·속성·메타데이터(①)"],
     ["ConceptPrompt", "자연어 질의 → 탐지 개념(②)"],
     ["Detection / DetectionResult", "탐지 bbox·라벨·신뢰도·mask(③)"],
     ["ReviewItem/Manifest/Report", "검수 상태·검수자·시각, 자동확정률·수정비율(⑤)"],
     ["PromptLog", "질의·모델·응답·시각(재현성)"]],
    [3600, 5760]),
);

// ── 7. 사용 방법 ──
children.push(
  H1("7. 사용 방법"),
  H2("7.1 설치 및 빠른 시작 (키 없이)"),
  CODE("pip install -r requirements.txt"),
  CODE("python scripts/make_sample_images.py"),
  CODE("python -m vlm doctor                 # 환경 진단"),
  CODE("python -m vlm --backend mock batch data/images"),
  H2("7.2 실제 추론 (무료 Gemini)"),
  CODE("pip install google-genai"),
  CODE("# .env 에 GEMINI_API_KEY 입력 (https://aistudio.google.com 무료 발급)"),
  CODE("python -m vlm --backend gemini doctor --ping"),
  H2("7.3 CLI 명령 요약"),
  table(["명령", "기능"],
    [["analyze / batch", "VLM 이미지 이해(①) 단건/배치"],
     ["query", "자연어 질의 → 개념(②)"],
     ["detect", "탐지·분할 + 오버레이(③)"],
     ["label", "COCO/YOLO 자동 라벨(④)"],
     ["review", "검수 트리아지/수정 비교(⑤)"],
     ["eval / compare", "모델 평가·비교(⑦)"],
     ["demo", "Gradio 데모 UI(⑧)"],
     ["doctor / info", "환경 진단 / 설정 확인"]],
    [2600, 6760]),
);

// ── 8. 한계 및 향후 ──
children.push(
  H1("8. 한계 및 향후 과제"),
  B("분할 mask: 무료 gemini-2.5-flash는 유효 PNG mask가 불안정 → 박스가 신뢰 출력. 정밀 mask는 SAM/FastSAM(CPU) 또는 gemini-2.5-pro 권장"),
  B("무료 한도: Gemini 일일 20회 → 상시 시연은 로컬 YOLO 사용"),
  B("정확도: yolov8n·50ep 베이스라인 → yolov8s/m·에폭↑·데이터 보강으로 향상 가능"),
  B("문서 산출물: 기능정의서·UI/UX·아키텍처 설계안·발표자료는 추가 작성 필요"),
);

// ── 9. 부록 ──
children.push(
  H1("9. 부록: 개발 산출물"),
  B("GitHub: team-trie-prj/VLM (feat/* 브랜치 + PR #1~#11 병합)"),
  B("핵심 모듈: vlm/{schemas,config,pipeline,prompts,keywords,labels,review,eval,compare,overlay,demo,cli}.py"),
  B("백엔드: vlm/backends/* (VLM), vlm/detectors/* (탐지)"),
  B("테스트: tests/test_{mock,compare,detect,labels,review,eval,demo,api_backends}.py"),
  B("Colab: notebooks/finetune_yolo_colab.ipynb"),
);

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: "1F4E79" }, paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "2E75B6" }, paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, color: "404040" }, paragraph: { spacing: { before: 140, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 280 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("VLM 부문 개발 보고서  ·  "), new TextRun({ children: [PageNumber.CURRENT] })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "VLM_프로젝트_보고서.docx");
  fs.writeFileSync(out, buf);
  console.log("saved:", out, (buf.length / 1024).toFixed(1) + " KB");
});
