---
marp: true
theme: default
paginate: true
size: 16:9
math: katex
---

<style>
@font-face {
  font-family: 'Noto Sans SC Local';
  src: url('./assets/group_meeting_2026_06_08/NotoSansCJKsc-Regular.otf') format('opentype');
  font-weight: 400;
}
@font-face {
  font-family: 'Noto Sans SC Local';
  src: url('./assets/group_meeting_2026_06_08/NotoSansCJKsc-Bold.otf') format('opentype');
  font-weight: 700;
}
:root {
  --bg: #f7f5ef;
  --ink: #172026;
  --muted: #5f6872;
  --line: #d7d0c3;
  --blue: #255f85;
  --cyan: #44a7b6;
  --green: #557a46;
  --orange: #d97625;
  --red: #b24a3a;
  --yellow: #e8bc4b;
  --panel: #fffdf7;
}
section {
  font-family: 'Noto Sans SC Local', 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif;
  background: var(--bg);
  color: var(--ink);
  padding: 48px 62px 42px;
}
h1 {
  font-size: 42px;
  line-height: 1.12;
  letter-spacing: 0;
  margin: 0 0 16px;
  color: var(--ink);
}
h2 {
  font-size: 28px;
  line-height: 1.2;
  margin: 0 0 20px;
  color: var(--blue);
}
p, li {
  font-size: 23px;
  line-height: 1.35;
}
strong {
  color: var(--orange);
}
.kicker {
  font-size: 18px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 14px;
}
.subtitle {
  font-size: 26px;
  color: var(--muted);
  max-width: 900px;
}
.source {
  position: absolute;
  left: 62px;
  bottom: 18px;
  font-size: 13px;
  color: #6d746f;
}
.tag {
  display: inline-block;
  border-radius: 999px;
  padding: 6px 12px;
  background: #ece6d8;
  color: #333b42;
  font-size: 17px;
  margin: 4px 6px 4px 0;
}
.tag.good { background: #dfead7; color: #2f5b2e; }
.tag.warn { background: #f6e1bf; color: #8b4c16; }
.tag.bad { background: #ead8d3; color: #8b3028; }
.grid2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 28px;
  align-items: stretch;
}
.grid3 {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 22px;
}
.big-number {
  font-size: 58px;
  font-weight: 700;
  color: var(--blue);
  line-height: 1;
}
.label {
  color: var(--muted);
  font-size: 18px;
}
.caption {
  font-size: 17px;
  color: var(--muted);
}
.flow {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 22px;
}
.node {
  min-height: 76px;
  flex: 1;
  background: var(--panel);
  border: 2px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  font-size: 20px;
  font-weight: 700;
}
.node.blue { border-color: var(--blue); color: var(--blue); }
.node.cyan { border-color: var(--cyan); color: #166a73; }
.node.green { border-color: var(--green); color: var(--green); }
.node.orange { border-color: var(--orange); color: var(--orange); }
.arrow {
  color: var(--muted);
  font-size: 28px;
  font-weight: 700;
}
.image-fit {
  width: 100%;
  max-height: 470px;
  object-fit: contain;
  background: white;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px;
}
.image-wide {
  width: 100%;
  max-height: 520px;
  object-fit: contain;
  background: white;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px;
}
.bar {
  --bar-color: var(--blue);
  width: 100%;
  height: 28px;
  margin: 10px 0 20px;
  justify-self: stretch;
}
.bar::before {
  content: "";
  display: block;
  width: 100%;
  height: 28px;
  border-radius: 5px;
  background: var(--bar-color);
}
.w95::before { width: 95%; }
.w94::before { width: 94%; }
.w90::before { width: 90%; }
.w28::before { width: 28%; }
.w27::before { width: 27%; }
.w23::before { width: 23%; }
.w20::before { width: 20%; }
.w0::before { width: 0; }
.bar.orange { --bar-color: var(--orange); }
.bar.red { --bar-color: var(--red); }
.bar.gray { --bar-color: #9aa0a6; }
.metric-row {
  display: grid;
  grid-template-columns: 210px 1fr 90px;
  align-items: center;
  gap: 14px;
  margin: 12px 0;
}
.metric-row span {
  font-size: 20px;
}
.status-grid {
  display: grid;
  grid-template-columns: 1.2fr 1fr 1.2fr;
  gap: 12px;
  margin-top: 20px;
}
.status {
  min-height: 104px;
  border-radius: 8px;
  padding: 16px;
  border: 1px solid var(--line);
  background: white;
}
.status .top {
  font-size: 22px;
  font-weight: 700;
}
.status .bottom {
  font-size: 16px;
  color: var(--muted);
  margin-top: 8px;
}
.status.pass { border-top: 7px solid var(--green); }
.status.obs { border-top: 7px solid var(--yellow); }
.status.fail { border-top: 7px solid var(--red); }
.status.todo { border-top: 7px solid #9097a0; }
.swimlane {
  display: grid;
  grid-template-columns: 170px 1fr 1fr;
  gap: 12px;
  align-items: stretch;
}
.lane-title {
  font-size: 21px;
  font-weight: 700;
  color: var(--blue);
  display: flex;
  align-items: center;
}
.lane-box {
  background: white;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
  font-size: 20px;
  min-height: 84px;
}
.danger {
  border-left: 6px solid var(--red);
}
.ok {
  border-left: 6px solid var(--green);
}
.warnline {
  border-left: 6px solid var(--orange);
}
.mini-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 17px;
}
.mini-table th,
.mini-table td {
  border-bottom: 1px solid var(--line);
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
}
.mini-table th {
  color: var(--blue);
  font-weight: 700;
}
.timeline {
  display: grid;
  grid-template-columns: 90px 1fr;
  gap: 10px 14px;
  align-items: start;
}
.time {
  color: var(--orange);
  font-size: 18px;
  font-weight: 700;
}
.event {
  font-size: 18px;
  line-height: 1.28;
}
.snippet {
  display: inline-block;
  margin-top: 4px;
  color: var(--muted);
  font-family: 'Noto Sans SC Local', sans-serif;
  font-size: 15px;
}
</style>

<!-- _paginate: false -->

<div class="kicker">Group Meeting · 2026-06-08</div>

# 从无梯度 SNN 到 DINO-WM 世界模型路线

<div class="subtitle">三周进展：ES / nano-egg 文献与工程适配，SNN-WAM 路线从 action-BC 收缩到 DINOv2 patch-latent dynamics。</div>

<div class="flow" style="margin-top:56px">
  <div class="node blue">DINO / DINO-WM</div>
  <div class="arrow">→</div>
  <div class="node orange">HyperscaleES / EGGROLL</div>
  <div class="arrow">→</div>
  <div class="node green">SNN latent world model</div>
</div>

<div class="source">本 deck 是组会展示材料；实验页仅引用当前登记证据或明确标注 diagnostic / hypothesis。</div>

---

<div class="kicker">Research Question</div>

# 核心问题：为什么不是直接 BC？

<div class="grid2">
  <div class="panel danger">
    <h2>旧目标</h2>
    <div class="flow" style="display:block">
      <div class="node">视觉 / 状态 / action history</div>
      <div class="arrow" style="text-align:center">↓</div>
      <div class="node orange">低层 action regression</div>
    </div>
    <p class="caption">容易混入 gripper metric、自回归误差、action representation 等问题。</p>
  </div>
  <div class="panel ok">
    <h2>当前目标</h2>
    <div class="flow" style="display:block">
      <div class="node blue">DINOv2 patch latent</div>
      <div class="arrow" style="text-align:center">↓</div>
      <div class="node green">action-conditioned future latent</div>
    </div>
    <p class="caption">把 SNN 放在 world model 的 latent dynamics 位置。</p>
  </div>
</div>

<div style="margin-top:22px">
  <span class="tag warn">hypothesis</span>
  <span class="tag">SNN 不是 policy head</span>
  <span class="tag">ES 是后续候选优化器</span>
</div>

---

<div class="kicker">Literature: DINOv2</div>

# DINOv2：把图像变成稳定的 spatial patch latent

<div class="grid2">
  <div>
    <div class="flow" style="display:block">
      <div class="node blue">RGB frame</div>
      <div class="arrow" style="text-align:center">↓</div>
      <div class="node cyan">ViT-S/14 frozen encoder</div>
      <div class="arrow" style="text-align:center">↓</div>
      <div class="node green">patch tokens [P, D]</div>
    </div>
    <p class="caption">本项目当前 patch cache 使用 ViT-S/14：P=256，D=384。</p>
  </div>
  <div class="panel">
    <h2>为什么适合 world model？</h2>
    <p>不重建像素，直接预测空间 patch features。</p>
    <p>目标从“生成图像”变成“预测可规划 latent”。</p>
    <p>更贴近 DINO-WM 的建模对象。</p>
  </div>
</div>

<div class="source">Sources: Meta AI DINOv2 blog; facebookresearch/dinov2; project gate evidence R-DWM-G1-001。</div>

---

<div class="kicker">Literature: DINO-WM</div>

# DINO-WM：不重建世界，预测未来 patch features

![w:1120](assets/group_meeting_2026_06_08/dino_wm_intro.png)

<div class="source">Official figure: DINO-WM project page, intro.png. Used for literature explanation only.</div>

---

<div class="kicker">Literature: DINO-WM</div>

# DINO-WM 的 planning loop

![w:1120](assets/group_meeting_2026_06_08/dino_wm_model_arch.png)

<div class="source">Official figure: DINO-WM project page, model_arch.png. This deck does not claim local DINO-WM reproduction.</div>

---

<div class="kicker">Literature: HyperscaleES / EGGROLL</div>

# EGGROLL：用低秩扰动做无梯度更新

![w:1050](assets/group_meeting_2026_06_08/eggroll_diagram.png)

<div style="margin-top:16px">
  <span class="tag">rank-one perturbation</span>
  <span class="tag">fitness evaluation</span>
  <span class="tag">weighted average update</span>
  <span class="tag warn">black-box objective</span>
</div>

<div class="source">Official figure: Evolution Strategies at the Hyperscale project page, diagram.png.</div>

---

<div class="kicker">Literature: nano-egg</div>

# nano-egg：小实现帮助理解 EGGROLL-style int8 pretraining

<div class="grid3">
  <div class="panel">
    <div class="big-number">1</div>
    <h2>单文件</h2>
    <p class="caption">更适合读清 noiser、fitness、update。</p>
  </div>
  <div class="panel">
    <div class="big-number">int8</div>
    <h2>纯整数路线</h2>
    <p class="caption">把“不能反传”的架构当成目标，而不是障碍。</p>
  </div>
  <div class="panel">
    <div class="big-number">0</div>
    <h2>本地指标日志</h2>
    <p class="caption">当前只能说代码阅读和本地适配，不能说完成复现实验。</p>
  </div>
</div>

<div class="flow" style="margin-top:38px">
  <div class="node blue">read implementation</div>
  <div class="arrow">→</div>
  <div class="node orange">understand update path</div>
  <div class="arrow">→</div>
  <div class="node green">design SNN toy sanity</div>
</div>

<div class="source">Source: ESHyperscale/nano-egg README and local code inspection; no SNN-WAM result claim.</div>

---

<div class="kicker">Research Framing</div>

# 我的目标：把两条文献线接到 SNN latent dynamics

<div class="flow" style="margin-top:38px">
  <div class="node blue">DINOv2 patch target</div>
  <div class="arrow">→</div>
  <div class="node cyan">DINO-WM dynamics task</div>
  <div class="arrow">→</div>
  <div class="node green">SNN world model</div>
  <div class="arrow">→</div>
  <div class="node orange">ES / EGGROLL candidate</div>
</div>

<div class="grid2" style="margin-top:46px">
  <div class="panel ok">
    <h2>先验证 task</h2>
    <p>DINO-WM-style ANN baseline 必须先过 real-data gate。</p>
  </div>
  <div class="panel warnline">
    <h2>再验证 optimizer</h2>
    <p>ES/EGGROLL 要先在 tiny SNN sanity 上证明非零更新和 fitness 改善。</p>
  </div>
</div>

---

<div class="kicker">Three-Week Progress</div>

# 三条工作线：文献方法、轻量实现、主路线重置

<div class="status-grid">
  <div class="status obs">
    <div class="top">ES / EGGROLL</div>
    <div class="bottom">RWKV-7 1.5B 多组 GSM8K/Countdown runs 已跑；epoch 50 起生成退化，不支持有效优化 claim。</div>
  </div>
  <div class="status obs">
    <div class="top">nano-egg</div>
    <div class="bottom">完成代码阅读与本地执行适配；无可引用指标日志。</div>
  </div>
  <div class="status pass">
    <div class="top">SNN-WAM route</div>
    <div class="bottom">从 action-BC 收缩到 DINO-WM patch-latent world model。</div>
  </div>
</div>

<div class="flow" style="margin-top:46px">
  <div class="node orange">optimizer candidate</div>
  <div class="arrow">+</div>
  <div class="node blue">latent target</div>
  <div class="arrow">+</div>
  <div class="node green">evidence gates</div>
</div>

---

<div class="kicker">Evidence: Legacy BC Route</div>

# 旧 action-BC 路线冻结：评估器有效，但 policy 失败

<div class="metric-row">
  <span>Expert replay</span>
  <div class="bar w90"></div>
  <span>27/30</span>
</div>
<div class="metric-row">
  <span>WAM-GRU future</span>
  <div class="bar red w0"></div>
  <span>0/30</span>
</div>
<div class="metric-row">
  <span>WAM-GRU no future</span>
  <div class="bar red w0"></div>
  <span>0/30</span>
</div>
<div class="metric-row">
  <span>zero / random</span>
  <div class="bar gray w0"></div>
  <span>0/30</span>
</div>

<div style="margin-top:24px">
  <span class="tag good">supported diagnostic</span>
  <span class="tag bad">不作为主路线</span>
  <span class="tag">R-G5-DIAGNOSTIC-EVAL-001</span>
</div>

<div class="source">Evidence: docs/CLAIMS_LEDGER.md C-G5-EVALUATOR-VALIDITY-001 and C-G5-WAM-GRU-FAILURE-001.</div>

---

<div class="kicker">Route Reset</div>

# 新路线：DINO-WM-style patch latent dynamics

<div class="flow" style="margin-top:28px">
  <div class="node blue">LIBERO frame</div>
  <div class="arrow">→</div>
  <div class="node cyan">frozen DINOv2 patch encoder</div>
  <div class="arrow">→</div>
  <div class="node green">z<sub>t</sub> [P,D]</div>
</div>

<div class="flow" style="margin-top:26px">
  <div class="node green">z context + action history</div>
  <div class="arrow">→</div>
  <div class="node orange">future_actions [H,A]</div>
  <div class="arrow">→</div>
  <div class="node blue">ẑ<sub>t+1:t+H</sub> [H,P,D]</div>
</div>

<div class="grid2" style="margin-top:40px">
  <div class="panel">
    <h2>现在问什么？</h2>
    <p>模型是否学会 action-conditioned future latent prediction。</p>
  </div>
  <div class="panel">
    <h2>暂时不问什么？</h2>
    <p>SNN 是否更强、是否低能耗、是否已经支持 closed-loop。</p>
  </div>
</div>

---

<div class="kicker">Gate Status</div>

# DWM-G1/G2/G3 synthetic 已过；real gate 未过

<div class="status-grid">
  <div class="status pass">
    <div class="top">DWM-G1 patch features</div>
    <div class="bottom">18 tests；patch tokens [B,256,384]。</div>
  </div>
  <div class="status pass">
    <div class="top">DWM-G2 transition dataset</div>
    <div class="bottom">12 tests；future_actions / z_target 对齐，无未来泄漏。</div>
  </div>
  <div class="status obs">
    <div class="top">DWM-G3 synthetic ANN</div>
    <div class="bottom">15 tests；shape / gradient / tiny-train，非 real-data evidence。</div>
  </div>
  <div class="status fail">
    <div class="top">DWM-G3 real baseline</div>
    <div class="bottom">未稳定 beat persistence，action-use 证据不足。</div>
  </div>
  <div class="status todo">
    <div class="top">DWM-G4 planning sanity</div>
    <div class="bottom">当前 real planning 证据不能当 gate acceptance。</div>
  </div>
  <div class="status todo">
    <div class="top">DWM-G5 SNN forward</div>
    <div class="bottom">还未实现 world-model SNN claim。</div>
  </div>
</div>

<div class="source">Evidence: R-DWM-G1-001, R-DWM-G2-001, R-DWM-G3-001, R-DWM-G3-DINOWM-BASELINE-REAL-001.</div>

---

<div class="kicker">Evidence: Real DWM-G3</div>

# 当前 real DWM-G3：能跑，但没有过 persistence gate

<div class="metric-row">
  <span>H=1 model</span>
  <div class="bar orange w28"></div>
  <span>0.02808</span>
</div>
<div class="metric-row">
  <span>H=1 persistence</span>
  <div class="bar w27"></div>
  <span>0.02719</span>
</div>
<div class="metric-row">
  <span>H=4 model</span>
  <div class="bar red w95"></div>
  <span>0.19858</span>
</div>
<div class="metric-row">
  <span>H=4 persistence</span>
  <div class="bar w23"></div>
  <span>0.04768</span>
</div>

<div style="margin-top:20px">
  <span class="tag warn">preliminary diagnostic</span>
  <span class="tag bad">not reportable</span>
  <span class="tag">single seed</span>
  <span class="tag">dirty=True</span>
</div>

<div class="source">Metric: patch_cosine_error. Evidence: R-DWM-G3-DINOWM-BASELINE-REAL-001.</div>

---

<div class="kicker">HyperscaleES Reproduction</div>

# RWKV-7 1.5B EGGROLL：配置与已完成 runs

<div class="grid2">
  <div class="panel">
    <h2>背景</h2>
    <p class="caption">复现 <em>Evolution Strategies at the Hyperscale</em>，尝试用 ES 替代梯度下降微调 RWKV-7 LLM 做数学推理。</p>
    <table class="mini-table">
      <tr><th>项</th><th>当前值</th></tr>
      <tr><td>模型</td><td>RWKV-7 1.5B (7gg1.5B)</td></tr>
      <tr><td>lr_scale</td><td>1.0</td></tr>
      <tr><td>sigma</td><td>1e-3</td></tr>
      <tr><td>gen_per_prompt</td><td>8</td></tr>
      <tr><td>generation_length</td><td>100</td></tr>
      <tr><td>parallel_gen/gpu</td><td>512</td></tr>
    </table>
  </div>
  <div class="panel">
    <h2>已完成实验</h2>
    <table class="mini-table">
      <tr><th>实验</th><th>任务</th><th>bs</th><th>诊断</th></tr>
      <tr><td>Jun4 GSM8K</td><td>GSM8K</td><td>4096</td><td>完整退化时间线</td></tr>
      <tr><td>Jun8 GSM8K</td><td>GSM8K</td><td>4096</td><td>复跑</td></tr>
      <tr><td>Jun4 GSM8K</td><td>GSM8K</td><td>128</td><td>小 batch 对照</td></tr>
      <tr><td>Jun4 Countdown</td><td>Countdown</td><td>4096</td><td>同样退化</td></tr>
      <tr><td>其他</td><td>GSM8K</td><td>256/768/800</td><td>batch sweep</td></tr>
    </table>
    <div style="margin-top:16px">
      <span class="tag warn">external engineering diagnostic</span>
      <span class="tag bad">not successful reproduction</span>
    </div>
  </div>
</div>

<div class="source">Source: user-provided HyperscaleES reproduction notes, 2026-06-08. Not a SNN-WAM result artifact.</div>

---

<div class="kicker">HyperscaleES Diagnostic</div>

# 生成质量退化：越训越不像数学推理

<div class="grid2">
  <div class="panel">
    <h2>质量样例</h2>
    <div class="timeline">
      <div class="time">Epoch 0</div>
      <div class="event">正常英文推理 <span class="snippet">Okay, let's see. Natalia sold...</span></div>
      <div class="time">Epoch 50</div>
      <div class="event">重复 token <span class="snippet">Thinking Think&lt; think&gt; ...</span></div>
      <div class="time">Epoch 100</div>
      <div class="event">乱码/重复字符 <span class="snippet">ic'ic'owe'o'io'o...</span></div>
      <div class="time">Epoch 250</div>
      <div class="event">更严重重复 <span class="snippet">d'n'e'n'a'n'a...</span></div>
    </div>
    <p class="caption">Countdown epoch 100 也出现无意义混合输出。</p>
  </div>
  <div class="panel">
    <h2>bs=4096 Jun4 时间线</h2>
    <div class="timeline">
      <div class="time">0</div><div class="event">正常</div>
      <div class="time">100</div><div class="event">重复字符：<span class="snippet">-o-, {, }</span></div>
      <div class="time">200</div><div class="event">重复 token：<span class="snippet">...-s-s-s...</span></div>
      <div class="time">300</div><div class="event">重复片段：<span class="snippet">//s* of his on-the-</span></div>
      <div class="time">400</div><div class="event">完全乱码：<span class="snippet">&lt;1., &lt;th.</span></div>
      <div class="time">450</div><div class="event">重复模式：<span class="snippet">A- ).</span></div>
    </div>
    <div style="margin-top:14px">
      <span class="tag bad">generation collapse</span>
      <span class="tag">不是正向优化证据</span>
    </div>
  </div>
</div>

<div class="source">Source: user-provided HyperscaleES reproduction notes, 2026-06-08. Output snippets shortened for presentation.</div>

---

<div class="kicker">Risks and Stop Rules</div>

# 风险：路线要收敛，claim 要克制

<div class="grid3">
  <div class="panel danger">
    <h2>Real baseline</h2>
    <p>DWM-G3 real baseline 若不 beat persistence，SNN/ES 结果不可解释。</p>
  </div>
  <div class="panel danger">
    <h2>ES variance</h2>
    <p>generation collapse 后，不继续扩大 batch/epoch；先查 update、fitness、KL、sigma/lr。</p>
  </div>
  <div class="panel danger">
    <h2>Claim boundary</h2>
    <p>不讲 SNN 更优、低能耗，或超出小规模 LIBERO 证据的结论。</p>
  </div>
</div>

<div class="flow" style="margin-top:44px">
  <div class="node blue">report evidence</div>
  <div class="arrow">→</div>
  <div class="node orange">mark diagnostic</div>
  <div class="arrow">→</div>
  <div class="node green">choose next gate</div>
</div>

<div class="source">Current bottom line: route is clearer, but DWM real baseline and SNN/ES are not yet accepted by gates.</div>
