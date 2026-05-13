# Open-Source Course Writing Conventions

This document captures the writing conventions, structural patterns, and formatting standards used in open-source community courses, distilled from the Hello-Agents project. Use this as a reference when writing GPU programming course chapters.

---

## 1. Directory Structure Convention

```
gpu-programming-course/
├── README.md                       # Bilingual project overview (Chinese)
├── README_EN.md                    # Bilingual project overview (English)
├── docs/
│   ├── _sidebar.md                 # Navigation sidebar for online reader
│   ├── 前言.md                     # Preface: motivation, background, reader guidance
│   ├── images/                     # All figures and images
│   │   ├── project-logo.png        # Project-level images
│   │   ├── chapter1-figures/       # Per-chapter figure directories
│   │   │   ├── chapter1-1.png
│   │   │   ├── chapter1-2.png
│   │   │   └── chapter1-table-1.png
│   │   ├── chapter2-figures/
│   │   └── ...
│   ├── chapter1/
│   │   ├── 第1章 中文标题.md        # Chinese chapter file
│   │   └── Chapter1-English-Title.md # English chapter file
│   ├── chapter2/
│   └── ...
├── code/                           # All runnable code organized by chapter
│   ├── chapter1/
│   │   ├── example_name.py
│   │   └── example_name.ipynb
│   ├── chapter2/
│   └── ...
├── Extra-Chapter/                  # Community contributions and supplementary materials
│   ├── Extra01-面试问题总结.md      # Extra content with serial numbers
│   ├── Extra01-参考答案.md          # Answer keys (only for standalone content, NOT for chapter exercises)
│   ├── Extra02-补充知识.md
│   └── readme.md
├── outputs/                        # Generated artifacts (PDFs, reports, syllabi)
├── Co-creation-projects/           # Community co-creation projects
└── LICENSE                         # Creative Commons BY-NC-SA 4.0
```

### Key Rules

- Images go in `docs/images/`, with per-chapter subdirectories named `chapterN-figures/`.
- Code goes in `code/chapterN/`, never inline in chapter markdown files.
- Extra chapters use `Extra-Chapter/` with `ExtraNN-` prefix naming.
- Both Chinese and English versions of each chapter exist (see Section 7).

---

## 2. Chapter Template

Every chapter follows this exact structural template. Use it as a fill-in-the-blanks guide.

```markdown
# 第N章 中文标题

[Opening motivation paragraph: 2-4 sentences welcoming the reader, framing why this chapter matters, connecting to previous chapters, and setting expectations. Use conversational Chinese with 欢迎 / 我们来 / 我们将 / 在本章.]

[Optional overview paragraph listing the specific topics to be covered, and a diagram for chapter structure]

<div align="center">
  <img src="../images/chapterN-figures/chapterN-opening.png" alt="章节概述" width="90%"/>
  <p>图 N.0 本章知识结构</p>
</div>

## N.1 第一节标题

[Section opening: 1-2 paragraphs introducing the core concept and its significance.]

### N.1.1 子节标题

[Subsection content: definitions, explanations, examples. Mix theory with practical examples.]

[For first-use technical terms, provide Chinese name first, then English in parentheses:]
<strong>线程块（Thread Block）</strong>

[For key concepts, reinforce with explanations of why they matter.]

### N.1.2 子节标题 (continued)

[If there are multiple subsections, continue logically.]

## N.2 动手体验：描述性标题

[Section opening paragraph: transition from theory to practice, motivating why hands-on matters.]

在本节中，我们将引导您 [activity description]，让您直观地感受 [the concept in action]。让我们开始吧！

[If the hands-on section has a defined goal/scenario, state it clearly:]
在本案例中，我们的目标是 [goal]。需要解决的 [task] 定义为："[specific task description]"。要完成这个任务，[brief explanation of what the reader will learn].

### N.2.1 准备工作

[Prerequisites: dependencies to install, accounts to create, environment setup.]

```bash
pip install package1 package2
```

[Configuration file setup, environment variables, etc.]

### N.2.2 核心实现

[Step-by-step explanation of the implementation, alternating between explanation and code.]

[Code should be COMPLETE and RUNNABLE -- see Section 5 for code standards.]

### N.2.3 运行案例分析

[Show the actual output of running the code.]

```bash
实际运行输出：
用户输入: ...
========================================
--- 循环 1 ---
...
```

[Analysis of the output: explain what happened, why the agent/GPU behaved this way, what we can learn from it.]

## N.3 下一节标题 (if applicable)

[Continue with more theory sections, following the same subsections pattern.]

## N.X 本章小结

[Summary section that recaps every major section in bullet points. Each bullet should:
- Use <strong>关键词</strong> to anchor each point
- Start with a question or topic framing
- Explain what was learned and why it matters]

在本章中，我们 [overall statement about the learning journey]. 我们的旅程从 [starting point] 开始：

- <strong>Topic 1 heading?</strong> Explanation of what was covered and its significance.
- <strong>Topic 2 heading?</strong> Explanation of what was covered and its significance.

[Closing transition paragraph that bridges to the next chapter:]
通过本章的学习，我们建立了 [knowledge summary]。在下一章中，我们将 [preview of next chapter]！

## 习题

> <strong>提示</strong>：以下的部分习题没有标准答案，重点在于培养学习者对[相关领域]批判性的深入思考和动手实践能力。

[Exercise list: 5-8 multi-part questions with escalating difficulty. See Section 4 for exercise design patterns.]

## 参考文献

[Numbered reference list in GB/T 7714 or similar format:]

[1] Author. Title[C]//Conference. Year.
[2] Author. Title[M]. Edition. Place: Publisher, Year.
[3] Author. Title[J]. Journal, Year, Volume(Issue): Pages.

---

## 💬 讨论与交流

本章学习过程中遇到问题?想与其他学习者交流心得?

**📝 前往 GitHub Discussions 讨论区:**
- [💬 习题讨论与问答](https://github.com/open-rdma/gpu-programming-guide/discussions)
- 在这里你可以:
  - ✅ 提问习题相关问题
  - ✅ 分享你的解题思路
  - ✅ 与其他学习者交流经验
  - ✅ 获得社区的帮助和反馈

**💡 提示:** 每个页面底部也有评论区,可以直接在页面内讨论!

---
```

---

## 3. Writing Style Rules

### 3.1 Language and Tone

- **Conversational Chinese** throughout. Use 我们 (we) and 你 (you) to build rapport.
- The tone is warm, encouraging, and teacherly -- but never patronizing.
- Opening chapters with phrases like:
  - 欢迎来到 [topic] 的世界！
  - 在本章中，让我们 [action]
  - 现在，是时候 [action]
- Chapter transitions at the end of summaries:
  - 在下一章中，我们将 [preview]
  - 敬请期待！
- Use rhetorical questions to engage: 为什么 [thing] 如此重要？

### 3.2 Technical Terms (Bilingual Convention)

**First occurrence in each chapter**: Chinese name first, then English in parentheses.

> <strong>智能体（Agent）</strong>

> <strong>大语言模型（Large Language Model, LLM）</strong>

> <strong>流多处理器（Streaming Multiprocessor, SM）</strong>

**Subsequent occurrences**: Use Chinese or the English abbreviation freely, whichever reads more naturally in context.

### 3.3 Bold Text

**ALWAYS use `<strong>...</strong>` -- NEVER use `**...**` for bold.**

```markdown
<!-- CORRECT -->
<strong>重要概念</strong>

<!-- INCORRECT (do not use) -->
**重要概念**
```

This convention ensures consistent rendering across all platforms (GitHub, Docsify, GitBook, PDF).

### 3.4 Figures

Every figure uses this exact HTML block:

```html
<div align="center">
  <img src="[path relative to the markdown file]" alt="[brief description]" width="90%"/>
  <p>图 X.Y 说明文字</p>
</div>
```

**Rules**:
- The `width` attribute is typically `"90%"` for content figures, `"70%"` for comparison tables/frameworks, and `"95%"` for architecture diagrams.
- Figure caption format: `图 章节号.序号 说明` (e.g., `图 1.3 智能体决策时间与质量关系图`)
- Image paths are relative from the markdown file: `../images/chapterN-figures/name.png`
- Images may be hosted on GitHub raw URLs for absolute references.

### 3.5 Tables

**Simple tables**: Use standard Markdown tables.

```markdown
| 特性     | 类型A    | 类型B    |
| :------- | :------- | :------- |
| 维度1    | 值       | 值       |
```

**Complex comparison tables**: Save as images and reference like figures.

```html
<div align="center">
  <p>表 N.1 表格标题</p>
  <img src="../images/chapterN-figures/chapterN-table-1.png" alt="表格说明" width="90%"/>
</div>
```

The rule of thumb: if a table has more than 3 columns with substantial text content, make it an image.

### 3.6 Code Blocks

- Always use triple backticks with a language tag: ` ```python `, ` ```bash `, ` ```cpp `
- Code must be **complete and runnable** (see Section 5 for full standards).
- Expected output is often shown in ` ```bash ` blocks labeled as "实际运行输出" or similar.

### 3.7 Tips and Warnings

Use blockquotes with bold labels:

```markdown
> <strong>提示</strong>：这是一条提示信息，帮助读者理解关键点。

> <strong>注意</strong>：这是一条警告，提醒读者避免常见错误。
```

Do not invent other labels (no 信息, 警告, 建议). Stick to 提示 and 注意 only.

### 3.8 Mathematics

- **Inline math**: `$...$` for single expressions within text.
- **Display math**: `$$...$$` for centered, standalone equations.

```markdown
时间复杂度为 $O(n \log n)$。

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
```

### 3.9 Lists Within Exercise Prompts

Use indented sub-items with letters/numbers. Wrap case descriptions in `<strong>` for the label.

### 3.10 Dashes and Separators

- Use `---` (three dashes) for major section separators before the 讨论与交流 block.
- Use `---` as horizontal rules sparingly -- usually only at the very end of a chapter.

---

## 4. Exercise Design Pattern

### 4.1 Structure

Every chapter ends with 5-8 exercises, each following a consistent pattern:

1. **Opening disclaimer** (always the same, adapted per chapter):

   ```markdown
   > <strong>提示</strong>：以下的部分习题没有标准答案，重点在于培养学习者对[领域]批判性的深入思考和动手实践能力。
   ```

2. **Multi-part questions** labeled 1, 2, 3, ... with sub-parts (a, b, c, d).

3. **Escalating difficulty** across three tiers:
   - **Tier 1 (Q1-Q2)**: Recall and basic analysis -- questions about concepts introduced in the chapter.
   - **Tier 2 (Q3-Q5)**: Application and design -- scenario-based questions requiring synthesis.
   - **Tier 3 (Q6-Q7)**: Critical thinking and capstone -- open-ended, philosophical, or multi-system design questions.

### 4.2 Question Types

**Analysis questions** (分析题):
```
请分析以下 [scenario/case]：
- [aspect a]
- [aspect b]
```

**Design questions** (设计题):
```
假设你需要设计一个 [system/scenario]。请：
- [requirement a]
- [requirement b]
```

**Hands-on coding questions** (实践题):
```
> <strong>提示</strong>：这是一道动手实践题，建议实际编写代码

基于 [existing code]，请完成以下扩展实践：
- [task a]
- [task b]
```

**Scenario-based capstone** (综合分析题):
```
某 [company/scenario] 现在希望 [goal]，它需要具备以下功能：
a. [function a]
b. [function b]
c. [function c]
d. [function d]
e. [function e]

此时作为 [role]：
- [decision question a]
- [design question b]
- [evaluation question c]
```

### 4.3 Exercise Examples (Common Patterns)

**Case identification with reasoning**:
```
请分析以下 N 个 case 中的主体是否属于 [concept]，说明理由：
case A：<strong>description</strong>
case B：<strong>description</strong>
...
```

**Choice and justification**:
```
某 [context] 正在考虑两种方案：
方案 A：[description]
方案 B：[description]
请分析：各自的优缺点？什么情况下 A 更合适？什么情况下 B 更有优势？是否存在方案 C 结合两者？
```

**Abstract concept application**:
```
请首先构思一个具体的 [concept] 的落地应用场景，然后说明场景中的：
- 哪些任务应该由 [component A] 处理？
- 哪些任务应该由 [component B] 处理？
- 这两个 [components] 如何协同工作以达成最终目标？
```

### 4.4 Answer Key Policy

- **Chapter exercises**: NO answer keys provided. These are for self-study and community discussion.
- **Extra-Chapter standalone assessments** (like interview Q&A): MAY have answer keys in `Extra01-参考答案.md`.
- Answer key files use a question-answer format:
  ```markdown
  #### <strong>Q: Question text?</strong>
  
  * <strong>参考答案：</strong>
      Answer content here.
  
  ---
  ```

---

## 5. Code Example Standards

### 5.1 Completeness

Every code example must be a **standalone runnable program**, not a snippet. Readers should be able to copy, paste, and run.

```python
# NO: snippet showing only the function
def my_func(x):
    return x * 2

# YES: complete example with imports, execution, and output
import sys

def my_func(x: int) -> int:
    """Multiply input by 2."""
    return x * 2

if __name__ == '__main__':
    result = my_func(21)
    print(f"Result: {result}")
```

### 5.2 Expected Output

Code examples in the chapter text should show expected output:

```
>>>
--- 执行结果 ---
Result: 42
```

Or as a separate bash block showing the full terminal output.

### 5.3 Inline Comments

Every logical block should have a comment explaining **why**, not just what.

```python
# 1. 从环境变量中读取API密钥
api_key = os.environ.get("API_KEY")
if not api_key:
    return "错误：未配置 API_KEY 环境变量。"

# 2. 初始化客户端
client = SomeClient(api_key=api_key)

# 3. 构造查询并发送请求
try:
    response = client.query(query)
    return response.result
except Exception as e:
    return f"错误：{e}"
```

### 5.4 Imports

All imports must be **explicit** and at the top of the file. No `from module import *`.

```python
# CORRECT
import os
import re
from typing import List, Dict, Optional

# INCORRECT
from some_lib import *
```

### 5.5 Error Handling

All I/O operations, network calls, and user-facing code must have `try/except` blocks:

```python
try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.RequestException as e:
    return f"错误：网络请求失败 - {e}"
except (KeyError, IndexError) as e:
    return f"错误：数据解析失败 - {e}"
```

### 5.6 File Organization

Each chapter's code directory should contain:

- Individual `.py` files for each major example or paradigm (e.g., `ReAct.py`, `Plan_and_solve.py`)
- A shared `tools.py` or `llm_client.py` for common utilities
- Optionally `.ipynb` Jupyter notebooks for interactive exploration
- A `__init__.py` if the directory is a Python package

---

## 6. Image Handling

### 6.1 Directory Structure

All images go under `docs/images/`:

```
docs/images/
├── project-logo.png              # Project-level images at root
├── project-logo.png
├── chapter1-figures/             # Per-chapter directories
│   ├── chapter1-1.png            # Or: 1-1.png, 1757242319667-0.png
│   ├── chapter1-2.png
│   └── chapter1-table-1.png      # Complex tables saved as images
├── chapter2-figures/
│   └── ...
```

### 6.2 Naming Convention

Figures within per-chapter directories use consistent naming:

- **Content figures**: `chapterN-seq.png` or `N-seq.png` (e.g., `4-1.png`, `8-3.png`)
- **Table images**: `chapterN-table-seq.png` or `N-table-seq.png` (e.g., `4-4.png` for a comparison table)
- **Opening/conclusion diagrams**: Can use descriptive names or the standard seq pattern

### 6.3 Path References

Always use **relative paths from the markdown file location**:

```markdown
<!-- File at: docs/chapter4/第四章 智能体经典范式构建.md -->
<!-- Image at: docs/images/4-figures/4-1.png -->

<div align="center">
  <img src="../images/4-figures/4-1.png" alt="..." width="90%"/>
  <p>图 4.1 说明文字</p>
</div>
```

Alternatively, absolute GitHub raw URLs are used:
```markdown
<img src="https://raw.githubusercontent.com/open-rdma/gpu-programming-guide/main/docs/images/4-figures/4-1.png" .../>
```

---

## 7. Bilingual Convention

### 7.1 Chapter Files

Each chapter has both Chinese and English versions:

| Language | File Pattern | Example |
|----------|-------------|---------|
| Chinese | `第N章 中文标题.md` | `第1章 初识智能体.md` |
| English | `ChapterN-English-Title.md` | `Chapter1-Meet-Agent.md` |

### 7.2 README Files

| Language | File |
|----------|------|
| Chinese | `README.md` |
| English | `README_EN.md` |

The README header includes a language switcher:

```html
<div align="right">
  <a href="./README_EN.md">English</a> | 中文
</div>
```

### 7.3 Inline Bilingual Text

Within Chinese chapters, technical terms use the pattern: Chinese (English).

```markdown
<strong>流处理器（Streaming Processor, SP）</strong>
```

The English version uses the reverse or presents only the English term, depending on the target audience.

---

## 8. README Structure

Every open-source course README follows this structure:

```markdown
<div align="right">
  <a href="./README_EN.md">English</a> | 中文
</div>

<div align='center'>
  <img src="./docs/images/project-logo.png" alt="alt text" width="100%">
  <h1>Project-Name</h1>
  <h3>🤖 《副标题》</h3>
  <div align="center">
    <!-- Badges: stars, language, online reading, etc. -->
  </div>
  <p><em>一句话介绍项目目标</em></p>
</div>

---

## 🎯 项目介绍
[2-3 paragraphs: problem statement, why this project exists, what it aims to do]

## 📚 快速开始
### 在线阅读
### 本地阅读
### ✨ 你将收获什么？
[Bullet list of learning outcomes]

## 📖 内容导航
[Full chapter table with status badges]

### 社区贡献精选 (Community Blog)
[Table of Extra-Chapter content]

## 💡 如何学习
[Reader guidance: prerequisites, how to use this material, learning path]

## 下一步规划
[Future roadmap]

## 🤝 如何贡献
[Contribution guide]

## 🙏 致谢
### 核心贡献者
### Extra-Chapter 贡献者
### 特别感谢

## Star History

## 读者交流群
[QR code for reader community]

## 关于社区
[Community info and QR code]

## 🎓 引用
[BibTeX citation]

---

## 📜 开源协议
[License: CC BY-NC-SA 4.0]
```

---

## 9. _sidebar.md Format

The navigation sidebar uses standard Markdown nested lists. Each chapter is a link relative to the `docs/` directory:

```markdown
- [Project Name](./README.md)
  - [前言](./前言.md)
  
- <strong>第一部分：Part Name</strong>
  - [第一章 Chapter Title](./chapter1/第一章%20Chapter%20Title.md)
  - [第二章 Chapter Title](./chapter2/第二章%20Chapter%20Title.md)
  - ...

- <strong>第二部分：Part Name</strong>
  - ...
```

Rules:
- Part headings use `<strong>...</strong>` for bold.
- File paths use URL-encoded spaces `%20`.
- No trailing empty lines between part sections.
- Use `- [text](path)` consistently, never `* [text](path)`.

---

## 10. Preface Writing Pattern

The 前言.md follows a specific structure:

```markdown
# 前言

[Opening: 1-2 paragraphs painting the historical context -- why this topic matters now.]

[Problem statement: what gap exists in current learning materials, why a systematic approach is needed.]

[Project vision: what this course aims to provide, the philosophy behind it (e.g., 理论与实战并重).]

[Mission statement: one sentence about the belief in learning by doing.]

## 写给读者的建议

[Warm, encouraging welcome.]

[Prerequisites: what readers should know before starting, in bullet points.]

[Learning path overview: description of each part of the course.]

[Call to action: run the code, experiment, contribute.]

[Closing: thank the reader, wish them well on their learning journey.]
```

Key style notes for the preface:
- First-person plural: 我们 for the author team.
- 本项目 is used self-referentially.
- Classic Chinese idioms and phrases: 纸上得来终觉浅，绝知此事要躬行.
- The tone is inspirational without being overblown.

---

## 11. Extra-Chapter Pattern

### 11.1 Purpose

Extra-Chapter holds community-contributed supplementary content that expands on or is independent of the main course chapters. It is listed in the README as "社区贡献精选".

### 11.2 Naming Convention

```
Extra-Chapter/
├── readme.md                           # Documentation about Extra-Chapter
├── images/                             # Images for extra content
├── Extra01-面试问题总结.md              # Serial numbering: ExtraNN-description
├── Extra01-参考答案.md                  # Answer key pairs with question files
├── Extra02-上下文工程补充知识.md
├── Extra03-Dify智能体创建保姆级操作流程.md
├── Extra04-CommunityFAQ.md
├── Extra05-AgentSkills解读.md
├── Extra06-GUIAgent科普与实战.md
├── Extra07-环境配置.md
└── Extra08-如何写出好的Skill.md
```

### 11.3 Answer Key Pattern

Answer keys are provided ONLY for Extra-Chapter content, never for chapter exercises. The format:

```markdown
# Title

[Brief intro paragraph explaining scope, what is covered, what is not.]

---

### <strong>1. Category Name</strong>

#### <strong>1.1 Question text?</strong>

* <strong>参考答案：</strong>
    [Answer content. Multi-paragraph where needed.]

---
```

---

## 12. Quick Reference Checklist

When writing a new chapter, verify:

- [ ] Chapter file follows the exact template structure (Section 2)
- [ ] All bold text uses `<strong>...</strong>` not `**...**`
- [ ] First occurrence of each technical term uses Chinese (English) format
- [ ] All figures use `<div align="center"><img .../><p>图 N.X ...</p></div>`
- [ ] Figure paths are relative (`../images/chapterN-figures/...`)
- [ ] Complex comparison tables are saved as images
- [ ] Code blocks have language tags and are complete/runnable
- [ ] Code has explicit imports, inline comments, and error handling
- [ ] Exercises start with the disclaimer quote and have escalating difficulty
- [ ] Exercises have NO answer keys (answer keys only for Extra-Chapter content)
- [ ] References section uses numbered format
- [ ] Discussion boilerplate is included at the end
- [ ] Both Chinese and English chapter files are created
- [ ] Code is placed in `code/chapterN/`, not in `docs/`
- [ ] Chapter is added to `_sidebar.md`
- [ ] Chapter is added to README content navigation table
- [ ] Images are organized in `docs/images/chapterN-figures/`

---

## 13. Section Numbering Convention

Chapter sections use hierarchical numbering:

| Level | Format | Example |
|-------|--------|---------|
| Chapter title | `# 第N章 Title` | `# 第一章 初识智能体` |
| Section | `## N.X Title` | `## 1.1 什么是智能体？` |
| Subsection | `### N.X.Y Title` | `### 1.1.1 传统视角下的智能体` |
| Sub-subsection | `#### Title` | `#### 方法一：直接法` |

The "动手体验" section (hands-on practice) is always a numbered section (N.X), not a special prefix.

The "本章小结" (chapter summary) always uses `## N.X 本章小结` format.

"习题" (exercises) always uses `## 习题` format (no section number).

"参考文献" (references) always uses `## 参考文献` format (no section number).

---

## 14. Discussion Boilerplate (Copy-Paste Template)

Add this exact block at the end of every chapter, before the final horizontal rule:

```markdown
---

## 💬 讨论与交流

本章学习过程中遇到问题?想与其他学习者交流心得?

**📝 前往 GitHub Discussions 讨论区:**
- [💬 习题讨论与问答](https://github.com/open-rdma/gpu-programming-guide/discussions)
- 在这里你可以:
  - ✅ 提问习题相关问题
  - ✅ 分享你的解题思路
  - ✅ 与其他学习者交流经验
  - ✅ 获得社区的帮助和反馈

**💡 提示:** 每个页面底部也有评论区,可以直接在页面内讨论!

---
```

Replace the GitHub link with the actual project discussions URL.

---

## References

This style guide is derived from:
- [Hello-Agents](https://github.com/open-rdma/gpu-programming-guide) -- open-source community course on AI Agent construction
- Open-source community writing conventions
