# scCChain Branding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace prose-only `ScCChain` branding with `scCChain` while keeping Julia/package/repository identifiers unchanged.

**Architecture:** This is a narrow content cleanup across docs, comments, and notebook narrative text. Verification relies on targeted searches before and after the edits to ensure only intentional `ScCChain` identifier usages remain.

**Tech Stack:** Markdown, Julia source comments, Python doc/comments, Jupyter notebook JSON, `rg`

---

### Task 1: Verify the current prose occurrences

**Files:**
- Modify: `README.md`
- Modify: `analysis/figure_2.jl`
- Modify: `analysis/figure_3_4.jl`
- Modify: `tutorials/01_preprocessing_example_data.ipynb`
- Modify: `tutorials/02_loading_LRpair_database.ipynb`

**Step 1: Write the failing test**

Use targeted searches as the failing check for this content-only change.

**Step 2: Run test to verify it fails**

Run: `rg -n "ScCChain \\(\\*\\*s\\*\\*ingle|ScCChain uses PythonCall|If you use ScCChain|new ScCChain API" README.md analysis/figure_2.jl analysis/figure_3_4.jl`
Expected: matches are found

Run: `rg -n '"[^"]*ScCChain[^"]*"' tutorials/01_preprocessing_example_data.ipynb tutorials/02_loading_LRpair_database.ipynb`
Expected: narrative notebook strings still include `ScCChain`

### Task 2: Apply the text-only rename

**Files:**
- Modify: `README.md`
- Modify: `analysis/figure_2.jl`
- Modify: `analysis/figure_3_4.jl`
- Modify: `tutorials/01_preprocessing_example_data.ipynb`
- Modify: `tutorials/02_loading_LRpair_database.ipynb`

**Step 1: Write minimal implementation**

- Change prose `ScCChain` branding to `scCChain`.
- Leave `using ScCChain`, `Pkg.add(...)`, repository URLs, module names, and file names unchanged.
- In notebooks, update only narrative strings and leave executable cells untouched.

**Step 2: Run test to verify it passes**

Run: `rg -n "ScCChain \\(\\*\\*s\\*\\*ingle|ScCChain uses PythonCall|If you use ScCChain|new ScCChain API" README.md analysis/figure_2.jl analysis/figure_3_4.jl`
Expected: no matches

Run: `rg -n "ScCChain|scCChain" README.md analysis tutorials | sed -n '1,120p'`
Expected: remaining `ScCChain` hits are limited to package/repository identifiers and command examples

### Task 3: Final verification

**Files:**
- Modify: `docs/plans/2026-03-17-sccchain-branding-design.md`
- Modify: `docs/plans/2026-03-17-sccchain-branding.md`

**Step 1: Run verification**

Run: `git diff -- README.md analysis/figure_2.jl analysis/figure_3_4.jl tutorials/01_preprocessing_example_data.ipynb tutorials/02_loading_LRpair_database.ipynb docs/plans/2026-03-17-sccchain-branding-design.md docs/plans/2026-03-17-sccchain-branding.md`
Expected: only prose/commentary/doc changes

**Step 2: Commit**

Skipped in this pass because the user explicitly requested no commit yet.
