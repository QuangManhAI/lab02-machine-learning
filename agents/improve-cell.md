# Agent Guide: Rewriting Notebook Markdown Cells with Deep Reasoning

## Goal

Rewrite every markdown cell in a Jupyter notebook so each cell delivers **complete understanding**: not just *what* the code does, but *why* it matters, *how* it works, and *when* to use it (or when to worry).

## Core Methodology — The Wh- Reasoning Loop

Before writing a single sentence, ask yourself a chain of questions for each step/cell. Answer them thoroughly, then weave the answers into natural prose — **never** use "What:", "Why:", "How:" as explicit headings.

### 1. What — Define the step

- What is this step doing? One sentence.
- What is the input? What is the output?
- What are the key variables, parameters, or functions?
- What would the output look like?

### 2. Why — Justify its existence

- Why does this step exist? What problem does it solve?
- Why is this approach chosen over alternatives?
- Why is this particular metric/algorithm/parameter appropriate here?
- Why would the reader care about the results?

### 3. How — Explain the mechanism

- How does the code actually work? (algorithm, formula, pipeline)
- How do the pieces fit together? (data flow, function calls)
- How should the reader interpret the output (tables, charts, metrics)?
- How do you distinguish good results from bad ones? (thresholds, benchmarks)

### 4. When — Set expectations and boundaries

- When does this approach work well?
- When does it break down or underperform? (edge cases, caveats)
- When should the reader take action based on the results?
- When would you switch to a different method?

## Writing Rules

### Language & Tone

- Write in **English** (unless otherwise specified).
- Use a technical but natural voice — like explaining to a peer, not teaching a beginner.
- Do **not** use explicit Wh- headings ("What:", "Why:", "How:", "When:"). The answers should be woven into the paragraph flow.

### Structure per Cell

#### `## Step N — Title` (major step)

1. **Opening** — One paragraph stating what this step achieves and why it matters.
2. **Configuration / setup table** — If there are parameters, models, or options, use a Markdown table.
3. **Key insight or caveat** — A paragraph explaining a critical design decision (e.g., shift-16 to prevent leakage, why this metric over another).
4. **Expectation** — What the reader should look for in the output. Include **thresholds** or **decision rules** (e.g., "If A > B, do X; if C < D, something is wrong").

#### `### Step N.M — Sub-Title` (sub-step)

1. **Short opening** — What this specific analysis does.
2. **Brief method** — How it's done (1–2 sentences).
3. **Interpretation guide** — How to read the charts/tables. What patterns to look for.
4. **Expectation** — Specific numbers or patterns expected, and what they imply.

### Formatting

- **Bold** for key terms (model names, metrics, important concepts).
- `code` for variable names, file paths, function names.
- Markdown tables for comparisons (> 2 items).
- **No emoji** unless the notebook already uses them heavily.
- End every cell with an **expectation** or **actionable insight** — never leave the reader hanging.

### Common Pitfalls to Avoid

| Pitfall | Bad | Good |
|---------|-----|------|
| Explicit Wh- headings | "What: we load data. Why: we need it." | "The data lives across 6 CSV files, each serving a distinct purpose..." |
| Code description only | "This cell defines `rmsle()` and `sales_metrics()`." | "RMSLE penalizes relative error — why that matters for right-skewed sales data..." |
| No expectation | *Ends with the table/plot code.* | "If Lag_28 wins, sales have strong weekly seasonality. Baseline RMSLE ~0.5–0.6." |
| Vague guidance | "Look at the plot." | "Points should cluster around y=x. Funnel shape = heteroscedasticity; log transform helps." |
| No decision rules | *No thresholds mentioned.* | "If |correlation| > 0.3, oil is useful; if < 0.1, it adds noise." |

## Example: Before → After

### Before (descriptive only)

```
## Step 6 — Baseline

Baseline models: Lag_16, Lag_28, Rolling_Mean_28_Shift16, Rolling_Mean_56_Shift16.
These are simple forecasts that shift sales by 16 days.
```

### After (deep reasoning)

```
## Step 6 — Baseline Time-Series Models

A baseline establishes a floor for performance.
If a complex ML model cannot beat a naive shift-16 forecast, either features are weak
or the problem is harder than expected.

| Baseline | Formula | When it works best |
| --- | --- | --- |
| **Lag_16** | pred(t) = sales(t-16) | Stable sales with no strong weekly pattern |
| **Lag_28** | pred(t) = sales(t-28) | Strong weekly seasonality |
| **Rolling_Mean_28_Shift16** | mean(sales[t-44:t-16]) | Noisy daily sales needing smoothing |
| **Rolling_Mean_56_Shift16** | mean(sales[t-72:t-16]) | Very noisy series; long-term average is stable |

**Reading the results:**
- If Lag_28 wins → strong weekly seasonality.
- If Rolling_Mean wins → sales are relatively stable.
- If all baselines have similar RMSLE → noise/trend dominates, not seasonality.

**Expectation:** Baseline RMSLE ~0.5–0.6. If ML models cannot get below 0.5, revisit features.
```

## Checklist for Quality

Before finishing, verify each markdown cell:

- [ ] Does the opening explain **why** this step exists?
- [ ] Is the **mechanism** explained (how it works, not just what it does)?
- [ ] Are **parameters/configurations** in a table (if > 2 items)?
- [ ] Does it include **thresholds** or **decision rules** for interpreting output?
- [ ] Is there an **expectation** or **what to look for** at the end?
- [ ] Are there **no explicit Wh- headings** ("What:", "Why:", etc.)?
- [ ] Is **bold** used for key terms, `code` for names?
- [ ] Is the tone **technical but natural** — like explaining to a peer?
- [ ] Would someone reading this know **when to take action** based on the results?
