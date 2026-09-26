# CodeSentinel-AI — Research Contribution Assessment

**Last updated:** 19 Sep 2026
**Status:** interim — final training in progress, test split not yet evaluated

This document separates what was *proposed* from what was *built*, and what is
*evidenced* from what is *asserted*. Where the implemented system departs from
the original proposal, the deviation is stated plainly with its technical
reason and its effect on the research scope.

> Narrative reports live in `reports/`. Machine-generated artefacts (leakage
> JSON, tuning trials, experiment logs) live in `artifacts/` and are
> git-ignored.

---

## 1. Problem statement

Automated detection of security vulnerabilities in source code at function
granularity. The operational difficulty is not classification accuracy in the
abstract but **extreme class imbalance** (~3% positive in PrimeVul) combined
with **distribution shift between splits**, which together make headline
accuracy meaningless and make threshold choice a first-class decision rather
than an afterthought.

---

## 2. Existing approaches this builds on

| Approach | Status here |
|---|---|
| Pattern/rule-based static analysis | Implemented (724 lines), measured |
| Taint analysis with sources/sinks/sanitizers | Implemented (1,001 lines), measured |
| AST / CFG / DFG / CPG construction | Implemented (~1,900 lines) |
| Transformer code models (CodeBERT family) | Implemented, hierarchical chunking for long functions |
| Graph neural networks over code (GATv2/R-GCN) | **NOT IMPLEMENTED** — see §4 |

None of these is novel in itself. Combining AST, CFG and DFG into a code
property graph is established practice (Yamaguchi et al.), and this project
makes **no novelty claim** for doing so.

---

## 3. Deviation 1 — CodeBERT instead of GraphCodeBERT

**Proposed:** GraphCodeBERT.
**Implemented:** `microsoft/codebert-base`.
**Status: DEVIATION, documented, not hidden.**

### Technical reason

GraphCodeBERT's advantage over CodeBERT comes from pre-training objectives
that consume an explicit **data-flow graph** alongside the token sequence —
edge prediction and node alignment over variable-level data flow. To benefit
from that at fine-tuning time, the model must be fed the same auxiliary input:
a DFG for each function, aligned to token positions.

This pipeline does not currently supply that. `chunk_dataset.py` produces
**token-id chunks only**. The repository does have a DFG service
(`dfg_service.py`, 377 lines), but it is a per-request analyzer for the API,
not an offline stage that emits token-aligned data-flow edges for 174k
functions.

Swapping the checkpoint name without supplying data-flow inputs would load
GraphCodeBERT's weights and then use them **exactly as a CodeBERT-style
sequence encoder**. That would be entitled to the label "GraphCodeBERT" in a
paper while providing none of the mechanism the label implies — a misleading
result, not a better one.

### Cost of doing it properly

Token-aligned DFG extraction for the full corpus, plus re-chunking,
re-tuning and re-training: several days on this hardware.

### Effect on scope

The ML component is a **sequence model over code tokens with hierarchical
chunk aggregation**, and is described that way throughout. No result in this
project may be reported as a GraphCodeBERT result.

---

## 4. Deviation 2 — GATv2 is NOT IMPLEMENTED

**Proposed:** GATv2 graph neural network over a security-aware code graph.
**Implemented:** nothing.
**Status: NOT IMPLEMENTED.**

### Evidence of absence

```
backend/ml/graphs/__init__.py      0 bytes
backend/ml/embeddings/__init__.py  0 bytes
torch_geometric                    not installed
```

No graph tensors are constructed anywhere in the ML pipeline. No GATv2 layer,
pooling, batching or training code exists.

### Why it was not rushed

Delivering GATv2 credibly requires, in order:

1. An **offline** graph-extraction stage converting 174,432 functions from the
   per-request CPG into serialised graphs — a pipeline that does not exist and
   is comparable in size to the chunking stage that already took hours.
2. A node-feature scheme (token embeddings? type one-hots? learned?) that is a
   research decision in its own right.
3. Edge typing across AST/CFG/DFG plus security metadata.
4. Batching of variable-sized graphs, then training a **second** model on a
   4 GB GPU that is already unstable under sustained load — two CUDA-level
   crashes have occurred during this project.
5. Evaluation, tuning and ablation to make any reported number meaningful.

A version of this rushed in a few hours would produce a model that runs but
whose reported numbers no one should trust. Per the project rules — *"Do not
rush an unvalidated graph pipeline"* and *"Never claim GATv2 was implemented if
it was not"* — it is recorded as absent.

### How excluding GATv2 changes the original research scope

This is a **material narrowing**, and four specific claims are withdrawn:

| Originally in scope | Now |
|---|---|
| "CodeBERT + GATv2 hybrid architecture" | **Withdrawn.** The system is CodeBERT + rules + taint. |
| Graph-vs-sequence comparison (Experiment D) | **Not runnable.** No graph model exists to compare. |
| Security-aware graph ablations (Experiments F–I: without-CFG, without-DFG, without-security-metadata) | **Not runnable as ML ablations.** These measure a graph model's inputs. The graph *construction* exists and is used by the rule/taint path, but nothing learns from it. |
| "Novel security-aware graph representation for GNN vulnerability detection" | **Withdrawn.** The graph is built and used for evidence linking, never as model input. |

What survives is a narrower but defensible scope: **a hierarchical
transformer classifier for long functions, evaluated honestly against
rule-based and taint-based baselines on a leakage-audited PrimeVul split.**

---

## 5. What is genuinely contributed

These are empirical findings, measured on real data, and they do not depend on
GATv2 existing. They are the strongest material this project currently has.

### 5.1 PrimeVul contains cross-split leakage its own de-duplication missed

**Evidence** (`artifacts/reports/leakage_raw.json`):

| Pair | exact_code | **normalized_code** | PrimeVul `hash` |
|---|---|---|---|
| train↔valid | 0 | **1,310** | 0 |
| train↔test | 0 | **796** | 0 |

PrimeVul deduplicated on *exact* content. **796 test functions (3.07%) are
whitespace-variants of training functions** and are invisible to exact-content
hashing — including PrimeVul's own `hash` field, which reports clean.

**Why it matters:** published PrimeVul results computed without normalized-code
de-duplication include a small but non-zero memorisation component. Removing it
cost 1.3% of the training split.

**Limitation:** whitespace normalisation catches copies, not semantic clones,
so 796 is a *lower bound*.

### 5.2 Fixed chunk caps are biased against the positive class

**Evidence** (full-population counts over every record in each split; the
token-loss figure is from a 5,176-function sample of train):

| Metric | Value |
|---|---|
| Functions truncated at 8 chunks | 2.97% train / 2.90% test (full population) |
| **Corpus tokens discarded** | **22.5%** |
| **Vulnerable functions truncated** | **19.24%** train / **20.86%** test (full-population) |
| Benign functions truncated | **2.43%** train / **2.40%** test |

Vulnerable functions are ~4× longer and truncate **7.9-8.7x more often**. A
truncation policy that looks negligible per-function (~2.9%) discards 22.5% of
all tokens and does so **disproportionately from the class being detected**,
placing a hard ceiling on achievable recall.

**Why it matters:** long-function truncation is usually reported, if at all, as
a single corpus-wide percentage. The per-class asymmetry is the part that
affects recall, and it is not visible in that statistic.

**Limitation:** measured on one dataset with one tokenizer and one cap; the
mechanism should generalise, the magnitude is not claimed to.

### 5.3 Metric choice materially changes the reported conclusion

**Evidence** (Stage 12 smoke run): the same model simultaneously scored
**ROC-AUC 0.8277** and **PR-AUC 0.1399**, while predicting **zero positives**
at threshold 0.5.

A ROC-AUC-only report would describe this as a good model. It detects nothing.

**Limitation:** this is a well-known property of ROC under imbalance. The
contribution is the concrete demonstration on this dataset, not the insight.

### 5.4 Measured baselines most papers omit

`rules_only` on the full test split: **F1 0.1065**, PR-AUC 0.0378 against a
0.0268 chance baseline — only **1.41× chance**. `taint_only` flags 4 of 25,911
functions and catches **zero** true positives.

Publishing the non-ML baseline on the same split is uncommon and makes any ML
improvement interpretable rather than free-floating.

---

## 6. Claims that remain unverified

| Claim | Why unverified |
|---|---|
| The ML model outperforms `rules_only` | Final training in progress; test split not evaluated |
| Any calibration or threshold result on real data | Stages 15–16 not yet run against a trained model |
| Hybrid ML+rules fusion improves over either alone | Requires the trained model |
| Any GATv2 or graph-model result | Not implemented |
| Any GraphCodeBERT result | Not implemented |

---

## 7. Threats to validity

1. **Distribution shift.** Train is dominated by CWE-119/20/264 in
   linux/Chrome/qemu; test by CWE-617 in vim/gpac/tensorflow. Test measures
   generalisation under shift, not i.i.d. performance.
2. **Project overlap.** 149 projects appear in both train and test (21,287 test
   records), so held-out metrics partly measure *within-project* generalisation.
   Reported as a concern, not removed — excluding it would delete 73% of train.
3. **Residual valid↔test overlap.** 14 functions (0.054% of test) remain shared.
   Not train-on-test leakage, but it slightly couples threshold selection to test.
4. **Undersampling.** Final training uses 10:1 negatives for compute budget.
   This shifts predicted probabilities (a calibration effect, not a ranking one)
   and is corrected on validation, which retains the true base rate.
5. **Single dataset, single seed.** No cross-dataset validation, no seed variance.
6. **Small tuning budget.** 4 trials × 1 epoch on a 9,000-function subsample.
7. **Hardware instability.** Two CUDA-level failures during this project;
   results come from runs that survived, which is a weak form of selection.

---

## 8. Honest assessment for a paper

**Sufficient for a paper draft?** Not yet — the test evaluation has not run.

**If the remaining stages complete, what would be publishable?** A short
empirical/reproducibility paper about *measurement practice* on PrimeVul:
leakage that survives the dataset's own de-duplication, class-asymmetric
truncation bias, and the ROC/PR divergence, with rule and taint baselines that
are usually absent.

**What would not be publishable:** a novel-architecture paper. The originally
proposed contribution was the security-aware graph plus GATv2, and neither is
implemented. Presenting this work as that contribution would misrepresent it.

No claim is made about IEEE acceptance or about novelty beyond what the
evidence above supports.
