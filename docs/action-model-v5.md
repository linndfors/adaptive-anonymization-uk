# Action-regression model

The active method is `adaptive_resume_v5_action_regression`. It combines typed
fragment extraction, deterministic text transformations, a supervised action
regressor and a constrained greedy controller. The model and fitted training
statistics are loaded from one artifact. New-resume inference uses no external
resume collection or retrieval index.

## Offline learning

Reserve whole people for training, validation and testing before fitting.
Another resume of the same person is the retrieval target; exclude the query's
own record. BM25 and character TF-IDF attackers measure candidate text states
in reference populations of different sizes.

For state `s`, edit `e`, reference population `v` and attacker `a`:

```text
T(s) = mean_v max_a [0.65 × reciprocal_rank(a, v, s)
                     + 0.35 × bounded_score_separation(a, v, s)]
target_gain(s, e) = T(s) − T(apply(s, e))
```

Bounded score separation compares the strongest same-person target with the
strongest other-person competitor. It provides a learning signal when an edit
changes scores without changing rank. These are declared vulnerability proxies,
not calibrated probabilities of identification.

Features describe the current state, proposed edit and affected values: types,
groups, scope, strength, lengths, date/numeric counts, train-fitted frequencies
and IDF, 32 TF-IDF/LSA coordinates and information-loss estimates. Person IDs,
live retrieval scores and reference documents are excluded from inference
features. Fitting balances people so resumes with more candidate edits do not
receive greater total weight.

The estimator is `HistGradientBoostingRegressor` after dictionary vectorization.
Candidate models are evaluated using validation decision regret: the measured
net benefit lost by the selected action compared with the best available
action, including choosing no action. Optional on-policy aggregation adds
candidate outcomes from states visited by the parent controller. Only training
labels fit the model; validation labels select models and operating points.

The frozen model uses 160 iterations, at most 31 leaves, a minimum of 40 rows
per leaf, learning rate 0.05 and L2 regularization 5. Its controller settings
are stored in the model artifact and its selection record.

## Inference

1. Parse headings and detect typed fragments using local regex, NER and context rules.
2. Mandatorily mask direct identifiers and sensitive fragments.
3. Enumerate stronger group and linked-fragment transformations.
4. Predict each candidate's gain and subtract its information-loss cost.
5. Apply the best feasible candidate with positive predicted net benefit.
6. Repeat until no beneficial candidate remains, constraints block further
   actions or the step limit is reached.
7. Render edits against the original text and check supported direct patterns.

Available strengths are keeping, category generalization, broader
generalization and typed masking. Group actions affect an entire field;
fragment actions affect linked occurrences of a value within that group.
Professional fields are eligible for transformation.

The selected controller maximizes:

```text
predicted_gain − 0.4 × cost
cost = 0.5 × positive_lexical_loss + 0.5 × positive_detail_loss
```

Its lexical floor is 0.55, typed-detail floor 0.35, maximum actions 12 and
fragment candidate limit eight classes in addition to group actions. The
minimum net-gain parameter is zero. Typed detail assigns less credit to
generalization and none to masking; it is an internal constraint, not an
independent factual evaluation.

Stopping does not establish anonymity. A correct detected span can still
receive an unsuitable category, and a well-formed result can remain linkable
through combinations of professional facts.

## Evaluation

- **Privacy:** same-person Recall@1 and MRR under BM25, character TF-IDF and an
  additional E5 semantic attacker. Lower values mean less successful linkage
  within the defined corpus scenario.
- **Lexical utility:** TF-IDF cosine between original and result. It is not the
  fraction of retained words or facts.
- **Semantic utility:** frozen multilingual embeddings for full text and
  source-anchored professional fields, processing all token windows.
- **Uncertainty:** paired bootstrap resampling of people, keeping their resumes together.

Fixed L2/L3 use the historical field renderer. Uniform controls use the current
parser and transformation tools. Fixed-order controls share candidate actions
and constraints. The type-prior control replaces contextual regression with
mean training gain by action group, strength and scope. Search controls have
live-corpus access; learned controllers do not.

Final test results must stay separate from validation selection, template
diagnostics and repeated software checks. Recorded scores and outputs are
external experiment deliverables. Commands and required inputs are documented
in [reproduction](reproduction.md).

## Main implementation files

| Component | Source |
|---|---|
| Artifact, features and controller | `src/anonrisk_uk/action_resume.py` |
| Local resource loading | `src/anonrisk_uk/resume_resources.py` |
| Typed parser and context rules | `src/anonrisk_uk/resume_parser.py`, `resume_context.py`, `resume_fields.py` |
| Transformations and rendering | `src/anonrisk_uk/resume_adaptive.py`, `category_taxonomy.py` |
| Attackers and ranking | `src/anonrisk_uk/attack.py` |
| Data reservation | `scripts/action_data.py` |
| Training and selection | `scripts/train_action_resume.py`, `train_action_onpolicy.py`, `select_action_operating_point.py` |
| Corpus evaluation | `scripts/evaluate_action_resume.py` |
| Reproduction | `scripts/frozen_runtime.py`, `reproduce_frozen.py` |
