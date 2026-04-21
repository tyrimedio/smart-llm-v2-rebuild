# SMART-LLM v2: Multi-Robot Task Planning with Reasoning Models

## Project goal

Rebuild and extend the SMART-LLM framework (Kannan, Venkatesh, Min. Purdue SMART Lab, 2023/2024, arXiv:2309.10062) using current-generation reasoning models, native tool calling, structured JSON plans, and closed-loop replanning. Target a reproducible delta over the paper's reported numbers on its own benchmark, then push into harder tasks the original benchmark does not cover.

Primary hypothesis: the paper's complex-task success rate of 0.71 (GPT-4 backbone) and high variance (0.48 ± 0.40 on repeated complex-task runs) are bottlenecked by three things that 2026-era models and patterns fix.

1. Single-pass chaining through non-reasoning models at each stage
2. Pythonic-prompt workaround for structured output (unnecessary with native tool calling)
3. Open-loop execution with no verifier and no failure recovery

## Original paper summary

SMART-LLM decomposes a natural-language instruction into a multi-robot plan through four sequential LLM stages.

1. **Task Decomposition**: LLM receives instruction + environment description + robot skill list, outputs sub-tasks as Python code using `threading` for parallel sub-tasks.
2. **Coalition Formation**: LLM matches sub-task skill requirements against individual robot capabilities and forms teams where no single robot satisfies the requirement.
3. **Task Allocation**: LLM produces final executable Python with specific robots assigned to specific sub-tasks.
4. **Execution**: Interpreter runs the code against AI2-THOR sim or real robots via API calls to low-level skills.

Each stage is few-shot prompted with Python scripts. Robot skills are encoded as function definitions, environment state as Python dicts. The authors chose Python over natural language because it elicited better structured output from 2023-era models. That choice is useful as historical context and as an optional control ablation, but it is not the target architecture for v2.

### Benchmark

36 tasks in AI2-THOR across four complexity tiers.

- Elemental (6): single robot with all required skills
- Simple (8): multi-object, pure sequential or pure parallel decomposition
- Compound (14): heterogeneous robots, mixed sequential/parallel
- Complex (8): requires multi-robot teaming because no single robot has all needed skills

### Metrics

- **SR** (Success Rate): 1 if GCR and RU both equal 1, else 0
- **TCR** (Task Completion Rate): 1 if GCR = 1, else 0
- **GCR** (Goal Condition Recall): fraction of ground-truth end-state conditions achieved
- **RU** (Robot Utilization): 1.0 when execution transition count matches ground truth. Falls toward 0 when parallelizable sub-tasks are executed sequentially.
- **Exe** (Executability): fraction of planned actions that can actually execute

### Reported numbers (GPT-4 backbone, their Table I)

| Category  | SR   | TCR  | GCR  | RU   | Exe  |
|-----------|------|------|------|------|------|
| Elemental | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Simple    | 0.62 | 1.00 | 1.00 | 0.62 | 1.00 |
| Compound  | 0.69 | 0.76 | 0.85 | 0.92 | 1.00 |
| Complex   | 0.71 | 0.85 | 0.92 | 1.00 | 0.97 |

Their variability table (5 seeds per category) shows complex tasks at SR 0.48 ± 0.40, meaning the method fails about half the time on complex tasks and results are highly unstable.

Repo, dataset, videos: https://sites.google.com/view/smart-llm/

## V2 design

### Architecture

```
[Instruction + Env State + Robot Catalog]
        |
        v
[Planner: reasoning model + structured JSON output]
        |
        v
[Plan Schema: phases, robot teams, action args, constraints]
        |
        v
[Verifier: checks plan feasibility against skill sets, constraints, temporal deps]
        |
     ---+---
    fail    pass
     |       |
[revise]  [Executor: invokes robot skills against AI2-THOR]
     ^       |
     |       v
     +--[Monitor: detects skill failures, state mismatches, timeouts]
```

The canonical interface in v2 is a typed JSON plan, not generated Python. The benchmark should stay faithful to the paper's tasks and metrics, but the planner should use the best available structured-output path rather than preserve a 2023 workaround.

### Key implementation decisions

1. **Reasoning model as planner**: Claude Opus 4.7 with extended thinking as primary. Comparison runs on o3 / GPT-5 / Gemini 2.5 Pro for cross-model variance. Keep Llama 3.3 70B (or equivalent open-weight) as the open-source baseline.

2. **Structured JSON is the plan contract**: Each robot skill is represented by a typed action schema. The model returns parseable plan data directly instead of generating Python. This removes the paper's "we use Python because natural language doesn't work" trick and makes validation, caching, and replanning much simpler.

3. **One-pass reasoning planner is the default**: Reasoning models can hold decomposition, coalition formation, and allocation in a single pass. The paper chained three LLM calls because 2023 non-reasoning models could not reliably do all three at once. Keep a staged version only if we want an ablation.

4. **Verifier as separate pass**: Before execution, a second call (same or different model, zero-shot, structured output) checks that each allocated sub-task has a robot with the required skills, that temporal dependencies are respected, and that parallel sub-tasks do not have state conflicts.

5. **Closed-loop replanning**: Wrap the executor in a monitor that catches skill failures (e.g., `PickUp` returns False because the object is out of reach). On failure, invoke the planner again with the failure context appended. Budget N replans per task before declaring failure.

6. **Reproduce the benchmark, not the workaround**: The control condition is the paper's task suite, environment assumptions, and metrics. We do not need to preserve Pythonic prompting or exact prompt chaining unless we explicitly want a legacy ablation. Every subsequent change is still measured as a delta against the paper's reported benchmark numbers.

7. **Long context and multimodality are future extensions, not blockers**: Modern long-context models let us include the full skill library, richer environment traces, and prior failures in-context. Native vision and VLM/VLA execution are promising extensions once the structured planner loop is stable.

8. **Per-robot MCP is an ablation, not the first milestone**: Exposing each robot as its own MCP server is a clean way to test decentralized coalition formation later. The first implementation should stay centralized and typed so the benchmark loop is controllable.

### Tech stack

- Python 3.11+
- `anthropic` SDK for Claude calls
- `openai` SDK for GPT-5 / o3 comparison
- `google-generativeai` for Gemini
- AI2-THOR (match the version the paper used; verify their repo)
- `pydantic` for structured plan schemas
- `pytest` for benchmark harness
- `structlog` with JSON output, one log per task run

Optional but worth considering:
- `LangGraph` if we want explicit state-machine orchestration over the verify/replan loop
- MCP server wrapper per robot as a separate ablation (tests whether per-robot agents negotiate better than a single centralized planner)

### Proposed file structure

```
smart-llm-v2/
  README.md
  pyproject.toml
  smart_llm_v2/
    __init__.py
    skills/              # robot skill definitions as tool schemas
      base.py
      mobile.py
      manipulation.py
    agents/
      planner.py         # planner interfaces
      json_planner.py    # primary structured-output planner
      verifier.py        # plan validator
      executor.py        # AI2-THOR runner
      monitor.py         # failure detection + replan trigger
      legacy/            # optional paper-style prompt path for ablations only
    env/
      ai2thor_wrapper.py
      state_extractor.py
    benchmark/
      tasks.py           # 36 benchmark tasks ported from paper
      metrics.py         # SR, TCR, GCR, RU, Exe implementations
      runner.py          # main experiment loop
  experiments/
    01_structured_control.py
    02_reasoning_model_swap.py
    03_add_verifier.py
    04_closed_loop.py
    05_cross_model_variance.py
    06_legacy_prompt_ablation.py
  results/
    <timestamp>/
      config.json
      per_task_logs/
      summary.csv
```

## First milestones (ordered)

1. **Scaffold + AI2-THOR setup**: Clone paper's repo, confirm their benchmark is runnable. Set up project structure. Get AI2-THOR running locally. Confirm metric computation against a known task.

2. **JSON plan contract**: Define the canonical structured plan schema for phases, robot teams, and action arguments. Make this the primary planner-executor boundary.

3. **Structured control run**: Implement a centralized planner that targets the paper's 36 tasks and metrics using structured outputs instead of generated Python. Run the benchmark and compare against Table I. This is the main control condition for v2.

4. **Reasoning model swap**: Point the planner at Claude Opus 4.7 with extended thinking. Keep the same structured interface. Run benchmark with 5 seeds per task to measure variance reduction.

5. **Verifier addition**: Bolt on the verifier pass. Run benchmark again.

6. **Closed-loop replanning**: Add monitor + replan trigger. Run benchmark with injected failures (artificial object occlusion, drop events) to actually exercise the loop.

7. **Cross-model comparison**: Run full v2 stack on o3, GPT-5, Gemini 2.5 Pro, Llama 3.3 70B. Report per-model SR and variance.

8. **Legacy prompt ablation** (optional): If we want a direct historical comparison, isolate the paper-style staged Python prompt path and run it as a legacy ablation. This is not the main implementation path.

9. **Harder benchmark** (stretch): If the 36-task benchmark saturates at near-1.0 SR, design extensions. Candidates: adversarial instructions, stochastic object placement, mid-task robot dropout, underspecified instructions requiring clarification.

## Constraints and preferences

- **Hardware**: Dev on standard laptop/desktop. Stretch goal: deploy a small-model variant (Llama 3.1 8B or Gemma 2 on Raspberry Pi 5 + AI HAT) for an "edge multi-robot planner" demo.
- **Code style**: Clean, typed (mypy or pyright strict), pytest-covered, well-documented. Portfolio piece and grad-app material. Readability matters more than cleverness.
- **Paper-matching**: Use the authors' original task descriptions and ground-truth states verbatim where available. Cite them. Frame as an extension, not a replacement. Match the benchmark, not necessarily the outdated prompt schema.
- **Cost awareness**: Reasoning model calls at scale are expensive. Cache prompts and responses aggressively. Persist every API call so reruns do not require re-billing.

## Known open questions (resolve before coding)

1. Does the paper's repo expose the full 36-task definitions and ground-truth state dictionaries, or only a subset? If only a subset, we need to reconstruct the rest from the paper's descriptions.
2. What version of AI2-THOR does the paper target, and does it still install cleanly in April 2026?
3. Sim-only, or also reproduce the real-robot visibility-coverage experiments?
4. Single centralized structured planner vs. one MCP server per robot: treat as an ablation or pick one?
5. How much of the roadmap should stay symbolic, and when should vision or VLA execution enter the stack?
6. API budget ceiling for benchmark runs (36 tasks × 5 seeds × N replans × cross-model = not trivial).

## References

- SMART-LLM paper: https://arxiv.org/abs/2309.10062
- Project page: https://sites.google.com/view/smart-llm/
- AI2-THOR: https://ai2thor.allenai.org/
