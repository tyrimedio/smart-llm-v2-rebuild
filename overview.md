# SMART-LLM v2: Multi-Robot Task Planning with Reasoning Models

## Project goal

Rebuild and extend the SMART-LLM framework (Kannan, Venkatesh, Min. Purdue SMART Lab, 2023/2024, arXiv:2309.10062) using current-generation reasoning models, native tool calling, and closed-loop replanning. Target a reproducible delta over the paper's reported numbers on its own benchmark, then push into harder tasks the original benchmark does not cover.

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

Each stage is few-shot prompted with Python scripts. Robot skills are encoded as function definitions, environment state as Python dicts. The authors chose Python over natural language because it elicited better structured output from 2023-era models.

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
[Instruction + Env State]
        |
        v
[Planner: reasoning model + tool-call schemas for robot skills]
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

### Key implementation decisions

1. **Reasoning model as planner**: Claude Opus 4.7 with extended thinking as primary. Comparison runs on o3 / GPT-5 / Gemini 2.5 Pro for cross-model variance. Keep Llama 3.3 70B (or equivalent open-weight) as the open-source baseline.

2. **Native tool calling replaces Python prompts**: Each robot skill becomes a tool with a JSON schema. The model emits structured tool calls rather than generating Python. This removes the paper's "we use Python because natural language doesn't work" trick entirely.

3. **Collapse decomposition + coalition + allocation into one planner call**: Reasoning models can hold the full problem in a single thinking pass. Original paper chained three LLM calls because 2023 non-reasoning models could not reliably do all three at once. Ablate this: run both the one-pass version and a three-call version for comparison.

4. **Verifier as separate pass**: Before execution, a second call (same or different model, zero-shot, structured output) checks that each allocated sub-task has a robot with the required skills, that temporal dependencies are respected, and that parallel sub-tasks do not have state conflicts.

5. **Closed-loop replanning**: Wrap the executor in a monitor that catches skill failures (e.g., `PickUp` returns False because the object is out of reach). On failure, invoke the planner again with the failure context appended. Budget N replans per task before declaring failure.

6. **Reproduce before extending**: Implement the paper's exact four-stage pipeline first and reproduce their Table I numbers on their 36 tasks. This is the control condition. Every subsequent change is measured as a delta against this baseline.

### Tech stack

- Python 3.11+
- `anthropic` SDK for Claude calls
- `openai` SDK for GPT-5 / o3 comparison
- `google-generativeai` for Gemini
- AI2-THOR (match the version the paper used; verify their repo)
- `pydantic` for tool schemas
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
      planner.py         # main reasoning-model planner
      verifier.py        # plan validator
      executor.py        # AI2-THOR runner
      monitor.py         # failure detection + replan trigger
    env/
      ai2thor_wrapper.py
      state_extractor.py
    benchmark/
      tasks.py           # 36 benchmark tasks ported from paper
      metrics.py         # SR, TCR, GCR, RU, Exe implementations
      runner.py          # main experiment loop
  experiments/
    01_reproduce_paper.py
    02_toolcall_swap.py
    03_reasoning_model_swap.py
    04_add_verifier.py
    05_closed_loop.py
    06_cross_model_variance.py
  results/
    <timestamp>/
      config.json
      per_task_logs/
      summary.csv
```

## First milestones (ordered)

1. **Scaffold + AI2-THOR setup**: Clone paper's repo, confirm their benchmark is runnable. Set up project structure. Get AI2-THOR running locally. Confirm metric computation against a known task.

2. **Baseline reproduction**: Implement the paper's four-stage pipeline with GPT-4 (or closest equivalent available in April 2026). Run full 36-task benchmark. Target numbers within a few points of Table I. This is the control.

3. **Tool-calling swap**: Replace Python-prompt trick with native tool calls. Same model. Run benchmark. Isolates the tool-calling contribution.

4. **Reasoning model swap**: Point the planner at Claude Opus 4.7 with extended thinking. Same tool-calling interface. Run benchmark with 5 seeds per task to measure variance reduction.

5. **Verifier addition**: Bolt on the verifier pass. Run benchmark again.

6. **Closed-loop replanning**: Add monitor + replan trigger. Run benchmark with injected failures (artificial object occlusion, drop events) to actually exercise the loop.

7. **Cross-model comparison**: Run full v2 stack on o3, GPT-5, Gemini 2.5 Pro, Llama 3.3 70B. Report per-model SR and variance.

8. **Harder benchmark** (stretch): If the 36-task benchmark saturates at near-1.0 SR, design extensions. Candidates: adversarial instructions, stochastic object placement, mid-task robot dropout, underspecified instructions requiring clarification.

## Constraints and preferences

- **Hardware**: Dev on standard laptop/desktop. Stretch goal: deploy a small-model variant (Llama 3.1 8B or Gemma 2 on Raspberry Pi 5 + AI HAT) for an "edge multi-robot planner" demo.
- **Code style**: Clean, typed (mypy or pyright strict), pytest-covered, well-documented. Portfolio piece and grad-app material. Readability matters more than cleverness.
- **Paper-matching**: Use the authors' original task descriptions and ground-truth states verbatim where available. Cite them. Frame as an extension, not a replacement.
- **Cost awareness**: Reasoning model calls at scale are expensive. Cache prompts and responses aggressively. Persist every API call so reruns do not require re-billing.

## Known open questions (resolve before coding)

1. Does the paper's repo expose the full 36-task definitions and ground-truth state dictionaries, or only a subset? If only a subset, we need to reconstruct the rest from the paper's descriptions.
2. What version of AI2-THOR does the paper target, and does it still install cleanly in April 2026?
3. Sim-only, or also reproduce the real-robot visibility-coverage experiments?
4. Single multi-tool planner vs. one MCP server per robot: treat as an ablation or pick one?
5. API budget ceiling for benchmark runs (36 tasks × 5 seeds × N replans × cross-model = not trivial).

## References

- SMART-LLM paper: https://arxiv.org/abs/2309.10062
- Project page: https://sites.google.com/view/smart-llm/
- AI2-THOR: https://ai2thor.allenai.org/
