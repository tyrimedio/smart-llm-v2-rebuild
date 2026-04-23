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

The implemented path today is narrower than the target diagram: provider-backed planning, JSON plan validation, baseline execution, and benchmark metrics are in place. A hybrid verifier exists behind the optional runner hook, but the structured-control experiment intentionally leaves it off so its archived metrics stay raw planner plus executor metrics. The monitor/replan loop is still a planned layer, not a finished module.

### Current implementation snapshot

- `BenchmarkRunner`, `tasks.py`, and `metrics.py` already cover the paper's 36-task dataset snapshot and scoring surface.
- `JsonPlanner` is the primary planner path. It validates structured plans into typed dataclasses (`TaskPlan`, `PlanPhase`, `ActionRequest`) instead of generated Python.
- Provider-backed planning is wired for Claude Opus 4.7, GPT-5.4 (the ChatGPT 5.4 family), and Kimi K2.6. Claude uses Anthropic's Messages API, while GPT-5.4 and Kimi use an OpenAI-compatible chat-completions tool-calling path.
- Both symbolic and multimodal planner variants exist. The multimodal variant attaches egocentric AI2-THOR frames as planning images.
- `BaselineExecutor` and `Ai2ThorEnvironment` run the current control path in simulation.
- `paper_planner.py` exists as the legacy staged ablation path, but it is not the main planner and the benchmark work is still centered on the JSON-first path.
- `verifier.py` is available as a hybrid pre-execution check: deterministic rules catch obvious conflicts first, then a provider-backed semantic verifier reviews the surviving plan through the same model profile as the planner. It is not wired into `01_structured_control.py`.
- `monitor.py` and closed-loop replanning are still roadmap items.

### Design note: benchmark fidelity vs paper inheritance

The paper gives us the benchmark and the comparison target. It does not define the best architecture for 2026-era planning models. For v2, we should preserve the paper where it defines the evaluation surface, and retire it where it reflects model or tooling limits from 2023.

Keep these for benchmark fidelity:

- The 36-task AI2-THOR benchmark categories and task instructions
- The reported evaluation surface: SR, TCR, GCR, RU, and Exe
- Heterogeneous robot capability matching and multi-robot teaming as the core planning problem
- Fixed low-level skill APIs for the control condition, so we can compare planning quality without changing the executor problem
- Symbolic task goals and scene-state reporting as the baseline benchmark interface

Treat these as paper-era choices to retire in the mainline system:

- Python few-shot prompting and generated executable code
- Multi-stage LLM chaining for decomposition, coalition formation, and allocation as the default path
- `threading`-style code generation as the way to express parallelism
- LLM-only coalition reasoning where deterministic validation can do the job more reliably
- Open-loop execution with no verifier, no monitor, and no recovery path
- "Use the minimum number of robots" as the default planning objective when it conflicts with success margin or makespan
- Phase-only execution as the long-term runtime model. Phases are acceptable for control experiments and RU reporting, but the long-term executor should move toward dependency-aware scheduling with explicit resource and ordering constraints.

Current stance:

- Preserve the benchmark, task suite, and metrics exactly enough to support direct comparison to SMART-LLM.
- Replace prompt-shape and orchestration decisions when better modern alternatives exist.
- Keep legacy prompt logic only as an ablation, not as the main product path.
- Treat symbolic-only planning as the control condition, not the ceiling. Multimodal context and richer monitoring should improve the system once the control run is stable.

### Scheduling contract: phases, sub-tasks, and actions

Use the paper's benchmark semantics for reporting, but do not collapse all scheduling concepts into one field.

The SMART-LLM paper decomposes a task into temporally ordered sub-tasks. Some sub-tasks can share the same temporal precedence and execute in parallel. Each sub-task still contains an ordered sequence of low-level robot actions. The upstream repo expresses that distinction with Python functions for sub-tasks and `threading` for parallel sub-tasks.

For v2, the planning contract should preserve those three levels:

- **Phase**: a temporal layer used for RU and transition-count reporting. A phase may contain multiple sub-tasks that are intended to run in parallel.
- **Sub-task**: an ordered bundle of low-level actions assigned to one robot or one robot team.
- **Action**: one concrete robot skill call such as `GoToObject`, `OpenObject`, or `PutObject`.

Current `TaskPlan -> PlanPhase -> ActionRequest` data is good enough for the first structured-control runs, but it is too flat for verifier and scheduling work. The next schedule schema should move toward:

```text
TaskPlan
  phases[]
    subtasks[]
      assigned_robots[]
      actions[]
```

This keeps benchmark comparability because RU still counts phase transitions, while making real concurrency and resource conflicts explicit.

Verifier rules should follow this contract:

- Same robot reused across multiple actions inside one sub-task: allow. This is a normal ordered action sequence.
- State dependency inside one sub-task, such as `OpenObject(Fridge)` before `PutObject(Mug, Fridge)`: allow.
- Same robot assigned to two different sub-tasks in the same phase: reject. Those sub-tasks cannot both execute in parallel with that robot.
- State dependency between two different sub-tasks in the same phase: reject. Same-phase sub-tasks should not depend on effects from each other.
- Same object or receptacle mutated by two different same-phase sub-tasks: reject unless the schema later represents a safe ordering or lock.

When benchmarking against the paper, do not change the task suite, goal states, robot catalog, low-level skill surface, or metric formulas just to make the new scheduler look better. Improvements should be reported as deltas on the original benchmark first, then as extensions on harder or more realistic scheduling benchmarks.

### Key implementation decisions

1. **Provider-backed reasoning planner**: Claude Opus 4.7 is the default planner profile, with GPT-5.4 (the ChatGPT 5.4 family) and Kimi K2.6 already wired as comparison models. Claude uses extended thinking through Anthropic's Messages API. GPT-5.4 and Kimi K2.6 both run through the OpenAI-compatible tool-calling path, with Kimi pointed at Moonshot's base URL.

2. **Structured JSON is the plan contract**: Each robot skill is represented by a typed action schema. The model returns parseable plan data directly instead of generating Python. In the current codebase, that contract is enforced with JSON-schema-like tool definitions plus typed dataclasses and explicit validation in `json_planner.py`, not with `pydantic`.

3. **One-pass reasoning planner is the default**: Reasoning models can hold decomposition, coalition formation, and allocation in a single pass. The paper chained three LLM calls because 2023 non-reasoning models could not reliably do all three at once. Keep a staged version only if we want an ablation.

4. **Verifier as separate pass**: Before execution, a second call checks that each allocated sub-task has a robot with the required skills, that temporal dependencies are respected, and that parallel sub-tasks do not have state conflicts. The current verifier implementation uses the same resolved model profile as the planner when enabled. Keep it out of the structured-control run, and add separate verifier-profile config only when we actually start cross-model verifier experiments.

5. **Closed-loop replanning (next layer)**: Wrap the executor in a monitor that catches skill failures (e.g., `PickUp` returns False because the object is out of reach). On failure, invoke the planner again with the failure context appended. Budget N replans per task before declaring failure.

6. **Reproduce the benchmark, not the workaround**: The control condition is the paper's task suite, environment assumptions, and metrics. We do not need to preserve Pythonic prompting or exact prompt chaining unless we explicitly want a legacy ablation. Every subsequent change is still measured as a delta against the paper's reported benchmark numbers.

7. **Long context and multimodality are useful extensions, not blockers**: The current planner already has symbolic and multimodal variants, and the multimodal path can attach egocentric frames from AI2-THOR. The benchmark control should still stay explicit about whether images are part of the condition being measured.

8. **Per-robot MCP is an ablation, not the first milestone**: Exposing each robot as its own MCP server is a clean way to test decentralized coalition formation later. The first implementation should stay centralized and typed so the benchmark loop is controllable.

9. **Planner objective should favor task success over robot minimization**: The paper often framed allocation around using the minimum number of robots necessary. That is a useful control prompt, but it is not the right default objective for v2. The mainline planner should optimize for successful completion, sensible parallelism, and recovery headroom first, then use robot-count minimization as a secondary preference or explicit ablation.

10. **Scheduling should become more explicit than paper-style phases**: The paper's transition metric makes phase boundaries useful for reporting, and the current typed plan preserves that. Longer term, the planner and executor should move toward the phase/sub-task/action contract above, then toward a partial-order or dependency-aware schedule that can represent real concurrency, resource contention, and reordering under failure without forcing everything into coarse sequential phases.

11. **Verifier logic should be hybrid, not purely generative**: A verifier pass is a second planning check before execution. In this project, the best version is a hybrid: deterministic checks enforce skill coverage, argument validity, and obvious conflicts, while a second model pass handles semantic gaps that are awkward to encode by hand.
12. **Verifier model selection should stay simple until we measure cross-model effects**: For the first semantic verifier pass, reuse the planner's resolved provider/model profile so benchmark runs only change one variable at a time. Add separate verifier-profile config only when we actually start cross-model verifier experiments.

### Tech stack

- Python 3.11+
- AI2-THOR 5.0.0
- `anthropic` SDK for Claude Opus 4.7 calls
- `openai` SDK for GPT-5.4 and Kimi K2.6 through chat-completions tool calling
- stdlib dataclasses plus explicit JSON-schema validation for structured plans
- `pytest` for benchmark harness
- `structlog` with JSON output, one log per task run
- `Pillow` as an optional runtime dependency when multimodal planning images are enabled

Optional but worth considering:
- `LangGraph` if we want explicit state-machine orchestration over the verify/replan loop
- MCP server wrapper per robot as a separate ablation (tests whether per-robot agents negotiate better than a single centralized planner)

### Current file structure

```
smart-llm-v2/
  README.md
  overview.md
  pyproject.toml
  uv.lock
  SMART-LLM/            # upstream reference repo and benchmark assets
  smart_llm_v2/
    agents/
      planner.py
      plan.py
      json_planner.py
      anthropic_client.py
      openai_client.py
      provider_factory.py
      model_profiles.py
      message_builders.py
      executor.py
      verifier.py
      paper_planner.py   # legacy staged ablation path, not the default
    env/
      ai2thor_wrapper.py
      state_extractor.py
      config.py
      profiles.py
    skills/
      base.py
      mobile.py
      manipulation.py
    benchmark/
      models.py
      tasks.py
      metrics.py
      runner.py
    robots.py
  experiments/
    01_structured_control.py
  tests/
    ...
  results/
    <timestamp>/
      config.json
      summary.json
      task_runs.jsonl
```

Planned additions, not implemented yet:

```
smart-llm-v2/
  smart_llm_v2/
    agents/
      monitor.py
  experiments/
    02_reasoning_model_swap.py
    03_add_verifier.py
    04_closed_loop.py
    05_cross_model_variance.py
    06_legacy_prompt_ablation.py
```

## Milestones and next steps

1. **Done: benchmark scaffold + AI2-THOR setup**. The project now has the task loader, metric layer, AI2-THOR wrapper, baseline executor, and test coverage for the control path.

2. **Done: JSON plan contract**. The typed phased-plan boundary is implemented and exercised through the structured planner tests.

3. **In progress: structured control run**. The provider-backed control path exists for Claude Opus 4.7, GPT-5.4, and Kimi K2.6. The next concrete step is to run and archive comparable benchmark outputs across the 36 tasks instead of stopping at unit coverage and a single experiment entrypoint.

4. **Done: optional hybrid verifier pass**. The runner can run deterministic pre-execution checks first, then send surviving plans through a structured semantic verifier that reuses the planner's model profile. A dedicated verifier experiment still needs to archive verifier-gated benchmark outputs.

5. **Next: closed-loop replanning**. Add a monitor that captures execution failures and state mismatches, then re-invokes the planner with failure context and a bounded replan budget.

6. **Next: replace paper-era runtime assumptions**. Move from phase-only execution toward dependency-aware scheduling, and change the planner objective from "fewest robots" to "highest chance of success with good utilization and reasonable parallelism." Keep the old framing only as a benchmark ablation.

7. **Next: cross-model comparison**. Use the same benchmark harness to compare Claude Opus 4.7, GPT-5.4, and Kimi K2.6 on success rate and variance. Add other models only after the three wired paths have stable benchmark outputs.

8. **Next: legacy prompt ablation**. `paper_planner.py` already preserves the staged prompt shape as an ablation path, but it still needs to be wired into a full experiment so we can measure it against the JSON-first control.

9. **Stretch: harder benchmark**. Once the core benchmark loop is stable, add adversarial instructions, stochastic object placement, robot dropout, or under-specified tasks.

## Constraints and preferences

- **Hardware**: Dev on standard laptop/desktop. Stretch goal: deploy a small-model variant (Llama 3.1 8B or Gemma 2 on Raspberry Pi 5 + AI HAT) for an "edge multi-robot planner" demo.
- **Code style**: Clean, typed (mypy or pyright strict), pytest-covered, well-documented. Portfolio piece and grad-app material. Readability matters more than cleverness.
- **Paper-matching**: Use the authors' original task descriptions and ground-truth states verbatim where available. Cite them. Frame as an extension, not a replacement. Match the benchmark, not necessarily the outdated prompt schema.
- **Cost awareness**: Reasoning model calls at scale are expensive. Cache prompts and responses aggressively. Persist every API call so reruns do not require re-billing.

## Known open questions

1. The repo snapshot does expose 36 task records across seven floor plans, but several lines are malformed or omit fields such as `object_states`, `trans`, or `max_trans`. The loader already handles this defensively, but we still need to decide how strictly to treat missing metadata in published benchmark results.
2. The environment is currently pinned to AI2-THOR 5.0.0. We still need to confirm whether that is close enough to the paper's setup for benchmark claims, or whether we need a paper-matched simulator pin as a separate control.
3. Should the benchmark control stay symbolic-only, with multimodal planning treated as a later ablation, or should symbolic and multimodal runs both be first-class result tables?
4. Sim-only, or also reproduce the real-robot visibility-coverage experiments?
5. How aggressive should we be about replacing phase-only execution with dependency-aware scheduling while still reporting RU in a way that stays comparable to the paper?
6. What replan budget and caching policy keep benchmark cost under control now that verifier and recovery loops add extra model calls?
7. How much of the legacy prompt path do we want to preserve as a real ablation, versus keeping it only as a parity and parser reference?

## References

- SMART-LLM paper: https://arxiv.org/abs/2309.10062
- SMART-LLM upstream repo: https://github.com/SMARTlab-Purdue/SMART-LLM
- Project page: https://sites.google.com/view/smart-llm/
- AI2-THOR: https://ai2thor.allenai.org/
