# SMART-LLM v2

This workspace is the clean rebuild of SMART-LLM described in [overview.md](overview.md). The original SMART-LLM release stays in `SMART-LLM/` as a reference implementation and source dataset, while new code lives in `smart_llm_v2/`.

The first slice in this rebuild is the benchmark layer. It gives the new codebase a typed loader for the paper's task files and a metric module that mirrors the paper's reported metrics without depending on the legacy executor script.

## Current layout

- `SMART-LLM/`: original paper code and benchmark assets, kept intact as the control reference
- `smart_llm_v2/`: new package for the rebuild
- `tests/`: package-level tests for the v2 scaffold

## Benchmark snapshot note

The repository snapshot in `SMART-LLM/data/final_test/` does expose 36 task records across seven floor plans, which matches the paper's benchmark count. Several records are malformed JSON or omit fields such as `object_states`, `trans`, or `max_trans`, so the v2 loader parses the files defensively instead of silently rewriting the source data.
