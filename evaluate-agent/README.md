# evaluate-agent

Runs the agents on SWE-QA-Pro (260 questions), SWE-QA (720 questions), and the 60 validation questions, runs the two
inference-time baselines, judges the answers, and prints scores and tool calls.

```
evaluate-agent/
  run_agent.py  runner.py  models.py  dataset.py  utils.py   agent runs, results, questions, logging
  models.yaml                          agent settings (format of the SWE-QA-Pro configs/models.yaml)
  tool_sandbox.py  tool_exec.py        every tool call in a bubblewrap sandbox
  budget_tracker/run.py                Budget Tracker baseline
  autorater/run.py  example.py  example.txt   AutoRater baseline and its one-shot example
  score/
    judge.py                           GPT-6 Luna judge (validation), shared judge settings
    gpt_judge/sweqapro.py              SWE-QA-Pro judge: GPT-6 Sol
    claude_judge/batch.py  sweqa.py    SWE-QA judge: Claude Sonnet 5.5 with Message Batches
    summary.py  main_results.py        scores and tool calls
  valid.sh                             validation runs of the checkpoints of one training run
```

Run the modules from this folder with the SWE-QA-Pro code on `PYTHONPATH`:

```bash
cd evaluate-agent
export PYTHONPATH=$PWD/../third_party/SWE-QA-Pro/eval
```

## Benchmark repositories

Clone the repositories of each benchmark at the commits fixed by its authors. Repositories that appear in both
benchmarks have different commits, so each benchmark keeps its own copy.

```bash
(cd ../third_party/SWE-QA-Pro/eval && bash clone_repos.sh)   # SWE-QA-Pro, into third_party/SWE-QA-Pro/eval/repos/
(cd ../third_party/SWE-QA-Bench && bash clone_repos.sh)      # SWE-QA, into third_party/SWE-QA-Bench/datas/repos/
```

The validation questions use the RL repositories cloned by `../build-dataset/questions/clone.py`.

## Agents

A key of `models.yaml` is one base model: its vLLM arguments, tool-call parser, temperature 0, and a step limit of 30.
Other values are the SWE-QA-Pro defaults (32,768-token context, a history window of 25 turns, and a forced answer once
the context reaches 82.5%). A trained agent is the same key with `--adapter <checkpoint>`, which vLLM serves as a LoRA
adapter on the base model.

## 1. `run_agent.py`: agent runs

```bash
CUDA_VISIBLE_DEVICES=0 python -m run_agent -r <run-id> -b sweqapro -m qwen3-8b-nothink                         # base model
CUDA_VISIBLE_DEVICES=0 python -m run_agent -r <run-id> -b sweqa -m qwen3-8b-nothink --adapter <checkpoint>     # trained
```

- `-b` is `sweqapro`, `sweqa`, `valid` (the 60 validation questions), or `sweqapro-rl` (all RL questions, for Self rollouts).
- The script starts a vLLM server on the GPUs of `CUDA_VISIBLE_DEVICES` and stops it at the end (`vllm` must be on `PATH`).
- The agent is the SWE-QA-Pro `ToolCallingAgent` with its prompt, three tools, step limit, and forced answer at the
  limit. Compared with the authors' runner, only the model config, the output format, the number of parallel questions
  (`--max-workers`, default 16), and the tool sandbox differ.
- **Tool sandbox.** The authors' tools walk any path the model gives and run shell commands, so a path such as `/` would
  reach the whole disk, other repositories, and the benchmark answers. Every tool call therefore runs in a bubblewrap
  (`bwrap`) root that contains only system binaries, Python, the tool package, and the repository of the question. The
  tool code is unchanged, and the agent sees the same relative paths (`repos/<name>`) as with the authors' runner.
- Output: `out/<run-id>/<benchmark>/results/<index>.json` (`Result`: the answer, status, and stop reason as in the
  authors' results, plus the full trajectory with the raw response and token usage of every call). Every request and
  response is also logged to `requests/<index>.jsonl` (`--no-record` turns this off).
- A question that fails 7 times in the authors' agent is saved with `answer = "Error: <reason>"`. Errors outside the
  agent, such as a lost server connection, write no file, so the next run tries again.
- `--num-samples N` runs the first N questions, `--max-iterations` changes the step limit, and `--repo-root` points to
  repositories cloned elsewhere.

## 2. Baselines

Both baselines reuse the server, output format, and arguments of `run_agent.py` and write to their own `out/`.

```bash
# Budget Tracker: tells the agent its used and remaining tool budget after every turn that does not answer
CUDA_VISIBLE_DEVICES=0 python -m budget_tracker.run -r <run-id> -b sweqapro -m qwen3-8b-nothink [--adapter <checkpoint>]
# AutoRater: after every tool turn, GPT-6 Luna judges whether the context is sufficient; the first sufficient step ends with a forced answer
CUDA_VISIBLE_DEVICES=0 python -m autorater.run -r <run-id> -b sweqapro -m qwen3-8b-nothink --rater-openai [--adapter <checkpoint>]
```

- Budget Tracker follows the prompt and `<budget>` block of the BATS authors (`third_party/budget-aware-agent`,
  `prompt_v011_oct`), with one budget for all tools and the search instructions adapted to our tools. The budget is the
  number of turns that can call a tool (step limit minus 1), and each executed tool call uses 1.
- AutoRater uses the one-shot autorater prompt of Sufficient Context (Joren et al., ICLR 2025). The example is one of
  our training rollouts (`autorater/example.txt`). The context is the conversation so far, flattened as in
  `_force_finish`. At the first step rated sufficient, the agent gives the same forced answer as in
  `../build-dataset/golden_step/answer.py`. If that answer fails, the agent keeps exploring. The ratings go to
  `autorater/out/<run-id>/<benchmark>/ratings/`. `--rater-base-url` uses a local vLLM rater instead.
- Results: `budget_tracker/out/` and `autorater/out/`, in the same layout as `out/`. Give the judges `-o` with that folder.

## 3. Judges

| Benchmark | Judge | Module | Scores in |
|---|---|---|---|
| SWE-QA-Pro | GPT-6 Sol, reasoning effort medium, Flex | `score.gpt_judge.sweqapro` | `results-gpt-6-sol/` |
| SWE-QA | Claude Sonnet 5.5, Message Batches | `score.claude_judge.sweqa` | `results-claude-sonnet-5-5/` |
| validation | GPT-6 Luna, reasoning effort medium, Flex | `score.judge` | `results/` |

```bash
python -m score.gpt_judge.sweqapro -r <run-id>                     # add -o budget_tracker/out for a baseline run
python -m score.claude_judge.sweqa submit -r <run-id>              # one batch with every unscored question
python -m score.claude_judge.sweqa collect -r <run-id>             # writes the scores once the batch has ended
python -m score.judge -r <run-id> -b valid
```

- Each judge calls the authors' judge of its benchmark with their prompt: SWE-QA-Pro scores 5 criteria from 1 to 10
  (5 to 50), and SWE-QA 5 criteria from 1 to 20 (5 to 100). Only the judge model changes.
- Empty answers and `"No answer found"` are not judged, as in the authors' judges. Failed questions are judged with
  their `"Error: <reason>"` answer.
- The GPT judges retry each question up to 10 times (Flex can return 429). Running a judge again fills only the questions
  without a score. A Claude batch that has not been collected blocks a new `submit`.
- Keys: `score.gpt_judge` reads `LITELLM_BASE_URL` and `LITELLM_API_KEY`, `score.claude_judge` reads
  `LITELLM_ANTHROPIC_BASE_URL` and `LITELLM_API_KEY`, and `score.judge` and the AutoRater read `OPENAI_BASE_URL` and
  `OPENAI_API_KEY` (see `../.env.example`).

## 4. Scores and tool calls

```bash
python -m score.summary -b valid -r <run-id> <run-id> ...          # validation scores (GPT-6 Luna)
python -m score.main_results                                        # rows of the main results table
```

- Score: the mean total score of the judged questions, with `(judged/total)` while judging is unfinished.
- Calls: the mean number of tool calls the agent executed per question. A question that failed in the end counts as
  the step limit minus 1, since the last turn is the forced answer.
- `score.summary` reads the GPT-6 Luna scores in `results/`. `score.main_results` reads the scores of the benchmark
  judges and lists the run id of each table row in `ROWS`.

## Validation runs

`valid.sh <run-name> <models.yaml key> <checkpoint steps...>` runs and judges each checkpoint of one training run on
the validation questions, under the run id `<run-name>-e<epoch>`. It reads the checkpoints from `../train/SHFT/out/`;
set `TRAIN_DIR=../train/plain/out` or `../train/sft/out` for the other trainers.

```bash
CUDA_VISIBLE_DEVICES=0 bash valid.sh qwen3-self-shft qwen3-8b-nothink 25 50 75 100
```

## Data flow

```
run_agent.py, budget_tracker/run.py, autorater/run.py  →  <out>/<run-id>/<benchmark>/results/<index>.json
score/judge.py                  →  same files, score (validation)
score/gpt_judge/sweqapro.py     →  <out>/<run-id>/sweqapro/results-gpt-6-sol/<index>.json
score/claude_judge/sweqa.py     →  <out>/<run-id>/sweqa/results-claude-sonnet-5-5/<index>.json
score/summary.py, main_results.py  →  scores and tool calls on the screen
```
