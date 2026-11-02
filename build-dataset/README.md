# build-dataset

Builds the SHFT training data. For every training trajectory it forces an answer from the agent at every prefix, judges
the answers, labels the golden step, asks a teacher model for a rationale, and assembles the halted trajectories.

```
build-dataset/
  dataset.py  runner.py  utils.py      trajectories (Expert or Self), parallel runner, logging
  golden_step/                         steps 1-3: forced answers, judging, golden step labels
    answer.py  judge.py  label.py  models.py
  rationale/                           steps 4-5: rationales and training data
    generate.py  assemble.py
  questions/                           RL questions for the Self setting
    load.py  clone.py  split.py  out/split.json
  pipeline.sh                          steps 1-5 for one model (Self or Expert)
  pipeline_two_stage.sh                steps 1-5 for Expert → Self
```

Run the modules from this folder with the SWE-QA-Pro code on `PYTHONPATH`. Steps of one dataset share a run id
(`-r`), and each step reads the output of the previous one.

```bash
cd build-dataset
export PYTHONPATH=$PWD/../third_party/SWE-QA-Pro/eval
```

## Trajectories

- **Expert**: the 1,000 Claude trajectories of `TIGER-Lab/SWE-QA-Pro-SFT-Trajectories`, read from the Hugging Face hub.
  The reference answer for judging is the final answer of each trajectory (a pseudo-reference).
- **Self**: the agent's own runs on the RL questions of SWE-QA-Pro (`--trajectories <results folder>`). The reference
  answer is the ground truth of the question. Only the 400 training questions of `questions/out/split.json` go into
  the training data; the 60 validation questions are kept for checkpoint selection.

```bash
python -m questions.clone    # clone the 464 RL repositories at their commits into questions/out/repos-rl/ (40 GB)
python -m questions.split    # only to rebuild split.json (seed 0); the included file is the one we used
```

## Servers

Steps 1 and 4 query vLLM servers. The scripts below start and stop them, and step 2 calls the OpenAI API.

```bash
# agent (step 1), as in the SWE-QA-Pro runner: no reasoning parser, 32,768-token context
vllm serve Qwen/Qwen3-8B --port 8091 --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes \
    --default-chat-template-kwargs '{"enable_thinking": false}'
# teacher (step 4)
vllm serve Qwen/Qwen3.8-27B --port 8102 --max-model-len 40960 --max-num-seqs 256
```

## 1. `golden_step/answer.py`: forced answers at every prefix

At every prefix i = 0..n, the untrained agent sees steps 0..i-1 and is forced to answer with the SWE-QA-Pro authors'
`_force_finish`, the call their runner makes at the step limit (reason `STOP AND ANSWER NOW`). We sample 3 answers per
prefix at temperature 0.6 (`NUM_ANSWERS`, `TEMPERATURE`; other sampling values are the model defaults).

```bash
python -m golden_step.answer -r <run-id> [--trajectories <results folder>] \
    --base-url http://localhost:8091/v1 --model Qwen/Qwen3-8B --max-workers 8 --inner-workers 8
```

- `--inner-workers` sends the prefixes of one trajectory together, so they share the vLLM prefix cache.
- Failures follow the authors' runner. An output without `<finish>` or with a tool call is sampled again up to 7 times;
  a degenerate output is not. A prefix longer than the context gets no answer, and the reason is kept in `error`.
- Output: `golden_step/out/<run-id>/answers/<trajectory_id>.json` (`Run`, with `steps[i].answers[k]`).
  `reasoning` keeps the text before `<finish>`. `--num-samples N` runs only the first N trajectories.

## 2. `golden_step/judge.py`: judging the forced answers

Each answer is scored with the SWE-QA-Pro judge (`_score_one`, the authors' prompt, 5 criteria from 1 to 10) against
the reference answer. The judge is GPT-6 Luna with reasoning effort medium, called through the Responses API with Flex
processing (`JUDGE_MODEL`, `REASONING_EFFORT`, `SERVICE_TIER`). Flex can return 429, so each answer is tried up to 10
times. It reads `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

```bash
python -m golden_step.judge -r <run-id> [--trajectories <results folder>]
```

- Scores go into the same files (`answers[k].score`). Run the command again to fill answers left unscored by API errors.
- A failed answer is judged as `"Error: <reason>"`, as the authors' runner does. Answers past the context are not judged.

## 3. `golden_step/label.py`: golden steps

The score of a prefix is the mean total score of its answers. The golden step i* is the prefix with the highest score,
the earliest one on ties. The answer of the halting turn is the best-scoring answer at i* (the first one on ties).
Answers judged as `"Error: ..."` count with their score, and answers without a score are left out.

```bash
python -m golden_step.label -r <run-id>     # fills halting_step and halting_answer in the same files
```

## 4. `rationale/generate.py`: rationales

For every trajectory cut before its end (i* < n), the teacher reads steps 0..i*-1 and the halting answer and writes the
rationale the agent says before answering: what it has checked and why that is enough. The system prompt
(`INSTRUCTION`) shows three closing messages of Claude from the Expert trajectories as style examples. The teacher sees
the agent's conversation with its own system prompt, runs with thinking disabled at temperature 0, and writes at most
800 tokens. If it goes on to write a tool call or the answer, the text is cut before that (logged as "trimmed").

```bash
python -m rationale.generate -r <run-id> [--trajectories <results folder>] \
    --base-url http://localhost:8102/v1 --model Qwen/Qwen3.8-27B
```

- Output: `rationale/out/<run-id>/rationales/<trajectory_id>-<step>.txt`.
- `--golden-step-dir` and `--golden-step-run-id` read labels from another folder or run id (also in step 5).

## 5. `rationale/assemble.py`: halted trajectories

```bash
python -m rationale.assemble -r <run-id> [--trajectories <results folder>] --model Qwen/Qwen3-8B               # SHFT data
python -m rationale.assemble -r <run-id> [--trajectories <results folder>] --model Qwen/Qwen3-8B --no-teacher  # Plain data
```

- One training example per labeled trajectory. If i* = n, the trajectory is kept whole. Otherwise it keeps the tool
  calls of steps 0..i*-1 and ends with the halting turn `<rationale>\n\n<finish>\n<answer>\n</finish>`.
- `--no-teacher` (Plain) uses the text the forced answer itself wrote before `<finish>` instead of the rationale.
- Examples longer than `--max-length` (32,768 tokens with the tokenizer of `--model`) are dropped, as are trajectories
  without a golden step, answer, or rationale.
- Output: `rationale/out/<run-id>/datasets/{teacher,no-teacher}/train.jsonl` (`messages`, `tools`), which `../train` reads.

## All steps at once

`pipeline.sh <models.yaml key> <model id> [vllm serve args for step 1]` runs the steps on one GPU and starts and stops
the servers. By default it builds the Self data: it first runs the agent on the RL questions
(`../evaluate-agent/out/<key>-own/sweqapro-rl/results/`) and uses the run id `<key>-own`. With `SFT=1` it builds the
Expert data under the run id `<key>`. Other variables: `PORT` (default 8091), `TEACHER` (default `Qwen/Qwen3.8-27B`),
and `TOKENIZER` (for the length check of step 5, default the model id).

```bash
# Qwen3-8B, Self and Expert
bash pipeline.sh qwen3-8b-nothink Qwen/Qwen3-8B --tool-call-parser hermes \
    --default-chat-template-kwargs '{"enable_thinking": false}'
SFT=1 bash pipeline.sh qwen3-8b-nothink Qwen/Qwen3-8B --tool-call-parser hermes \
    --default-chat-template-kwargs '{"enable_thinking": false}'
# Ministral-3-8B (the tokenizer folder comes from ../train/mistral_tokenizer.py)
TOK=../train/out/ministral-8b-tokenizer
TOKENIZER=$TOK bash pipeline.sh ministral-8b mistralai/Ministral-3-8B-Instruct-2512-BF16 \
    --tool-call-parser mistral --limit-mm-per-prompt '{"image":0}' --tokenizer-mode hf --tokenizer $TOK
```

`pipeline_two_stage.sh <qwen3|ministral>` builds the Expert → Self data. The Expert SHFT checkpoint runs on the RL
questions and also gives the forced answers, so the golden step is where that model has enough information. Set
`ADAPTER` in the script to your selected Expert SHFT checkpoint. The run id is `<key>-expert-sis-online`. `GPU_MEM` sets the vLLM memory fraction (default 0.90), and
`UNTIL=label` stops after step 3. Training then continues from the same checkpoint (`../train/README.md`).

## Data flow

```
golden_step/answer.py   →  golden_step/out/<run-id>/answers/<trajectory_id>.json      Run: 3 forced answers per prefix
golden_step/judge.py    →  same files, answers[k].score
golden_step/label.py    →  same files, halting_step and halting_answer
rationale/generate.py   →  rationale/out/<run-id>/rationales/<trajectory_id>-<step>.txt
rationale/assemble.py   →  rationale/out/<run-id>/datasets/{teacher,no-teacher}/train.jsonl
```
