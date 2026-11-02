# SHFT: Sufficiency-Halted Fine-Tuning

Code for **Knowing When You Know Enough: Learning Information Sufficiency for Repository-Level QA Agents**.

Tool-using agents for repository-level question answering often keep calling tools after they already have enough
information to answer. SHFT trains the agent to stop. For each training trajectory it finds the *golden step*, the
earliest prefix at which answers forced from the agent score highest. A teacher model writes a short rationale for
stopping there, the trajectory is cut at that step (a *halted trajectory*), and the agent is fine-tuned with a higher
loss weight on the halting turn.

## Layout

| Folder | Contents |
|---|---|
| [`build-dataset/`](build-dataset/README.md) | Golden step labels, rationales, and SHFT training data |
| [`train/`](train/README.md) | LoRA training of the SFT baseline, SHFT, and Plain |
| [`evaluate-agent/`](evaluate-agent/README.md) | Agent runs on SWE-QA-Pro and SWE-QA, the Budget Tracker and AutoRater baselines, judges, and scores |
| [`third_party/`](third_party/README.md) | Author repositories to clone (SWE-QA-Pro, SWE-QA, budget-aware-agent) |

Run every module from inside its folder with `python -m <module>`. Outputs go to `out/` next to the script
(`<folder>/out/<run-id>/...`). Every step skips items whose output file already exists, so an interrupted step resumes
when you run the same command again.

## Setup

1. Clone the author repositories at the commits we used, and the repositories of both benchmarks
   ([third_party/README.md](third_party/README.md)).
2. Install the pinned versions in `requirements.txt`. We used two Python environments.
   - Python 3.11 for agent runs, dataset steps, and judges, with `third_party/SWE-QA-Pro/eval` on `PYTHONPATH`.
     Activate it before running the shell scripts, which call `python` (or `$PY`, which defaults to `python`).
   - Python 3.10 for vLLM 0.20.2 and training (transformers 5.8.1, TRL 1.12.0, PEFT 0.20.0, DeepSpeed 0.19.6).
     The scripts start servers with `vllm serve`, so `vllm` must be on `PATH`.
3. Install bubblewrap (`bwrap`). Every tool call of the agent runs in a sandbox that sees only the repository of the
   question ([evaluate-agent/README.md](evaluate-agent/README.md)).
4. For Ministral, write the fixed tokenizer once: `cd train && python -m mistral_tokenizer` ([train/README.md](train/README.md)).
5. Copy `.env.example` to `.env`, fill in the keys, and load it with `set -a && source .env && set +a`.
   The judges, golden step labels, and AutoRater call hosted models (GPT-6 Sol, GPT-6 Luna, Claude Sonnet 5.5).

We ran each vLLM server (the 8B agents with a 32,768-token context, the Qwen3.8-27B teacher) on one 96 GB GPU, and
trained on 1 to 8 GPUs with an effective batch of 16.

## Models and data

- Agents: Qwen3-8B with thinking disabled (`evaluate-agent/models.yaml` key `qwen3-8b-nothink`) and
  Ministral-3-8B-Instruct-2512 in BF16 (`ministral-8b`). Both use the SWE-QA-Pro scaffold with a step limit of 30.
- Expert trajectories: the 1,000 Claude trajectories of
  [TIGER-Lab/SWE-QA-Pro-SFT-Trajectories](https://huggingface.co/datasets/TIGER-Lab/SWE-QA-Pro-SFT-Trajectories).
- Self rollouts: the agent's own runs on the RL questions of SWE-QA-Pro. Of the 464 questions, 400 are for training and
  60 for validation (`build-dataset/questions/out/split.json`). Two repositories no longer exist, and two questions
  share a repository with the test benchmarks.
- Test benchmarks: SWE-QA-Pro (260 questions, judged by GPT-6 Sol) and SWE-QA (720 questions, judged by Claude Sonnet 5.5).
  Checkpoints are chosen on the 60 validation questions, judged by GPT-6 Luna.

## Reproducing the main results

Every row of the main results table (`tab:main-results`) comes from the same four steps, done once per agent.

| Step | Self | Expert | Expert → Self |
|---|---|---|---|
| 1. Training data (`build-dataset`) | `pipeline.sh` on the agent's rollouts | `SFT=1 pipeline.sh` on the Expert trajectories | `pipeline_two_stage.sh` on rollouts of the Expert SHFT agent |
| 2. Training (`train`) | Plain and SHFT from the base model | SFT, Plain, and SHFT from the base model | Plain and SHFT from the Expert SHFT checkpoint |
| 3. Checkpoint selection (`evaluate-agent/valid.sh`) | compared with Base | compared with SFT | compared with SFT |
| 4. Test runs and judges (`evaluate-agent`) | Base, Budget Tracker, AutoRater, Plain, SHFT | SFT, SFT + Budget Tracker, SFT + AutoRater, Plain, SHFT | SHFT → Plain, SHFT → SHFT |

- Plain trains on halted trajectories without the rationale (`datasets/no-teacher`) with the plain loss, and SHFT
  trains on halted trajectories with the rationale (`datasets/teacher`) with halting weight 5.
- We save one checkpoint per epoch. SFT uses the checkpoint with the highest validation score. Plain and SHFT use the
  checkpoint with the highest validation score among those with fewer tool calls than the agent they are compared with,
  or the one with the fewest tool calls if none has fewer.
- Budget Tracker and AutoRater need no training and run on top of Base or the SFT checkpoint.

Commands for SHFT in the Self setting with Qwen3-8B:

```bash
set -a && source .env && set +a
export PYTHONPATH=$PWD/third_party/SWE-QA-Pro/eval

# 1. Training data: rollouts on the RL questions, forced answers, golden steps, rationales, halted trajectories
cd build-dataset
python -m questions.clone
CUDA_VISIBLE_DEVICES=0 bash pipeline.sh qwen3-8b-nothink Qwen/Qwen3-8B \
    --tool-call-parser hermes --default-chat-template-kwargs '{"enable_thinking": false}'

# 2. SHFT with halting weight 5 (one GPU, so gradient accumulation 16)
cd ../train
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file deepspeed_zero2.yaml --num_processes 1 \
    -m SHFT.train_qwen -r qwen3-self-shft \
    --dataset-dir ../build-dataset/rationale/out/qwen3-8b-nothink-own/datasets/teacher \
    --learning-rate 5e-4 --stop-weight 5 --gradient-accumulation-steps 16

# 3. Validation runs of Base and of the four checkpoints (one per epoch of 25 steps), then their scores and tool calls
cd ../evaluate-agent
CUDA_VISIBLE_DEVICES=0 python -m run_agent -r qwen3-8b-nothink -b valid -m qwen3-8b-nothink
python -m score.judge -r qwen3-8b-nothink -b valid
CUDA_VISIBLE_DEVICES=0 bash valid.sh qwen3-self-shft qwen3-8b-nothink 25 50 75 100
python -m score.summary -b valid -r qwen3-8b-nothink qwen3-self-shft-e1 qwen3-self-shft-e2 qwen3-self-shft-e3 qwen3-self-shft-e4

# 4. Test runs with the chosen checkpoint, then the benchmark judges
for b in sweqapro sweqa; do
  CUDA_VISIBLE_DEVICES=0 python -m run_agent -r qwen3-self-shft-e1 -b $b -m qwen3-8b-nothink \
      --adapter ../train/SHFT/out/qwen3-self-shft/checkpoint-25
done
python -m score.gpt_judge.sweqapro -r qwen3-self-shft-e1
python -m score.claude_judge.sweqa submit -r qwen3-self-shft-e1   # run `collect` once the batch is done
```

`score/main_results.py` prints the score and tool calls of every row of the table. The run id of each row is listed in
its `ROWS`, so set them to the names of your runs. The folder READMEs explain each step, the other settings, the
baselines, and the options.

Some names in the code predate the paper: `own` is the Self setting, `sis` is SHFT, `bats` is the Budget Tracker,
`sufficiency` is the AutoRater, and in `main_results.py` `On-Policy`, `Off-Policy`, and `Off + On` are Self, Expert, and
Expert → Self.

## Reproducibility notes

- The judges are hosted models. Judging the same answers again later can give slightly different scores, and the golden
  step labels also depend on GPT-6 Luna. We called every hosted model through one LiteLLM proxy with the request
  settings in the code (reasoning effort medium and Flex processing for the GPT models, Message Batches for Claude).
- Agents decode at temperature 0 with vLLM, which is still not bit-exact across GPUs and batch sizes.

## License

MIT ([LICENSE](LICENSE)). The author repositories in `third_party/` keep their own licenses
([third_party/README.md](third_party/README.md)). The Budget Tracker prompt follows
[google-research/budget-aware-agent](https://github.com/google-research/budget-aware-agent) (Apache-2.0).
