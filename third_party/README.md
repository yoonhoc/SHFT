# third_party

Our code imports the authors' code unchanged. Clone each repository here at the commit we used.

## SWE-QA-Pro

The agent prompt, tools, forced answer (`_force_finish`), runner pieces, and SWE-QA-Pro judge come from
[TIGER-AI-Lab/SWE-QA-Pro](https://github.com/TIGER-AI-Lab/SWE-QA-Pro) (MIT). The RL questions of the Self setting are
its `train/RL/verl-tool/dataset/train.parquet`.

```bash
cd third_party
git clone https://github.com/TIGER-AI-Lab/SWE-QA-Pro.git
git -C SWE-QA-Pro checkout 93ac6a4
pip install -r SWE-QA-Pro/eval/requirements.txt
export PYTHONPATH=$PWD/SWE-QA-Pro/eval     # the package `sweqapro`
```

## SWE-QA

The questions (`Benchmark/*.jsonl`), repository list and clone script, and judge
(`Benchmark construction/score/llm-as-a-judge.py`) of the second test benchmark come from
[peng-weihan/SWE-QA-Bench](https://github.com/peng-weihan/SWE-QA-Bench).

```bash
cd third_party
git clone https://github.com/peng-weihan/SWE-QA-Bench.git
git -C SWE-QA-Bench checkout c13deac
```

## budget-aware-agent

The Budget Tracker baseline (`evaluate-agent/budget_tracker/run.py`) follows the budget prompt and `<budget>` block in
`src/prompts_budget_tracker.py` (`prompt_v011_oct`) and `src/agent_budget_tracker.py` of
[google-research/budget-aware-agent](https://github.com/google-research/budget-aware-agent) (Apache-2.0). The code
does not import it; clone it only to compare the prompt.

```bash
cd third_party
git clone https://github.com/google-research/budget-aware-agent.git
git -C budget-aware-agent checkout 37d23f6
```
