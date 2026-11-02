"""정지 구간에 무게 λ 를 둔 교차 엔트로피 (HaltingTrainer). SHFT/train_qwen.py 와 SHFT/train_mistral.py 가 쓴다.

학습 행의 마지막 assistant 턴 = "정지 서술 + <finish> 답 </finish>". 토큰별 무게:
    마지막 assistant 턴의 시작 ~ <finish> 앞   λ (--stop-weight) — 도구 호출 대신 글을 쓰기로 하는 결정과 정지 서술
    그 밖의 assistant 토큰                    1 — 탐색 턴의 도구 호출, <finish> 답 </finish>
    system · user · 도구 결과                 0 (손실 없음)
행마다 무게의 합이 assistant 토큰 수와 같게 맞추고, 누적 배치의 assistant 토큰 수로 나눈다. 그래서 λ=1 이면 보통 SFT 손실
(TRL chunked_nll) 과 같고, λ 는 행 안에서 정지 구간과 나머지 사이의 비율만 바꾼다.

출력층은 손실이 걸리는 위치에서만 계산한다 (행 전체 로짓은 3만 토큰 × 어휘 15만이면 fp32 로 십수 GB).
GPU 당 행 1개 전제 (per_device_train_batch_size=1).
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from trl import SFTTrainer

IGNORE_INDEX = -100
FINISH_TAG = "<finish>"


def find_last_turn(labels: torch.Tensor) -> tuple[int, int]:
    """마지막 assistant 턴 [시작, 끝) = labels 가 IGNORE_INDEX 가 아닌 마지막 연속 구간."""
    end = int(torch.nonzero(labels != IGNORE_INDEX).flatten()[-1]) + 1
    start = end - 1
    while start > 0 and labels[start - 1] != IGNORE_INDEX:
        start -= 1
    return start, end


def find_finish_tag(ids: list[int], start: int, end: int, tokenizer: PreTrainedTokenizerBase) -> int:
    """ids[start:end] 에서 처음 나오는 FINISH_TAG 의 '<' 가 든 토큰 위치.

    태그는 토큰 경계와 맞지 않을 수 있어 ('.<' 'finish' '>\\n' 처럼 앞뒤 글자와 붙는다) 토큰 id 가 아니라 글자로 찾는다.
    """
    tag_end = None
    for idx in range(start, end):
        window_start = max(start, idx - len(FINISH_TAG))
        if FINISH_TAG in tokenizer.decode(ids[window_start : idx + 1]):
            tag_end = idx
            break
    if tag_end is None:
        raise ValueError("No <finish> in the last assistant turn")

    # 태그가 온전히 들어가는 가장 짧은 구간의 시작이 '<' 가 든 토큰이다
    tag_start = tag_end
    while FINISH_TAG not in tokenizer.decode(ids[tag_start : tag_end + 1]):
        tag_start -= 1
    return tag_start


def compute_token_weights(labels: torch.Tensor, tokenizer: PreTrainedTokenizerBase, stop_weight: float) -> torch.Tensor:
    """assistant 토큰 1, 그중 마지막 턴의 시작 ~ <finish> 앞만 stop_weight, 손실이 없는 토큰 0. 합은 assistant 토큰 수로 맞춘다."""
    weights = (labels != IGNORE_INDEX).float()
    start, end = find_last_turn(labels)
    stop_end = find_finish_tag(labels.tolist(), start, end, tokenizer)
    weights[start:stop_end] = stop_weight
    return weights * ((labels != IGNORE_INDEX).sum() / weights.sum())


class HaltingTrainer(SFTTrainer):
    """정지 구간에 무게 λ 를 둔 교차 엔트로피 (모듈 docstring). 정규화는 SFT 기준선의 손실과 같다."""

    def __init__(self, stop_weight: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stop_weight = stop_weight

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, object]:
        labels = inputs.pop("labels")
        if labels.shape[0] != 1:
            raise ValueError("HaltingTrainer expects one row per device")
        # 다음 토큰 예측이라 한 칸 밀린다
        labels = labels[0, 1:]
        inputs["use_cache"] = False

        valid = labels != IGNORE_INDEX
        weights = compute_token_weights(labels, self.processing_class, self.stop_weight)

        # 모델 몸통만 돌리고, 출력층은 손실이 걸리는 위치에서만 모델과 같은 계산(bf16)으로 한 뒤 fp32 로 올린다
        causal_lm = self.accelerator.unwrap_model(model).get_base_model()
        hidden = causal_lm.model(**inputs).last_hidden_state[0, :-1][valid]
        logits = causal_lm.get_output_embeddings()(hidden)
        token_loss = F.cross_entropy(logits.float(), labels[valid], reduction="none")
        weighted_sum = (token_loss * weights[valid]).sum()

        # SFT 기준선과 같게 누적 배치의 토큰 수로 나눈다. 모든 GPU 의 합계라 GPU 수를 곱한다 (DDP 가 GPU 평균을 내므로)
        if num_items_in_batch is None:
            raise ValueError("HaltingTrainer expects num_items_in_batch from the Trainer")
        loss = weighted_sum / num_items_in_batch * self.accelerator.num_processes
        if return_outputs:
            return loss, logits
        return loss
