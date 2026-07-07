"""Prompt-conditioned length-penalty math env (verifiers / prime-rl).

reward = correctness - lam * (completion_tokens / scale)
lam is per-sample and STATED IN LANGUAGE in the prompt: strict (default,
including unmarked prompts) pushes terse dense reasoning; explicit
"think harder / extremely important" phrasing relaxes it. The mixture teaches
the phrasing -> token-budget mapping.
"""

import random

import verifiers as vf
from datasets import load_dataset
from verifiers.rubrics.math_rubric import MathRubric
from verifiers.utils.data_utils import extract_boxed_answer

SYSTEM_PROMPT = (
    "You are being evaluated on both correctness and response length: this "
    "environment penalizes every generated token, including reasoning. The "
    "user's instructions state how strict the length penalty is for their "
    "request; calibrate how deeply you reason accordingly."
)
BUDGETS = {
    "strict": dict(lam=0.20, phrases=[
        "Strict length penalty: answer as efficiently as possible; every token counts.",
        "Be terse - shortest correct solution wins.",
        ""]),  # unmarked prompts are also strict (the default behavior)
    "lax": dict(lam=0.02, phrases=[
        "Length penalty is relaxed for this one - it is extremely important, so think as hard and as long as you need.",
        "Take all the space you need; thoroughness matters far more than brevity here."]),
}
INSTR = "Solve the following math problem. Put the final answer in \\boxed{}."


def load_environment(dataset_name="PrimeIntellect/Hendrycks-Math", dataset_subset="default",
                     dataset_split="train", p_lax=0.3, scale=1000.0, seed=0, **kwargs):
    rng = random.Random(seed)

    def build_dataset():
        def to_row(x):
            tag = "lax" if rng.random() < p_lax else "strict"
            b = BUDGETS[tag]
            phrase = rng.choice(b["phrases"])
            q = (phrase + "\n" if phrase else "") + INSTR + "\n\n" + x["question"]
            return {"question": q, "answer": x["answer"],
                    "info": {"budget": tag, "lam": b["lam"]}}
        ds = load_dataset(dataset_name, dataset_subset, split=dataset_split)
        return ds.map(to_row).select_columns(["question", "answer", "info"])

    rubric = MathRubric(parser=vf.MaybeThinkParser(extract_boxed_answer))

    def n_tokens(state) -> int:
        usage = state.get("usage")
        assert usage is not None, "token usage missing from state"  # fail hard
        return int(usage["output_tokens"])

    def length_penalty(state, info, **kwargs) -> float:
        return -float(info["lam"]) * n_tokens(state) / scale

    def completion_tokens(state, **kwargs) -> float:
        return float(n_tokens(state))

    def budget_is_lax(info, **kwargs) -> float:
        return 1.0 if info["budget"] == "lax" else 0.0

    rubric.add_reward_func(length_penalty, weight=1.0)
    rubric.add_metric(completion_tokens)
    rubric.add_metric(budget_is_lax)
    return vf.SingleTurnEnv(dataset=build_dataset, rubric=rubric, system_prompt=SYSTEM_PROMPT)
