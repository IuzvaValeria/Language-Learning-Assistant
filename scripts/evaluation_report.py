from datetime import datetime
import json

from evaluation_config import (
    BASE_MODEL,
    LORA_PATH,
    RESULTS_DIR,
    TEST_SETS,
    DISPLAY_NAMES,
)

def format_accuracy(result: dict) -> str:
    passed = result.get("passed", 0)
    total = result.get("total", 0)
    accuracy = result.get("accuracy", 0)

    return f"{passed}/{total}, {accuracy:.0%}"

def format_change(base_accuracy: float, lora_accuracy: float) -> str:
    diff = (lora_accuracy - base_accuracy) * 100

    if diff >= 0:
        return f"+{diff:.0f} pp"

    return f"{diff:.0f} pp"

def write_detailed_answers(file, title: str, results: dict) -> None:
    file.write(f"### {title}\n\n")

    for set_name in TEST_SETS:
        task_name = DISPLAY_NAMES.get(set_name, set_name)
        file.write(f"#### {task_name}\n\n")

        items = results.get(set_name, {}).get("items", [])

        for index, item in enumerate(items, start=1):
            file.write(f"**Question {index}:** {item['question']}\n\n")
            file.write(f"**Expected keyword:** `{item['expected']}`\n\n")
            file.write(f"**Correct:** `{item['correct']}`\n\n")
            file.write("**Model answer:**\n\n")
            file.write(f"{item['answer']}\n\n")
            file.write("---\n\n")

def save_report(base_results: dict, lora_results: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    jsonl_path = RESULTS_DIR / f"evaluation_{timestamp}.jsonl"
    md_path = RESULTS_DIR / f"evaluation_{timestamp}.md"

    data = {
        "timestamp": datetime.now().isoformat(),
        "base_model": BASE_MODEL,
        "lora_path": str(LORA_PATH),
        "base_results": base_results,
        "lora_results": lora_results,
    }

    with jsonl_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")

    with md_path.open("w", encoding="utf-8") as file:
        file.write("# LLM Evaluation Report\n\n")

        file.write("## 1. Goal\n\n")
        file.write(
            "The goal of this evaluation is to compare the base model with "
            "the LoRA-adapted model on tasks used in our Japanese-English "
            "learning assistant.\n\n"
        )

        file.write("## 2. Evaluation Method\n\n")
        file.write(
            "We use a small fixed test set. Each model answer is checked with "
            "simple keyword matching.\n\n"
        )
        file.write(
            "This is a rule-based evaluation, not a perfect semantic evaluation. "
            "It is suitable for this educational project because LLM answers may "
            "be phrased differently.\n\n"
        )

        file.write("## 3. Model Information\n\n")
        file.write(f"- Base model: `{BASE_MODEL}`\n")
        file.write(f"- LoRA path: `{LORA_PATH}`\n")
        file.write(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        file.write("## 4. Tested Tasks\n\n")
        for set_name in TEST_SETS:
            task_name = DISPLAY_NAMES.get(set_name, set_name)
            file.write(f"- {task_name}\n")
        file.write("\n")

        file.write("## 5. Results\n\n")
        file.write("| Task | Base model | LoRA model | Change |\n")
        file.write("|---|---:|---:|---:|\n")

        for set_name in TEST_SETS:
            task_name = DISPLAY_NAMES.get(set_name, set_name)

            base_item = base_results.get(set_name, {})
            lora_item = lora_results.get(set_name, {})

            base_acc = base_item.get("accuracy", 0)
            lora_acc = lora_item.get("accuracy", 0)

            base_text = format_accuracy(base_item)
            lora_text = format_accuracy(lora_item)
            change_text = format_change(base_acc, lora_acc)

            file.write(
                f"| {task_name} | "
                f"{base_text} | "
                f"{lora_text} | "
                f"{change_text} |\n"
            )

        base_overall = base_results.get("overall", {})
        lora_overall = lora_results.get("overall", {})

        base_overall_acc = base_overall.get("accuracy", 0)
        lora_overall_acc = lora_overall.get("accuracy", 0)

        file.write(
            f"| **Total** | "
            f"**{format_accuracy(base_overall)}** | "
            f"**{format_accuracy(lora_overall)}** | "
            f"**{format_change(base_overall_acc, lora_overall_acc)}** |\n\n"
        )

        file.write("## 6. Interpretation\n\n")
        file.write(
            "The LoRA model is expected to perform better because it was trained "
            "on task-specific examples from our project dataset.\n\n"
        )
        file.write(
            "The largest improvement is expected in translation, because translation "
            "examples are the largest part of the training dataset.\n\n"
        )

        file.write("## 7. Limitations\n\n")
        file.write(
            "This evaluation uses keyword matching. A response may be counted as "
            "incorrect even if it is partially correct, or counted as correct if it "
            "contains the keyword but the full explanation is weak.\n\n"
        )
        file.write(
            "For the final project evaluation, we also manually inspect several "
            "model answers.\n\n"
        )

        file.write("## 8. Detailed Answers\n\n")

        write_detailed_answers(
            file=file,
            title="Base Model Answers",
            results=base_results,
        )

        if lora_results:
            write_detailed_answers(
                file=file,
                title="LoRA Model Answers",
                results=lora_results,
            )

    print(f"\nJSONL report saved to: {jsonl_path}")
    print(f"Markdown report saved to: {md_path}")

    return md_path