import os
import json
from pathlib import Path
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_curve, auc
from tqdm import tqdm

from lettucedetect.datasets.hallucination_dataset import HallucinationData, HallucinationSample


def get_api_key():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise EnvironmentError("OPENROUTER_API_KEY must be set.")
    return key


def hf_to_hallucination_sample(example):
    return HallucinationSample(
        prompt=example["prompt"],
        answer=example["answer"],
        labels=[{}] if example["labels"] else [],  # labels varsa halüsinasyon
        split="test",
        task_type=example.get("task_type", "unknown"),
        dataset="hf_turkish",
        language="tr"
    )


def split_prompt_fields(prompt, task_type):
    if task_type == "Summary":
        user_input = prompt.split(":")[0]
        contexts = prompt.split(":")[1:] or [prompt]
    else:
        parts = prompt.split(":")
        user_input = " ".join(parts[:2]) if len(parts) >= 2 else task_type
        contexts = parts[2:] if len(parts) > 2 else [prompt]
    return user_input.strip(), contexts


def evaluate_faithfulness(sample, llm):
    user_input, retrieved_contexts = split_prompt_fields(sample.prompt, sample.task_type)
    s = SingleTurnSample(user_input=user_input, response=sample.answer, retrieved_contexts=retrieved_contexts)
    metric = Faithfulness(llm=llm)
    try:
        return metric.single_turn_score(s)
    except Exception as e:
        print(f"⚠️ Error scoring: {e}")
        return 0.0


def run_pipeline(threshold=0.5, max_samples=100):
    print("📥 Loading dataset from Hugging Face...")
    raw_dataset = load_dataset("newmindai/hallucination_detection_turkish")["train"]
    test_examples = [ex for ex in raw_dataset if ex.get("split") == "test"][:max_samples]
    samples = [hf_to_hallucination_sample(ex) for ex in test_examples]

    print("🔑 Initializing LLM for RAGAS...")
    llm = LangchainLLMWrapper(ChatOpenAI(
        model="qwen/qwen3-14b:free",
        openai_api_key=get_api_key(),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.0
    ))

    pred_samples = []
    for sample in tqdm(samples, desc="Scoring with RAGAS"):
        score = evaluate_faithfulness(sample, llm)
        label = {
            "faithfulness": score,
            f"threshold_{threshold}": int(score < threshold)
        }
        pred = HallucinationSample(
            prompt=sample.prompt,
            answer=sample.answer,
            labels=[label],
            split="test",
            task_type=sample.task_type,
            dataset=sample.dataset,
            language=sample.language
        )
        pred_samples.append(pred)

    # 📊 Evaluation
    y_true = [1 if s.labels else 0 for s in samples]
    y_pred = [1 if s.labels[0][f"threshold_{threshold}"] else 0 for s in pred_samples]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Supported", "Hallucinated"], digits=4)

    print("\n🧾 Classification Report:")
    print(report)
    print("📈 AUROC:", auc(fpr, tpr))
    print(f"✅ Done: {len(samples)} samples evaluated.")


if __name__ == "__main__":
    run_pipeline(threshold=0.5, max_samples=100)


# import torch
# import json
# from pathlib import Path
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_fscore_support
# from tqdm import tqdm

# from lettucedetect.datasets.hallucination_dataset import HallucinationSample, HallucinationData



# def load_model():
#     model = AutoModelForSequenceClassification.from_pretrained(
#         "newmindai/TurkEmbed4STS-HallucinationDetection", trust_remote_code=True
#     )
#     tokenizer = AutoTokenizer.from_pretrained(
#         "newmindai/TurkEmbed4STS-HallucinationDetection", trust_remote_code=True
#     )
#     return tokenizer, model


# def predict_samples(tokenizer, model, dataset):
#     model.eval()
#     new_samples = []

#     for ex in tqdm(dataset, desc="Predicting"):
#         prompt = ex["prompt"]
#         answer = ex["answer"]  
#         split = ex.get("split", "train")
#         task_type = ex.get("task_type", "unknown")
#         dataset_name = ex.get("dataset", "unknown")
#         language = ex.get("language", "tr")

#         text = f"Soru: {prompt} Yanıt: {answer}"
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = model.to(device)
#         inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)


#         with torch.no_grad():
#             logits = model(**inputs).logits
#             pred = torch.argmax(logits, dim=1).item()

#         sample = HallucinationSample(
#             prompt=prompt,
#             answer=answer,
#             labels=[{f"threshold_0.5": pred}],
#             split=split,
#             task_type=task_type,
#             dataset=dataset_name,
#             language=language,
#         )
#         new_samples.append(sample)

#     return HallucinationData(samples=new_samples)


# def evaluate_model(ground_truth: HallucinationData, predictions: HallucinationData, threshold=0.5):
#     true_labels = []
#     pred_labels = []

#     for gt, pred in zip(ground_truth.samples, predictions.samples):
#         is_hallucinated = 1 if gt.labels else 0
#         pred_hallucinated = 1 if pred.labels[0][f"threshold_{threshold}"] else 0
#         true_labels.append(is_hallucinated)
#         pred_labels.append(pred_hallucinated)

#     precision, recall, f1, _ = precision_recall_fscore_support(
#         true_labels, pred_labels, labels=[0, 1], average=None, zero_division=0
#     )

#     report = classification_report(true_labels, pred_labels, target_names=["Supported", "Hallucinated"], digits=4)

#     fpr, tpr, _ = roc_curve(true_labels, pred_labels)
#     auroc = auc(fpr, tpr)

#     print("\n--- Classification Report ---")
#     print(report)
#     print(f"\nAUROC: {auroc:.4f}")

#     return {
#         "supported": {"precision": precision[0], "recall": recall[0], "f1": f1[0]},
#         "hallucinated": {"precision": precision[1], "recall": recall[1], "f1": f1[1]},
#         "auroc": auroc,
#         "report": report,
#     }


# def main():
#     print("🔹 Model ve tokenizer yükleniyor...")
#     tokenizer, model = load_model()

#     print("🔹 Dataset yükleniyor...")
#     full_ds = load_dataset("newmindai/hallucination_detection_turkish")["train"]
#     # Sadece test split'indeki örnekleri al
#     ds = [ex for ex in full_ds if ex.get("split") == "test"]

#     print("🔹 Ground truth verisi hazırlanıyor...")
#     gt_samples = []
#     for ex in ds:
#         # labels alanı varsa halüsinasyon, yoksa desteklenen
#         is_hallucinated = 1 if ex["labels"] else 0
#         gold_label = [{f"threshold_0.5": 1}] if is_hallucinated == 1 else []
#         sample = HallucinationSample(
#             prompt=ex["prompt"],
#             answer=ex["answer"],  # 'response' yerine 'answer' kullanıyoruz
#             labels=gold_label,
#             split=ex.get("split", "test"),  # test split'ini kullan
#             task_type=ex.get("task_type", "unknown"),
#             dataset=ex.get("dataset", "unknown"),
#             language=ex.get("language", "tr"),
#         )
#         gt_samples.append(sample)
#     ground_truth_data = HallucinationData(samples=gt_samples)

#     print("🔹 Tahminler başlatılıyor...")
#     pred_data = predict_samples(tokenizer, model, ds)

#     output_path = Path("lettucedetect_predictions.json")
#     output_path.write_text(json.dumps(pred_data.to_json(), indent=4, ensure_ascii=False))
#     print(f"🔸 Tahminler kaydedildi: {output_path.absolute()}")

#     print("🔹 Değerlendirme başlatılıyor...")
#     results = evaluate_model(ground_truth_data, pred_data, threshold=0.5)

#     with open("lettucedetect_eval_report.txt", "w") as f:
#         f.write(results["report"])
#         f.write(f"\nAUROC: {results['auroc']:.4f}")
#     print("🔸 Değerlendirme raporu kaydedildi: lettucedetect_eval_report.txt")





# if __name__ == "__main__":
#     main()


