from lettucedetect.models.inference import HallucinationDetector

# For English:
# detector = HallucinationDetector(
#     method="transformer", 
#     model_path="selvatas/lettucedect-210m-eurobert-tr-v1",
# )

# For other languages (e.g., German):
detector = HallucinationDetector(
    method="transformer", 
    model_path="selvatas/lettucedect-210m-eurobert-tr-v1",
    lang="tr",
    **{"trust_remote_code": True}
)

contexts = ["Elon Musk, 2022 yılında Twitter'ı satın aldı. Satın alma süreci oldukça tartışmalı geçti ve platformda büyük değişiklikler yapıldı",]
question = "Twitter'ı kim satın aldı?"
answer = "Bill"
# contexts = ["Elon Musk acquired Twitter in 2022. The acquisition process was highly controversial, and major changes were made to the platform."]
# question = "Who acquired Twitter?"
# answer = "Bill Gates"

# Get span-level predictions indicating which parts of the answer are considered hallucinated.
predictions = detector.predict(context=contexts, question=question, answer=answer, output_format="spans")
print("Predictions:", predictions)

# Predictions: [{'start': 31, 'end': 71, 'confidence': 0.9944414496421814, 'text': ' The population of France is 69 million.'}]





# from lettucedetect.detectors.prompt_utils import PromptUtils
 
# ctx = ["LLM'ler büyük dillerde oldukça güçlüdür.",

#        "Küçük diller için ise ek veri toplanması gerekebilir."]

# question = "LLM'lerin küçük dillerde zayıf olmasının nedeni nedir?"
 
# print(PromptUtils.format_context(ctx, question, "tr"))

 