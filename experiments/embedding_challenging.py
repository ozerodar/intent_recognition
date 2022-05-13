from intent_detection.intent.embedding import EmbeddingModel

models = ["all-MiniLM-L6-v2", "all-roberta-large-v1"]
sentences1 = [
    "call me tom",
    "She is going to the park",
    "The man took a bow to shoot",
    "I went to a river bank",
    "She knows him better than I",
]
sentences2 = [
    "i will call you tom",
    "She is going to park the car",
    "The man took a bow to the king",
    "I went to a bank",
    "She knows him better than me",
]


for model_name in models:
    model = EmbeddingModel(model_name)
    model.initialize()
    scores = model.pairwise_cosine_scores(sentences1, sentences2)
    print(f"model: {model_name}, score: {scores}")
