# to try out
from src.adaptive_rag import AdaptiveRAG


def main():
    rag = AdaptiveRAG()
    # there re just the examples - change to whatever u like
    queries = [
        "What mathematical techniques help prove stability of recurrent neural networks?",
        "What are the latest advances in transformer architectures this week?",
        "What techniques reduce catastrophic forgetting in continual learning?",
        "How to prevent mode collapse in GANs during training?",
        "What happened in machine learning in the last 2 weeks?",
    ]
    
    # run first query
    user_query = queries[0]
    result = rag.run_full_pipeline(user_query)
    
    if result:
        print("\n\n Pipeline completed successfully")
        print(f"Loss: {result['loss']:.3f}")
        print(f"NDCG@10: {result['metrics']['ndcg@k']:.3f}")


if __name__ == "__main__":
    main()

