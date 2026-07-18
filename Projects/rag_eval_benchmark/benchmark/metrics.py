"""
Metrics — the "RAGAS + Custom Metrics" step. Scores one built pipeline
against the Q&A test set using both a fast custom metric (hit_at_k) and,
if an LLM is configured, RAGAS's LLM-judge metrics.
"""
import os

#simple metric for hit@k: does the expected keyword appear in the top-k retrieved docs?
def hit_at_k(docs, expected_keyword):
    combined = " ".join(doc.page_content.lower() for doc in docs)
    return expected_keyword.lower() in combined

#this is custom metric that scores the pipeline based on whether the expected keyword appears in the top-k retrieved documents. It returns a list of scores (1 for hit, 0 for miss) for each question-answer pair in the test set.
def score_pipeline_custom(pipeline, qa_pairs):
    scores = []
    for pair in qa_pairs:
        docs = pipeline["retriever"].invoke(pair["question"])
        scores.append(1 if hit_at_k(docs, pair["expected_keyword"]) else 0)
    return scores


def score_pipeline_ragas(pipeline, qa_pairs, llm=None, embeddings=None):
    if llm is None or not os.getenv("GROQ_API_KEY"):
        return None

#this try and except block is used to handle the case where the RAGAS library is not installed or not available. It attempts to import the necessary modules from RAGAS and, if successful, evaluates the pipeline using RAGAS metrics. If any exception occurs during this process, it prints an error message and returns None, indicating that RAGAS scoring failed for the given pipeline.
#it also includes a workaround for the case where the langchain_community.chat_models.vertexai module is not available, by creating stub classes to avoid import errors.
    try:
        import sys
        import types
        try:
            import langchain_community.chat_models.vertexai
        except ModuleNotFoundError:
            stub1 = types.ModuleType("langchain_community.chat_models.vertexai")
            stub1.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules["langchain_community.chat_models.vertexai"] = stub1
            stub2 = types.ModuleType("langchain_community.llms.vertexai")
            stub2.VertexAI = type("VertexAI", (), {})
            sys.modules["langchain_community.llms.vertexai"] = stub2

        from ragas import evaluate, EvaluationDataset, SingleTurnSample
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig

        answer_relevancy.strictness = 1  # Groq only supports n=1 means generates 1 alternative question

        samples = []
        for pair in qa_pairs:
            docs = pipeline["retriever"].invoke(pair["question"])
            contexts = [d.page_content for d in docs]
            context_text = "\n\n".join(contexts)
            answer = llm.invoke(
                f"Answer using only this context. Say 'I don't know' if it's "
                f"not covered.\n\nContext:\n{context_text}\n\n"
                f"Question: {pair['question']}\n\nAnswer:"
            ).content
            reference = pair.get("ground_truth", f"The answer relates to: {pair['expected_keyword']}.")
            samples.append(SingleTurnSample(
                user_input=pair["question"], retrieved_contexts=contexts,
                response=answer, reference=reference,
            ))

        dataset = EvaluationDataset(samples=samples)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=LangchainLLMWrapper(llm),
            embeddings=LangchainEmbeddingsWrapper(embeddings),
            run_config=RunConfig(max_workers=1, timeout=180),
            raise_exceptions=False,
        )
        df = result.to_pandas()
        return {
            "faithfulness": df["faithfulness"].tolist() if "faithfulness" in df else None,
            "answer_relevancy": df["answer_relevancy"].tolist() if "answer_relevancy" in df else None,
            "context_precision": df["context_precision"].tolist() if "context_precision" in df else None,
        }
    except Exception as e:
        print(f"[metrics] RAGAS scoring failed for {pipeline['name']}: {e}")
        return None