from pathlib import Path

from sentence_transformers import SentenceTransformer


# run_and_saver is a (lambda) function that takes 3 arguments: model, summary_path, targets_path. For example:
# lambda model, s_path: run_and_save(n_iter=150, length=6, capacity=.2137, model=model, summary_path=s_path,
#                                    targets_path="targets", subset_size=None)
def search_model(run_and_saver, summary_path="summary"):
    # 384 dimensions
    model = SentenceTransformer('all-MiniLM-L6-v2')
    run_and_saver(model, Path(summary_path) / "all-MiniLM-L6-v2")

    # 384 dimensions
    model = SentenceTransformer('all-MiniLM-L12-v2')
    run_and_saver(model, Path(summary_path) / "all-MiniLM-L12-v2")

    # 1024 dimensions (too big)
    # model = SentenceTransformer('all-roberta-large-v1')
    # run_and_saver(model, Path(summary_path) / "all-roberta-large-v1")

    # 768 dimensions
    model = SentenceTransformer('all-distilroberta-v1')
    run_and_saver(model, Path(summary_path) / "all-distilroberta-v1")

    # 768 dimensions
    model = SentenceTransformer('all-mpnet-base-v2')
    run_and_saver(model, Path(summary_path) / "all-mpnet-base-v2")
