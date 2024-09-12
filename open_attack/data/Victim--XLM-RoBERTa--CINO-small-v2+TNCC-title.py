from OpenAttack.utils import make_zip_downloader

NAME = "Victim.XLM-RoBERTa.CINO-small-v2+TNCC-title"

DOWNLOAD = make_zip_downloader("")


def LOAD(path):
    from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
    model = XLMRobertaForSequenceClassification.from_pretrained(path, num_labels=12)
    tokenizer = XLMRobertaTokenizer.from_pretrained(path)

    from OpenAttack.victim.classifiers import TransformersClassifier
    return TransformersClassifier(model, tokenizer, embedding_layer=model.roberta.embeddings.word_embeddings,
                                  max_length=512, batch_size=32, lang="tibetan")
