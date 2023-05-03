import logging
import re
import string
from pathlib import Path
from typing import Callable, Dict, List, Optional
from tqdm.autonotebook import tqdm
import nltk
from haystack.nodes.file_converter import (
    BaseConverter,
    DocxToTextConverter,
    PDFToTextConverter,
    TextConverter,
)
from haystack.schema import Document
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline

nltk.download("punkt", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

logger = logging.getLogger(__name__)


class TextProcessor:
    def __init__(self, min_characters: int = 20, threshold=0.6):
        """Initialize TextProcessor."""

        self.min_characters = min_characters
        self.threshold = threshold
        self.model = pipeline(
            "text-classification",
            model="dennlinger/bert-wiki-paragraphs",
            truncation=True,
        )
        self.last_scores = None  # store last scores for debugging

    def _get_is_continuous(self, sentences) -> List[bool]:
        """Get a list of boolean indicating whether two sentences are about continuous topic."""

        xs = [
            f"{last} [SEP] {this}" for last, this in zip(sentences[:-1], sentences[1:])
        ]
        ys = self.model(xs)
        self.last_scores = [y["score"] for y in ys]
        return [y["score"] > self.threshold for y in ys]

    def to_paragraphs(self, text) -> str:
        """Separate text into paragraphs using `\n\n` as separator."""

        text = re.sub("\s+", " ", text).strip()
        sentences = sent_tokenize(text)
        sentences = [s for s in sentences if len(s) > self.min_characters]
        is_continuous = self._get_is_continuous(sentences)

        # Group sentences into paragraphs
        paragraphs = [sentences[0]]
        for i, sentence in enumerate(sentences[1:]):
            not_too_long = len(word_tokenize(paragraphs[-1] + " " + sentence)) < 1024
            if is_continuous[i] and not_too_long:
                paragraphs[-1] += " " + sentence  # Append to last paragraph
            else:
                paragraphs.append(sentence)  # Start a new paragraph

        return "\n\n".join(paragraphs)


def convert_files_to_docs(
    dir_path: str,
    clean_func: Optional[Callable] = None,
    split_paragraphs: bool = True,
    encoding: Optional[str] = None,
    id_hash_keys: Optional[List[str]] = None,
) -> List[Document]:
    """
    Convert all files(.txt, .pdf, .docx) in the sub-directories of the given path to Documents that can be written to a
    Document Store.

    :param dir_path: The path of the directory containing the Files.
    :param clean_func: A custom cleaning function that gets applied to each Document (input: str, output: str).
    :param split_paragraphs: Whether to split text by paragraph.
    :param encoding: Character encoding to use when converting pdf documents.
    :param id_hash_keys: A list of Document attribute names from which the Document ID should be hashed from.
            Useful for generating unique IDs even if the Document contents are identical.
            To ensure you don't have duplicate Documents in your Document Store if texts are
            not unique, you can modify the metadata and pass [`"content"`, `"meta"`] to this field.
            If you do this, the Document ID will be generated by using the content and the defined metadata.
    """
    file_paths = [p for p in Path(dir_path).glob("**/*")]
    allowed_suffixes = [".pdf", ".txt", ".docx"]
    suffix2converter: Dict[str, BaseConverter] = {}

    suffix2paths: Dict[str, List[Path]] = {}
    for path in file_paths:
        file_suffix = path.suffix.lower()
        if file_suffix in allowed_suffixes:
            if file_suffix not in suffix2paths:
                suffix2paths[file_suffix] = []
            suffix2paths[file_suffix].append(path)
        elif not path.is_dir():
            logger.warning(
                "Skipped file %s as type %s is not supported here. "
                "See haystack.file_converter for support of more file types",
                path,
                file_suffix,
            )

    # No need to initialize converter if file type not present
    for file_suffix in suffix2paths.keys():
        if file_suffix == ".pdf":
            suffix2converter[file_suffix] = PDFToTextConverter()
        if file_suffix == ".txt":
            suffix2converter[file_suffix] = TextConverter()
        if file_suffix == ".docx":
            suffix2converter[file_suffix] = DocxToTextConverter()

    documents = []
    for suffix, paths in tqdm(suffix2paths.items()):
        for path in paths:
            logger.info("Converting %s", path)
            # PDFToTextConverter, TextConverter, and DocxToTextConverter return a list containing a single Document
            document = suffix2converter[suffix].convert(
                file_path=path, meta=None, encoding=encoding, id_hash_keys=id_hash_keys
            )[0]
            text = document.content

            if clean_func:
                text = clean_func(text)

            if split_paragraphs:
                for para in text.split("\n\n"):
                    if not para.strip():  # skip empty paragraphs
                        continue
                    documents.append(
                        Document(
                            content=para,
                            meta={"name": path.name},
                            id_hash_keys=id_hash_keys,
                        )
                    )
            else:
                documents.append(
                    Document(
                        content=text,
                        meta={"name": path.name},
                        id_hash_keys=id_hash_keys,
                    )
                )

    return documents


def to_sentences(text: str) -> List[str]:
    """Generic text cleaning function."""

    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]
    words = [
        [word.lower() for word in sentence if word not in string.punctuation]
        for sentence in words
    ]

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = [
        [word for word in sentence if word not in stop_words] for sentence in words
    ]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in words]

    return [" ".join(sentence) for sentence in words]


def to_chunks(sentences: List[str], n=5000) -> List[str]:
    """Concatenate sentences into chunks with around n characters."""
    chunks = []

    for i, sentence in enumerate(sentences):
        if i == 0:
            chunks.append(sentence)
        else:
            if len(chunks[-1]) + len(sentence) < n:
                chunks[-1] += "\n" + sentence
            else:
                chunks.append(sentence)

    return chunks