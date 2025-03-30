from __future__ import annotations

from enum import Enum
from pathlib import Path

import jiwer
import polars as pl
from loguru import logger

transform_complete = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
    ],
)

transform_whitespace = jiwer.Compose(
    [
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ],
)

transform_punctuation = jiwer.Compose(
    [
        jiwer.RemovePunctuation(),
    ],
)


class NormalizationType(Enum):
    """Enumeration of different text normalization types for ASR evaluation.

    The normalization types control how reference and hypothesis texts
    are preprocessed before metric calculation.

    Parameters
    ----------
    value : str
        The string representation of the normalization type.

    Attributes
    ----------
    NONE : str
        No normalization applied, texts remain unchanged.
    LOWERCASE : str
        Convert text to lowercase.
    WHITESPACE : str
        Normalize whitespace characters, remove extra spaces.
    PUNCTUATION : str
        Remove punctuation from text.
    TAGS : str
        Remove special tags like <unk>, <silence>, etc.
    DICTIONARY : str
        Apply dictionary-based normalization (not implemented yet).
    ALL : str
        Apply all normalization types in sequence.

    """

    NONE = "none"
    LOWERCASE = "lowercase"
    WHITESPACE = "whitespace"
    PUNCTUATION = "punctuation"
    TAGS = "tags"
    DICTIONARY = "dictionary"
    ALL = "all"


def remove_tags(
    text: str,
    tags: list[str] | None = None,
) -> str:
    """Remove tags from a string.

    Parameters
    ----------
    text : str
        String to remove tags from.
    tags : list[str], optional
        List of tags to remove. Default is ["<unk>", "<silence>", "trunc"].

    Returns
    -------
    str
        String with tags removed.

    """
    if tags is None:
        tags = ["<unk>", "<silence>", "trunc"]

    lowercase_tags = [tag.lower() for tag in tags]

    words = text.split()

    # Remove standalone tags (exact matches)
    # Example: "this is <unk>" -> "this is"
    without_stand_alone_tags = [
        word for word in words if word.lower() not in lowercase_tags
    ]

    # Remove words containing any tag as substring
    # Examples:
    # - "word_trunc" -> filtered out
    # - "trunc_word" -> filtered out
    # - "word<unk>word" -> filtered out
    # Case-insensitive matching is used
    return " ".join(
        word
        for word in without_stand_alone_tags
        if not any(tag in word.lower() for tag in lowercase_tags)
    )


def prepare_references_and_hyphoteses(
    lf_eval_input: pl.LazyFrame,
    ref_col_name: str,
    hyp_col_name: str,
    audio_path_col_name: str,
    normalization_type: NormalizationType,
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
]:
    """Prepare reference and hypothesis pairs for evaluation with text normalization.

    This function reads a TSV file containing reference and hypothesis pairs,
    applies various text normalization techniques based on the specified type,
    and returns the processed text along with audio paths and IDs.

    Parameters
    ----------
    lf_eval_input : pl.LazyFrame
        LazyFrame containing reference and hypothesis pairs.
    ref_col_name : str
        Name of the column containing reference transcriptions.
    hyp_col_name : str
        Name of the column containing hypothesis transcriptions.
    audio_path_col_name : str
        Name of the column containing audio paths.
    normalization_type : NormalizationType
        Type of text normalization to apply to both references and hypotheses.

    Returns
    -------
    references : list[str]
        List of normalized reference transcriptions.
    hypotheses : list[str]
        List of normalized hypothesis transcriptions.
    audio_paths : list[str]
        List of paths to the audio files.
    ids : list[str]
        List of audio file names extracted from the audio paths.

    Raises
    ------
    ValueError
        If the specified columns do not exist in the input file or if the
        number of references and hypotheses are not equal.
    NotImplementedError
        If dictionary normalization is requested (not yet implemented).

    """
    logger.opt(colors=True).info(
        (
            "Preprocessing references and hypotheses | "
            "NormType: <magenta>{}</magenta>"
        ),
        normalization_type.value,
    )

    # Check if ref_col_name and hyp_col_name exist in the lazyframe
    if (
        ref_col_name not in lf_eval_input.collect_schema().names()
        or hyp_col_name not in lf_eval_input.collect_schema().names()
    ):
        msg = (
            f"Columns '{ref_col_name}' or '{hyp_col_name}' "
            "not found in the input file"
        )
        raise ValueError(msg)

    lf_eval_input = lf_eval_input.filter(
        pl.col(hyp_col_name).is_not_null(),
        ~pl.col(hyp_col_name).is_in(["EMPTY", "INVALID"]),
    )

    # Also filter out null references to ensure both columns have valid data
    lf_eval_input = lf_eval_input.filter(
        pl.col(ref_col_name).is_not_null(),
    )

    df_eval_input = lf_eval_input.collect()

    # Verify each row has both reference and hypothesis
    row_count = df_eval_input.height
    ref_count = df_eval_input.filter(pl.col(ref_col_name).is_not_null()).height
    hyp_count = df_eval_input.filter(pl.col(hyp_col_name).is_not_null()).height

    if ref_count != hyp_count or ref_count != row_count:
        msg = (
            f"Number of valid references ({ref_count}) and "
            "hypotheses ({hyp_count}) must be equal"
        )
        raise ValueError(msg)

    df_eval_input = df_eval_input.with_columns(
        [
            pl.col(ref_col_name).cast(pl.Utf8),
            pl.col(hyp_col_name).cast(pl.Utf8),
        ],
    )

    match normalization_type:
        case NormalizationType.NONE:
            pass
        case NormalizationType.LOWERCASE:
            df_eval_input = df_eval_input.with_columns(
                [
                    pl.col(ref_col_name).str.to_lowercase(),
                    pl.col(hyp_col_name).str.to_lowercase(),
                ],
            )
        case NormalizationType.WHITESPACE:
            df_eval_input = df_eval_input.with_columns(
                [
                    pl.col(ref_col_name).map_elements(
                        transform_whitespace,
                        return_dtype=pl.Utf8,
                    ),
                    pl.col(hyp_col_name).map_elements(
                        transform_whitespace,
                        return_dtype=pl.Utf8,
                    ),
                ],
            )
        case NormalizationType.PUNCTUATION:
            df_eval_input = df_eval_input.with_columns(
                [
                    pl.col(ref_col_name).map_elements(
                        transform_punctuation,
                        return_dtype=pl.Utf8,
                    ),
                    pl.col(hyp_col_name).map_elements(
                        transform_punctuation,
                        return_dtype=pl.Utf8,
                    ),
                ],
            )
        case NormalizationType.TAGS:
            df_eval_input = df_eval_input.with_columns(
                [
                    pl.col(ref_col_name).map_elements(
                        remove_tags,
                        return_dtype=pl.Utf8,
                    ),
                    pl.col(hyp_col_name).map_elements(
                        remove_tags,
                        return_dtype=pl.Utf8,
                    ),
                ],
            )
        case NormalizationType.DICTIONARY:
            # TODO: Implement dictionary normalization
            msg = "Dictionary normalization is not implemented yet"
            raise NotImplementedError(msg)
        case NormalizationType.ALL:
            df_eval_input = df_eval_input.with_columns(
                [
                    # Apply all transformations in sequence
                    pl.col(ref_col_name)
                    .str.to_lowercase()
                    .map_elements(transform_whitespace, return_dtype=pl.Utf8)
                    .map_elements(transform_punctuation, return_dtype=pl.Utf8)
                    .map_elements(remove_tags, return_dtype=pl.Utf8),
                    pl.col(hyp_col_name)
                    .str.to_lowercase()
                    .map_elements(transform_whitespace, return_dtype=pl.Utf8)
                    .map_elements(transform_punctuation, return_dtype=pl.Utf8)
                    .map_elements(remove_tags, return_dtype=pl.Utf8),
                ],
            )

    # Filter empty references and hypotheses
    # Use parentheses to clarify logical operation precedence
    df_eval_input = df_eval_input.filter(
        (pl.col(ref_col_name) != "") & (pl.col(hyp_col_name) != ""),
    )

    references = df_eval_input.get_column(ref_col_name).to_list()
    hypotheses = df_eval_input.get_column(hyp_col_name).to_list()
    audio_paths = df_eval_input.get_column(audio_path_col_name).to_list()
    ids = [Path(audio_path).name for audio_path in audio_paths]

    return references, hypotheses, audio_paths, ids
