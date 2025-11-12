from collections import defaultdict
from dataclasses import dataclass, field
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")


## DEFINING TYPES


@dataclass
class CorefMention:
    start_idx: int
    end_idx: int


@dataclass
class CorefChain:
    mentions: list[CorefMention] = field(default_factory=list)


CorefOutput = list[CorefChain]


@dataclass
class MatchedMention:
    gold: int
    pred: int


@dataclass
class MatchedMentionsOutput:
    mentions: list[MatchedMention]
    gold_coref: dict[int, int]
    pred_coref: dict[int, int]
    gold_chains: dict[int, set[int]]
    pred_chains: dict[int, set[int]]


MentionIdx = int
ChainIdx = int


## DATA LOADING


def _find_sentence_spans(text: str) -> list[tuple[int, int]]:
    """
    Find character positions of sentences in the original text.

    Args:
        text: The original text

    Returns:
        list of (start_pos, end_pos) tuples for each sentence
    """
    sentences = nltk.sent_tokenize(text)
    spans = []
    search_pos = 0

    for sentence in sentences:
        # Find this sentence in the text starting from search_pos
        start_pos = text.find(sentence, search_pos)
        if start_pos == -1:
            # Sentence not found exactly, skip it
            continue
        end_pos = start_pos + len(sentence)
        spans.append((start_pos, end_pos))
        search_pos = end_pos

    return spans


def read_text_in_sentences(text: str, k: int = 3) -> list[str]:
    """
    Read the text into k-sentence chunks, preserving original whitespace.

    Args:
        text: The original text
        k: Number of sentences per chunk

    Returns:
        list[str]: List of text chunks with original whitespace preserved
    """
    # Get sentence spans with character positions
    sentence_spans = _find_sentence_spans(text)

    chunks = []
    for i in range(0, len(sentence_spans), k):
        # Get k sentences starting at index i
        chunk_sentences = sentence_spans[i : i + k]
        if not chunk_sentences:
            continue

        # Extract from the start of the first sentence to the end of the last sentence
        start_pos = chunk_sentences[0][0]
        end_pos = chunk_sentences[-1][1]
        chunks.append(text[start_pos:end_pos])

    return chunks


def load_conll_data(
    text_file: str, conll_file: str, k: int = 3
) -> tuple[list[str], list[CorefOutput]]:
    """
    Load coreference data from a CoNLL format file.

    Args:
        text_file: Path to the original text file
        conll_file: Path to the CoNLL format annotation file
        k: Number of sentences per chunk

    Returns:
        list[str]: List of text chunks
        list[CorefOutput]: List of CorefChain objects with character-level span indices for each text chunk
    """
    # Read the original text to map token positions to character positions
    with open(text_file, "r") as f:
        text = f.read()

    # Parse the entire CoNLL file once to get all mentions in the full text
    full_text_chains = _parse_conll_file(text, conll_file)

    # Get sentence spans with character positions
    sentence_spans = _find_sentence_spans(text)

    # Create chunks and their ranges
    chunks = []
    chunk_ranges = []
    for i in range(0, len(sentence_spans), k):
        # Get k sentences starting at index i
        chunk_sentences = sentence_spans[i : i + k]
        if not chunk_sentences:
            continue

        # Extract from the start of the first sentence to the end of the last sentence
        start_pos = chunk_sentences[0][0]
        end_pos = chunk_sentences[-1][1]
        chunks.append(text[start_pos:end_pos])
        chunk_ranges.append((start_pos, end_pos))

    # Extract mentions for each chunk and adjust indices
    coref_outputs = []
    for chunk_start, chunk_end in chunk_ranges:
        coref_outputs.append(
            _extract_chunk_mentions(full_text_chains, chunk_start, chunk_end)
        )

    return chunks, coref_outputs


def _parse_conll_file(text: str, conll_file: str) -> dict[int, list[tuple[int, int]]]:
    """
    Parse the CoNLL file and return all mentions with character positions in the full text.

    Returns:
        dict mapping chain_id -> list of (start_char, end_char) tuples
    """
    chains = defaultdict(list)
    open_mentions = defaultdict(list)  # chain_id -> stack of start positions

    char_position = 0

    with open(conll_file, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Split by tabs
            parts = line.split("\t")
            if len(parts) < 12:
                continue

            token = parts[3]
            coref_annotation = parts[-1]

            # Find the token in the text starting from current position
            token_start = text.find(token, char_position)
            if token_start == -1:
                # Token not found - skip
                continue

            token_end = token_start + len(token)
            char_position = token_end

            # Parse coreference annotations
            if coref_annotation and coref_annotation != "_":
                # Process each annotation in the field
                # Examples: "(0)", "(0", "0)", "(0|(1)"
                i = 0
                while i < len(coref_annotation):
                    if coref_annotation[i] == "(":
                        # Start of a mention
                        j = i + 1
                        while (
                            j < len(coref_annotation) and coref_annotation[j].isdigit()
                        ):
                            j += 1
                        chain_id = int(coref_annotation[i + 1 : j])

                        # Check if this is a single-token mention
                        if j < len(coref_annotation) and coref_annotation[j] == ")":
                            chains[chain_id].append((token_start, token_end))
                            i = j + 1
                        else:
                            # Multi-token mention start
                            open_mentions[chain_id].append(token_start)
                            i = j
                    elif coref_annotation[i].isdigit():
                        # End of a mention
                        j = i
                        while (
                            j < len(coref_annotation) and coref_annotation[j].isdigit()
                        ):
                            j += 1
                        chain_id = int(coref_annotation[i:j])
                        if j < len(coref_annotation) and coref_annotation[j] == ")":
                            if open_mentions[chain_id]:
                                start_char = open_mentions[chain_id].pop()
                                chains[chain_id].append((start_char, token_end))
                            i = j + 1
                        else:
                            i = j
                    else:
                        i += 1

    return dict(chains)


def _extract_chunk_mentions(
    full_text_chains: dict[int, list[tuple[int, int]]], chunk_start: int, chunk_end: int
) -> CorefOutput:
    """
    Extract mentions that fall within a chunk and adjust indices to be relative to chunk start.

    Args:
        full_text_chains: dict mapping chain_id -> list of (start_char, end_char) in full text
        chunk_start: Start character position of chunk in full text
        chunk_end: End character position of chunk in full text

    Returns:
        CorefOutput with mentions adjusted to be relative to chunk start
    """
    # Group mentions by chain, only including those within the chunk
    chunk_chains = defaultdict(list)

    for chain_id, mentions in full_text_chains.items():
        for start, end in mentions:
            # Check if mention overlaps with chunk
            if start < chunk_end and end > chunk_start:
                # Adjust indices to be relative to chunk start
                adjusted_start = max(0, start - chunk_start)
                adjusted_end = min(chunk_end - chunk_start, end - chunk_start)
                chunk_chains[chain_id].append((adjusted_start, adjusted_end))

    # Convert to CorefOutput format
    result = []
    for chain_id in sorted(chunk_chains.keys()):
        mention_spans = chunk_chains[chain_id]
        # Sort mentions by start index
        mention_spans.sort(key=lambda x: x[0])

        mentions = [
            CorefMention(start_idx=start, end_idx=end) for start, end in mention_spans
        ]
        result.append(CorefChain(mentions=mentions))

    return result


## EVALUATION


def strict(mention_a: CorefMention, mention_b: CorefMention):
    return (
        mention_a.start_idx == mention_b.start_idx
        and mention_a.end_idx == mention_b.end_idx
    )


def jaccard(threshold=0.0):
    # jaccard matching matches two mentions if the span overlaps are greater than a threshold
    def _closure(mention_a: CorefMention, mention_b: CorefMention):
        intersection_start = max(mention_a.start_idx, mention_b.start_idx)
        intersection_end = min(mention_a.end_idx, mention_b.end_idx)
        intersection_size = max(0, intersection_end - intersection_start)

        union_start = min(mention_a.start_idx, mention_b.start_idx)
        union_end = max(mention_a.end_idx, mention_b.end_idx)
        union_size = union_end - union_start

        # Calculate Jaccard similarity
        if union_size == 0:
            jaccard_score = 0.0
        else:
            jaccard_score = intersection_size / union_size

        # Return True if Jaccard score exceeds threshold
        return jaccard_score > threshold

    return _closure


def overlap(mention_a: CorefMention, mention_b: CorefMention):
    return jaccard(0.0)(mention_a, mention_b)


def match_mentions(
    gold: CorefOutput, pred: CorefOutput, strategy=strict
) -> MatchedMentionsOutput:
    # match mentions between two coref outputs
    # strict matching only matches two mentions if the spans are exactly the same
    # overlap matching matches two mentions if the spans overlap
    # jaccard matching matches two mentions if the span overlaps are greater than a threshold

    # Flatten gold and pred mentions with their chain indices
    def _flatten(
        coref_output: CorefOutput,
    ) -> tuple[list[CorefMention], dict[MentionIdx, ChainIdx]]:
        mentions = []
        corefs = {}  # mention_idx -> chain_idx
        for chain_idx, chain in enumerate(coref_output):
            for mention in chain.mentions:
                mention_idx = len(mentions)
                mentions.append(mention)
                corefs[mention_idx] = chain_idx
        return mentions, corefs

    gold_mentions, gold_corefs = _flatten(gold)
    pred_mentions, pred_corefs = _flatten(pred)

    # Match mentions using the provided strategy
    matched = []
    used_pred = set()

    # Match gold mentions to pred mentions
    for gold_idx, gold_mention in enumerate(gold_mentions):
        for pred_idx, pred_mention in enumerate(pred_mentions):
            if pred_idx not in used_pred and strategy(gold_mention, pred_mention):
                matched.append(MatchedMention(gold=gold_idx, pred=pred_idx))
                used_pred.add(pred_idx)
                break
        else:
            # No match found for this gold mention
            matched.append(MatchedMention(gold=gold_idx, pred=-1))

    # Add unmatched pred mentions
    for pred_idx in range(len(pred_mentions)):
        if pred_idx not in used_pred:
            matched.append(MatchedMention(gold=-1, pred=pred_idx))

    matched_gold = {}
    matched_pred = {}
    for match_id, match in enumerate(matched):
        matched_gold[match_id] = gold_corefs[match.gold] if match.gold != -1 else -1
        matched_pred[match_id] = pred_corefs[match.pred] if match.pred != -1 else -1

    gold_chains: dict[int, set[int]] = defaultdict(set)
    pred_chains: dict[int, set[int]] = defaultdict(set)
    for match_id, coref in matched_gold.items():
        if coref == -1:
            continue
        gold_chains[coref].add(match_id)
    for match_id, coref in matched_pred.items():
        if coref == -1:
            continue
        pred_chains[coref].add(match_id)
    gold_chains = dict(gold_chains)
    pred_chains = dict(pred_chains)
    gold_chains[-1] = set()
    pred_chains[-1] = set()

    return MatchedMentionsOutput(
        mentions=matched,
        gold_coref=matched_gold,
        pred_coref=matched_pred,
        gold_chains=gold_chains,
        pred_chains=pred_chains,
    )


def b3_recall(gold: CorefOutput, pred: CorefOutput, strategy=strict):
    matched_mentions = match_mentions(gold, pred, strategy)

    n = len(matched_mentions.mentions)
    total = 0
    for mention_id, _ in enumerate(matched_mentions.mentions):
        gold_chain = matched_mentions.gold_chains[
            matched_mentions.gold_coref[mention_id]
        ]
        pred_chain = matched_mentions.pred_chains[
            matched_mentions.pred_coref[mention_id]
        ]

        if len(gold_chain) == 0:
            continue

        total += len(gold_chain & pred_chain) / len(gold_chain)

    return total / n


def b3_precision(gold: CorefOutput, pred: CorefOutput, strategy=strict):
    matched_mentions = match_mentions(gold, pred, strategy)

    n = len(matched_mentions.mentions)
    total = 0
    for mention_id, _ in enumerate(matched_mentions.mentions):
        gold_chain = matched_mentions.gold_chains[
            matched_mentions.gold_coref[mention_id]
        ]
        pred_chain = matched_mentions.pred_chains[
            matched_mentions.pred_coref[mention_id]
        ]

        if len(pred_chain) == 0:
            continue

        total += len(gold_chain & pred_chain) / len(pred_chain)

    return total / n


def evaluate(
    gold_chunks: list[CorefOutput], pred_chunks: list[CorefOutput], strategy=strict
) -> tuple[float, float]:
    recalls = []
    precisions = []
    for gold_chunk, pred_chunk in zip(gold_chunks, pred_chunks):
        recalls.append(b3_recall(gold_chunk, pred_chunk, strategy))
        precisions.append(b3_precision(gold_chunk, pred_chunk, strategy))

    return sum(recalls) / len(recalls), sum(precisions) / len(precisions)
