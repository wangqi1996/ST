from fairseq.scoring.tokenizer import EvaluationTokenizer


class WerScorer:
    def __init__(self):
        self.reset()
        try:
            import editdistance as ed
        except ImportError:
            raise ImportError("Please install editdistance to use WER scorer")
        self.ed = ed
        wer_tokenizer = "13a"
        wer_lowercase = False
        wer_remove_punct = False
        wer_char_level = False
        self.tokenizer = EvaluationTokenizer(
            tokenizer_type=wer_tokenizer,
            lowercase=wer_lowercase,
            punctuation_removal=wer_remove_punct,
            character_tokenization=wer_char_level,
        )

    def reset(self):
        self.distance = 0
        self.ref_length = 0

    def add_string(self, ref, pred):
        ref_items = self.tokenizer.tokenize(ref).split()
        pred_items = self.tokenizer.tokenize(pred).split()
        self.distance += self.ed.eval(ref_items, pred_items)
        self.ref_length += len(ref_items)

    def result_string(self):
        return f"WER: {self.score():.2f}"

    def score(self):
        return 100.0 * self.distance / self.ref_length if self.ref_length > 0 else 0


if __name__ == '__main__':
    src_filename = "/home/wangdq/test/asr"
    tgt_filename = "/home/wangdq/test/ref"
    scorer = WerScorer()
    with open(src_filename) as f_src, open(tgt_filename) as f_tgt:
        for src, tgt in zip(f_src, f_tgt):
            scorer.add_string(tgt, src)
    print(scorer.score())
