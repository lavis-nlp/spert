import os
import warnings
from typing import List, Tuple, Dict

import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from transformers import BertTokenizer

from spert.entities import Entity, Relation, Document, Dataset
from spert.input_reader import JsonInputReader
from spert.opt import jinja2
from spert.sampling import TrainTensorBatch, EvalTensorBatch

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class Evaluator:
    def __init__(self, dataset: Dataset, input_reader: JsonInputReader, text_encoder: BertTokenizer,
                 rel_filter_threshold: float, example_count: int, example_path: str,
                 epoch: int, dataset_label: str):
        self._text_encoder = text_encoder
        self._input_reader = input_reader
        self._dataset = dataset
        self._rel_filter_threshold = rel_filter_threshold

        self._epoch = epoch
        self._dataset_label = dataset_label

        self._example_count = example_count
        self._examples_path = example_path

        # relations
        self._gt_relations = []  # ground truth
        self._pred_relations = []  # prediction

        # entities
        self._gt_entities = []  # ground truth
        self._pred_entities = []  # prediction

        self._convert_gt(self._dataset.documents)

    def eval_batch(self, batch_entity_clf: torch.tensor, batch_rel_clf: torch.tensor,
                   rels: torch.tensor, batch: EvalTensorBatch):
        batch_size = batch_rel_clf.shape[0]
        rel_class_count = batch_rel_clf.shape[2]

        # get maximum activation (index of predicted entity type)
        batch_entity_types = batch_entity_clf.argmax(dim=-1)
        # apply entity sample mask
        batch_entity_types *= batch.entity_sample_masks.long()

        batch_rel_clf = batch_rel_clf.view(batch_size, -1)

        # apply threshold to relations
        if self._rel_filter_threshold > 0:
            batch_rel_clf[batch_rel_clf < self._rel_filter_threshold] = 0

        for i in range(batch_size):
            # get model predictions for sample
            pred_rel_clf = batch_rel_clf[i]
            pred_entity_types = batch_entity_types[i]

            # get predicted relation labels and corresponding entity pairs
            indices = pred_rel_clf.nonzero().view(-1)
            values = pred_rel_clf[indices]

            pred_rel_types = (indices % rel_class_count) + 1  # model does not predict None class (+1)
            pred_rel_indices = indices // rel_class_count

            pred_rels = rels[i][pred_rel_indices]

            # get masks of entities in relation
            pred_rel_entity_spans = batch.entity_spans[i][pred_rels].long()

            # get predicted entity types
            pred_rel_entity_types = torch.zeros([pred_rels.shape[0], 2])
            if pred_rels.shape[0] != 0:
                pred_rel_entity_types = torch.stack([pred_entity_types[pred_rels[j]]
                                                     for j in range(pred_rels.shape[0])])

            # convert predicted relations for evaluation
            sample_gt_relations = self._convert_pred_relations(pred_rel_types, pred_rel_entity_spans,
                                                               pred_rel_entity_types, values)

            self._pred_relations.append(sample_gt_relations)

            # get entities that are not classified as 'None'
            valid_entity_types, valid_entity_bounds = self._get_entity_predictions(pred_entity_types,
                                                                                   batch.entity_spans[i])

            sample_pred_entities = self._convert_pred_entities(valid_entity_types, valid_entity_bounds)
            self._pred_entities.append(sample_pred_entities)

    def compute_scores(self):
        print("Evaluation")

        print("")
        print("--- Entities (NER) ---")
        print("")
        ner_final_eval = self._score_entities(self._gt_entities, self._pred_entities, print_results=True)

        print("")
        print("--- Relations ---")
        print("")
        print("Without NER")
        rel_final_eval = self._score_relations(self._gt_relations, self._pred_relations,
                                               include_ner=False, print_results=True)
        print("")
        print("With NER")
        rel_ner_final_eval = self._score_relations(self._gt_relations, self._pred_relations,
                                                   include_ner=True, print_results=True)

        return ner_final_eval, rel_final_eval, rel_ner_final_eval

    def store_examples(self):
        if jinja2 is None:
            warnings.warn('Examples cannot be stored since Jinja2 is not installed. ')
            return

        examples = []
        examples_ner = []

        for i, doc in enumerate(self._dataset.documents):
            example = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i], include_ner=False)
            examples.append(example)

            example_ner = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i],
                                                include_ner=True)
            examples_ner.append(example_ner)

        label, epoch = self._dataset_label, self._epoch
        self._store_examples(examples[:self._example_count],
                             file_path=self._examples_path % ('rel', label, epoch))

        self._store_examples(examples_ner[:self._example_count],
                             file_path=self._examples_path % ('rel_ner', label, epoch))

        self._store_examples(sorted(examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('rel_sorted', label, epoch))

        self._store_examples(sorted(examples_ner[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('rel_ner_sorted', label, epoch))

    def _get_entity_predictions(self, entity_types: torch.tensor, entity_spans: torch.tensor):
        indices = entity_types.nonzero().view(-1)
        valid_candidate_types = entity_types[indices]
        valid_spans = entity_spans[indices]

        return valid_candidate_types, valid_spans

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            gt_relations = doc.relations
            gt_entities = doc.entities

            # convert ground truth relations and entities for precision/recall/f1 evaluation
            sample_gt_relations = self._convert_gt_relations(gt_relations)
            sample_gt_entities = self._convert_gt_entities(gt_entities)

            self._gt_relations.append(sample_gt_relations)
            self._gt_entities.append(sample_gt_entities)

    def _convert_gt_entities(self, entities: List[Entity]):
        converted_entities = []

        for e in entities:
            converted_entity = e.as_tuple()
            converted_entities.append(converted_entity)

        return converted_entities

    def _convert_gt_relations(self, rels: List[Relation]):
        converted_rels = []

        for rel in rels:
            converted_rel = rel.as_tuple()
            converted_rels.append(converted_rel)

        return converted_rels

    def _convert_pred_relations(self, pred_rel_types: torch.tensor, pred_spans: torch.tensor,
                                pred_entities: torch.tensor, pred_scores: torch.tensor):
        converted_rels = []
        check = set()

        for i in range(pred_rel_types.shape[0]):
            label_idx = pred_rel_types[i].item()
            pred_rel_type = self._input_reader.get_relation_type(label_idx)
            pred_head_label_idx, pred_tail_label_idx = pred_entities[i][0].item(), pred_entities[i][1].item()
            pred_head_type = self._input_reader.get_entity_type(pred_head_label_idx)
            pred_tail_type = self._input_reader.get_entity_type(pred_tail_label_idx)
            score = pred_scores[i].item()

            spans = pred_spans[i]
            head_start, head_end = spans[0].tolist()
            tail_start, tail_end = spans[1].tolist()

            converted_rel = ((head_start, head_end, pred_head_type),
                             (tail_start, tail_end, pred_tail_type), pred_rel_type)
            converted_rel = self._adjust_rel(converted_rel)

            if converted_rel not in check:
                check.add(converted_rel)
                converted_rels.append(tuple(list(converted_rel) + [score]))

        return converted_rels

    def _convert_pred_entities(self, pred_types: torch.tensor, pred_spans: torch.tensor):
        converted_preds = []

        for i in range(pred_types.shape[0]):
            label_idx = pred_types[i].item()
            entity_type = self._input_reader.get_entity_type(label_idx)

            start, end = pred_spans[i].tolist()

            converted_pred = (start, end, entity_type)
            converted_preds.append(converted_pred)

        converted_preds = list(set(converted_preds))

        return converted_preds

    def _adjust_rel(self, rel: Tuple):
        adjusted_rel = rel
        if rel[-1].symmetric:
            head, tail = rel[:2]
            if tail[0] < head[0]:
                adjusted_rel = tail, head, rel[-1]

        return adjusted_rel

    def _score_entities(self, gt_entities: List[List[Tuple]], pred_entities: List[List[Tuple]],
                        print_results: bool = False):
        assert len(gt_entities) == len(pred_entities)

        gt_flat = []
        pred_flat = []
        entity_types = set()
        none_type = self._input_reader.get_entity_type(0)

        for sample_gt, sample_pred in zip(gt_entities, pred_entities):
            entities = set()
            entities.update(sample_gt)
            entities.update(sample_pred)

            for e in entities:
                if e in sample_gt:
                    entity_type = e[2]
                    gt_flat.append(entity_type)
                    entity_types.add(entity_type)
                else:
                    gt_flat.append(none_type)

                if e in sample_pred:
                    entity_type = e[2]
                    pred_flat.append(entity_type)
                    entity_types.add(entity_type)
                else:
                    pred_flat.append(none_type)

        metrics = self._compute_metrics(gt_flat, pred_flat, entity_types, print_results)
        return metrics

    def _score_relations(self, gt_rels: List[List[Tuple]], pred_rels: List[List[Tuple]],
                         include_ner: bool = False, print_results: bool = False):
        assert len(gt_rels) == len(pred_rels)

        def convert(r):
            if not include_ner:
                # remove entity types for evaluation
                return r[0][:2], r[1][:2], r[2]
            return r[:3]

        gt_flat = []
        pred_flat = []
        relation_types = set()
        none_type = self._input_reader.get_relation_type(0)

        for (sample_gt, sample_pred) in zip(gt_rels, pred_rels):
            sample_gt = [convert(s) for s in sample_gt]
            sample_pred = [convert(s) for s in sample_pred]

            relations = set()
            relations.update(sample_gt)
            relations.update(sample_pred)

            for r in relations:
                if r in sample_gt:
                    rel_type = r[2]
                    gt_flat.append(rel_type)
                    relation_types.add(rel_type)
                else:
                    gt_flat.append(none_type)

                if r in sample_pred:
                    rel_type = r[2]
                    pred_flat.append(rel_type)
                    relation_types.add(rel_type)
                else:
                    pred_flat.append(none_type)

        metrics = self._compute_metrics(gt_flat, pred_flat, relation_types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t.index for t in types]
        gt_all = [gt.index for gt in gt_all]
        pred_all = [p.index for p in pred_all]

        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t.short_name))
            results.append('\n')

        results.append('\n')

        # micro
        results.append(row_fmt % self._get_row(micro, 'micro'))
        results.append('\n')

        # macro
        results.append(row_fmt % self._get_row(macro, 'macro'))

        results_str = ''.join(results)
        print(results_str)

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    def _convert_example(self, doc, gt_relations, pred_relations, include_ner=False):
        encoding = doc.encoding

        # get micro precision/recall/f1 scores
        if gt_relations or pred_relations:
            precision, recall, f1 = self._score_relations([gt_relations], [pred_relations],
                                                          include_ner=include_ner)[:3]
        else:
            # corner case: no GT relations and no predictions
            precision, recall, f1 = [100] * 3

        scores = [p[-1] for p in pred_relations]
        pred_relations = [p[:-1] for p in pred_relations]
        relations = set(gt_relations + pred_relations)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for r in relations:
            if r in gt_relations:
                if r in pred_relations:
                    score = scores[pred_relations.index(r)]
                    tp.append(self._to_html(encoding, r, score=score))
                else:
                    fn.append(self._to_html(encoding, r))
            else:
                score = scores[pred_relations.index(r)]
                fp.append(self._to_html(encoding, r, score=score))

        tp = sorted(tp, key=lambda p: p[-1], reverse=True)
        fp = sorted(fp, key=lambda p: p[-1], reverse=True)

        text = self._prettify(self._text_encoder.decode(encoding))
        return dict(text=text, tp=tp, fn=fn, fp=fp, precision=precision, recall=recall, f1=f1,
                    length=len(doc.tokens))

    def _to_html(self, encoding: List[int], relation: Tuple, score: float = -1):
        rel_label = relation[2].verbose_name
        head_label = relation[0][2].verbose_name
        tail_label = relation[1][2].verbose_name

        if relation[0][0] < relation[1][0]:
            e1, e2, reverse = relation[0], relation[1], False
        else:
            e1, e2, reverse = relation[1], relation[0], True

        segments = [encoding[:e1[0]], encoding[e1[0]:e1[1]], encoding[e1[1]:e2[0]],
                    encoding[e2[0]:e2[1]], encoding[e2[1]:]]

        head_start = ' <span class="head" style="display:inline-block; text-align:center">'
        tail_start = ' <span class="tail" style="display:inline-block; text-align:center">'
        first = head_start if not reverse else tail_start
        second = tail_start if not reverse else head_start
        first += ('<span style="display:block; font-size:12px;">%s</span>' %
                  (head_label if not reverse else tail_label))
        second += ('<span style="display:block; font-size:12px;">%s</span>' %
                   (tail_label if not reverse else head_label))

        ctx_before = self._text_encoder.decode(segments[0])
        e1 = self._text_encoder.decode(segments[1])
        ctx_between = self._text_encoder.decode(segments[2])
        e2 = self._text_encoder.decode(segments[3])
        ctx_after = self._text_encoder.decode(segments[4])

        html_text = (ctx_before + first + e1 + '</span> '
                     + ctx_between + second + e2 + '</span> ' + ctx_after)
        html_text = self._prettify(html_text)

        return html_text, rel_label, score

    def _prettify(self, text: str):
        text = text.replace('_start_', '').replace('_classify_', '').replace('<unk>', '').replace('⁇', '')
        text = text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        return text

    def _store_examples(self, examples: List[Dict], file_path: str):
        template_path = os.path.join(SCRIPT_PATH, 'templates', 'examples.html')

        # read template
        with open(os.path.join(SCRIPT_PATH, template_path)) as f:
            template = jinja2.Template(f.read())

        # write to disc
        template.stream(examples=examples).dump(file_path)
