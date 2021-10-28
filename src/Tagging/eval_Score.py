# coding: utf-8
from conll_eval import *
from conll_eval_partial import *
from seqeval.metrics import classification_report
from copy import deepcopy
from collections import namedtuple
import logging
import os


def read_special_format(filename):
    data = []
    sents_all = []
    with open(filename, encoding='utf-8') as reader:
        i = -1
        for line in reader:

            i += 1
            if i % 5 == 0:
                sents = line.split(" ||| ")[0].split(" ")
                sents = [y.replace('``', '"') for y in sents]
                sents = [y.replace("''", '"') for y in sents]

                class_label = line.split(" ||| ")[1].strip()

            elif i % 5 == 3:
                labels = line.split(" ||| ")[0].split(" ")

            elif i % 5 == 4:
                data.append((sents, labels, class_label))
                sents_all.append(" ".join(sents))
            else:
                continue
    print("All sents", len(sents_all))
    print("All unique sents", len(set(sents_all)))
    return data

# Categorization of errors.


def fix(sent, index1, index2):
    sent = [y.lower() for y in sent]
    for i in range(len(index1)-1):

        if index1[i] == "B-bias" and sent[i] in ['a', 'an', 'the', ",", "and"]:
            index1[i] = "O"
            if index1[i+1] == "I-bias":
                index1[i+1] = "B-bias"
        elif index1[i] == "I-bias" and sent[i] in ['a', 'an', 'the', ",", "and"] and index1[i+1] == "O":
            index1[i] = "O"

        elif index1[i] == "O" and index1[i+1] == "I-bias":
            index1[i+1] = "B-bias"

    if index1[-1] != "O" and sent[-1] in ['a', 'an', 'the', ","]:
        index1[-1] = "O"

    tmp_index1 = index1[::]
    index1 = index2

    for i in range(len(index1)-1):
        if index1[i] == "B-bias" and sent[i] in ['a', 'an', 'the', ",", "and"]:
            index1[i] = "O"
            if index1[i+1] == "I-bias":
                index1[i+1] = "B-bias"
        elif index1[i] == "I-bias" and sent[i] in ['a', 'an', 'the', ",", "amd"] and index1[i+1] == "O":
            index1[i] = "O"

        elif index1[i] == "O" and index1[i+1] == "I-bias":
            index1[i+1] = "B-bias"

    if index1[-1] != "O" and sent[-1] in ['a', 'an', 'the', ","]:
        index1[-1] = "O"

    # fix broken IOB.
    for i in range(1, len(tmp_index1)):
        if tmp_index1[i] == "I-bias" and tmp_index1[i-1] == "O":
            tmp_index1[i] = "B-bias"
    for i in range(1, len(index1)):
        if index1[i] == "I-bias" and index1[i-1] == "O":
            index1[i] = "B-bias"
    return tmp_index1, index1


def fix_fine_grained(sent, index1, index2):
    sent = [y.lower() for y in sent]
    for i in range(len(index1)-1):
        cur_bias = index1[i]
        if cur_bias.startswith("B") and sent[i] in ['a', 'an', 'the', ",", "and"]:
            index1[i] = "O"
            if index1[i+1].startswith("I-"):
                index1[i+1] = "B-" + index1[i+1][2:]
        elif index1[i].startswith("I-") and sent[i] in ['a', 'an', 'the', ",", "and"] and index1[i+1] == "O":
            index1[i] = "O"

        elif index1[i] == "O" and index1[i+1].startswith("I-"):
            index1[i+1] = "B" + index1[i+1][1:]

    if index1[-1] != "O" and sent[-1] in ['a', 'an', 'the', ","]:
        index1[-1] = "O"

    tmp_index1 = index1[::]
    index1 = index2

    for i in range(len(index1)-1):
        cur_bias = index1[i]
        if cur_bias.startswith("B") and sent[i] in ['a', 'an', 'the', ",", "and"]:
            index1[i] = "O"
            if index1[i+1].startswith("I-"):
                index1[i+1] = "B-" + index1[i+1][2:]
        elif index1[i].startswith("I-") and sent[i] in ['a', 'an', 'the', ",", "and"] and index1[i+1] == "O":
            index1[i] = "O"

        elif index1[i] == "O" and index1[i+1].startswith("I-"):
            index1[i+1] = "B" + index1[i+1][1:]

    if index1[-1] != "O" and sent[-1] in ['a', 'an', 'the', ","]:
        index1[-1] = "O"

    # fix broken IOB.
    for i in range(1, len(tmp_index1)):
        if tmp_index1[i].startswith("I-") and tmp_index1[i-1] == "O":
            tmp_index1[i] = "B" + index1[i+1][1:]
    for i in range(1, len(index1)):
        if index1[i].startswith("I-") and index1[i-1] == "O":
            index1[i] = "B" + index1[i+1][1:]
    return tmp_index1, index1


'''
Code for NER tagging results evaluation.
Support evaluation on:
    # Entity type
    # Exact Match
    # Strict Match
    # Partial Match

A large part of the code is borrowed from https://github.com/davidsbatista/NER-Evaluation/
The introduction blog. http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/

Special Note:
 On partial match POSSIBLE(POS)=COR+INC+PAR+MIS=TP+FN may not always hold, as partial matching may include cases.
 ['B-bias
'''


Entity = namedtuple("Entity", "e_type start_offset end_offset")


class Evaluator():

    def __init__(self, true, pred, tags):
        """
        """

        if len(true) != len(pred):
            raise ValueError(
                "Number of predicted documents does not equal true")

        self.true = true
        self.pred = pred
        self.tags = tags

        # Setup dict into which metrics will be stored.

        self.metrics_results = {
            'correct': 0,
            'incorrect': 0,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'possible': 0,
            'actual': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
        }

        # Copy results dict to cover the four schemes.

        self.results = {
            'strict': deepcopy(self.metrics_results),
            'ent_type': deepcopy(self.metrics_results),
            'partial': deepcopy(self.metrics_results),
            'exact': deepcopy(self.metrics_results),
        }

        # Create an accumulator to store results

        self.evaluation_agg_entities_type = {
            e: deepcopy(self.results) for e in tags}

    def evaluate(self):

        logging.info(
            "Imported %s predictions for %s true examples",
            len(self.pred), len(self.true)
        )

        for true_ents, pred_ents in zip(self.true, self.pred):

            # Check that the length of the true and predicted examples are the
            # same. This must be checked here, because another error may not
            # be thrown if the lengths do not match.

            if len(true_ents) != len(pred_ents):
                raise ValueError(
                    "Prediction length does not match true example length")

            # Compute results for one message

            tmp_results, tmp_agg_results = compute_metrics(
                collect_named_entities(true_ents),
                collect_named_entities(pred_ents),
                self.tags
            )

            # Cycle through each result and accumulate

            for eval_schema in self.results:

                for metric in self.results[eval_schema]:

                    self.results[eval_schema][metric] += tmp_results[eval_schema][metric]

            # Calculate global precision and recall

            self.results = compute_precision_recall_wrapper(self.results)

            # Aggregate results by entity type

            for e_type in self.tags:

                for eval_schema in tmp_agg_results[e_type]:

                    for metric in tmp_agg_results[e_type][eval_schema]:

                        self.evaluation_agg_entities_type[e_type][eval_schema][
                            metric] += tmp_agg_results[e_type][eval_schema][metric]

                # Calculate precision recall at the individual entity level

                self.evaluation_agg_entities_type[e_type] = compute_precision_recall_wrapper(
                    self.evaluation_agg_entities_type[e_type])

        return self.results, self.evaluation_agg_entities_type


def collect_named_entities(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.
    :param tokens: a list of tags
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):

        if token_tag == 'O':
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(
                    Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == 'B'):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    # Yang: major change here to avoided treating ["B-bias", "I-bias"] as no entities.

    # if ent_type and start_offset and end_offset is None
    if ent_type is not None and (start_offset != None) and end_offset is None:
        #         print("END_CHUNK")
        named_entities.append(Entity(ent_type, start_offset, len(tokens)-1))
    return named_entities


def compute_metrics(true_named_entities, pred_named_entities, tags):

    eval_metrics = {'correct': 0, 'incorrect': 0, 'partial': 0,
                    'missed': 0, 'spurious': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    # overall results

    evaluation = {
        'strict': deepcopy(eval_metrics),
        'ent_type': deepcopy(eval_metrics),
        'partial': deepcopy(eval_metrics),
        'exact': deepcopy(eval_metrics)
    }

    # results by entity type

    evaluation_agg_entities_type = {e: deepcopy(evaluation) for e in tags}

    # keep track of entities that overlapped

    true_which_overlapped_with_pred = []

    # Subset into only the tags that we are interested in.
    # NOTE: we remove the tags we don't want from both the predicted and the
    # true entities. This covers the two cases where mismatches can occur:
    #
    # 1) Where the model predicts a tag that is not present in the true data
    # 2) Where there is a tag in the true data that the model is not capable of
    # predicting.

    true_named_entities = [
        ent for ent in true_named_entities if ent.e_type in tags]
    pred_named_entities = [
        ent for ent in pred_named_entities if ent.e_type in tags]

    true_named_entities_dict = {ent: False for ent in true_named_entities}
#     pred_named_entities_dict = {ent:False for ent in pred_named_entities}
    # go through each predicted named-entity

    for pred in pred_named_entities:
        found_overlap = False

        # Check each of the potential scenarios in turn. See
        # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
        # for scenario explanation.

        # Scenario I: Exact match between true and pred

        if pred in true_named_entities:
            true_which_overlapped_with_pred.append(pred)
            evaluation['strict']['correct'] += 1
            evaluation['ent_type']['correct'] += 1
            evaluation['exact']['correct'] += 1
            evaluation['partial']['correct'] += 1

            # for the agg. by e_type results
            evaluation_agg_entities_type[pred.e_type]['strict']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['ent_type']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['exact']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['partial']['correct'] += 1

        else:

            # check for overlaps with any of the true entities
            for true in true_named_entities:

                pred_range = list(
                    [y for y in range(pred.start_offset, pred.end_offset+1)])
                true_range = list(
                    [y for y in range(true.start_offset, true.end_offset+1)])

                # Scenario IV: Offsets match, but entity type is wrong

                if true.start_offset == pred.start_offset and pred.end_offset == true.end_offset and true.e_type != pred.e_type:

                    # overall results
                    evaluation['strict']['incorrect'] += 1
                    evaluation['ent_type']['incorrect'] += 1
                    evaluation['partial']['correct'] += 1
                    evaluation['exact']['correct'] += 1

                    # aggregated by entity type results
                    evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                    evaluation_agg_entities_type[true.e_type]['ent_type']['incorrect'] += 1
                    evaluation_agg_entities_type[true.e_type]['partial']['correct'] += 1
                    evaluation_agg_entities_type[true.e_type]['exact']['correct'] += 1

                    true_which_overlapped_with_pred.append(true)
                    found_overlap = True

                    break

                # check for an overlap i.e. not exact boundary match, with true entities

                elif find_overlap(true_range, pred_range):
                    if true_named_entities_dict[true] == True:
                        break
                    else:
                        true_named_entities_dict[true] = True
                        true_which_overlapped_with_pred.append(true)

                        # Scenario V: There is an overlap (but offsets do not match
                        # exactly), and the entity type is the same.
                        # 2.1 overlaps with the same entity type

                        if pred.e_type == true.e_type:
                            #                         if not found_overlap:
                            # overall results
                            evaluation['strict']['incorrect'] += 1
                            evaluation['ent_type']['correct'] += 1
                            evaluation['partial']['partial'] += 1
                            evaluation['exact']['incorrect'] += 1

                            # aggregated by entity type results
                            evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                            evaluation_agg_entities_type[true.e_type]['ent_type']['correct'] += 1
                            evaluation_agg_entities_type[true.e_type]['partial']['partial'] += 1
                            evaluation_agg_entities_type[true.e_type]['exact']['incorrect'] += 1

                            found_overlap = True

                            break

                        # Scenario VI: Entities overlap, but the entity type is
                        # different.

                        else:
                            # overall results
                            evaluation['strict']['incorrect'] += 1
                            evaluation['ent_type']['incorrect'] += 1
                            evaluation['partial']['partial'] += 1
                            evaluation['exact']['incorrect'] += 1

                            # aggregated by entity type results
                            # Results against the true entity

                            evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                            evaluation_agg_entities_type[true.e_type]['partial']['partial'] += 1
                            evaluation_agg_entities_type[true.e_type]['ent_type']['incorrect'] += 1
                            evaluation_agg_entities_type[true.e_type]['exact']['incorrect'] += 1

                            # Results against the predicted entity

                            # evaluation_agg_entities_type[pred.e_type]['strict']['spurious'] += 1

                            found_overlap = True

                            break

            # Scenario II: Entities are spurious (i.e., over-generated).

            if not found_overlap:

                # Overall results

                evaluation['strict']['spurious'] += 1
                evaluation['ent_type']['spurious'] += 1
                evaluation['partial']['spurious'] += 1
                evaluation['exact']['spurious'] += 1

                # Aggregated by entity type results

                # NOTE: when pred.e_type is not found in tags
                # or when it simply does not appear in the test set, then it is
                # spurious, but it is not clear where to assign it at the tag
                # level. In this case, it is applied to all target_tags
                # found in this example. This will mean that the sum of the
                # evaluation_agg_entities will not equal evaluation.

                for true in tags:

                    evaluation_agg_entities_type[true]['strict']['spurious'] += 1
                    evaluation_agg_entities_type[true]['ent_type']['spurious'] += 1
                    evaluation_agg_entities_type[true]['partial']['spurious'] += 1
                    evaluation_agg_entities_type[true]['exact']['spurious'] += 1

    # Scenario III: Entity was missed entirely.

    for true in true_named_entities:
        if true in true_which_overlapped_with_pred:
            continue
        else:
            # overall results
            evaluation['strict']['missed'] += 1
            evaluation['ent_type']['missed'] += 1
            evaluation['partial']['missed'] += 1
            evaluation['exact']['missed'] += 1

            # for the agg. by e_type
            evaluation_agg_entities_type[true.e_type]['strict']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['ent_type']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['partial']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['exact']['missed'] += 1

    # Compute 'possible', 'actual' according to SemEval-2013 Task 9.1 on the
    # overall results, and use these to calculate precision and recall.

    for eval_type in evaluation:
        evaluation[eval_type] = compute_actual_possible(evaluation[eval_type])

    # Compute 'possible', 'actual', and precision and recall on entity level
    # results. Start by cycling through the accumulated results.

    for entity_type, entity_level in evaluation_agg_entities_type.items():

        # Cycle through the evaluation types for each dict containing entity
        # level results.

        for eval_type in entity_level:

            evaluation_agg_entities_type[entity_type][eval_type] = compute_actual_possible(
                entity_level[eval_type]
            )

    return evaluation, evaluation_agg_entities_type


def find_overlap(true_range, pred_range):
    """Find the overlap between two ranges
    Find the overlap between two ranges. Return the overlapping values if
    present, else return an empty set().
    Examples:
    >>> find_overlap((1, 2), (2, 3))
    2
    >>> find_overlap((1, 2), (3, 4))
    set()
    """

    true_set = set(true_range)
    pred_set = set(pred_range)

    overlaps = true_set.intersection(pred_set)

    return overlaps


def compute_actual_possible(results):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with actual, possible populated.
    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    correct = results['correct']
    incorrect = results['incorrect']
    partial = results['partial']
    missed = results['missed']
    spurious = results['spurious']

    # Possible: number annotations in the gold-standard which contribute to the
    # final score

    possible = correct + incorrect + partial + missed

    # Actual: number of annotations produced by the NER system

    actual = correct + incorrect + partial + spurious

    results["actual"] = actual
    results["possible"] = possible

    return results


def compute_precision_recall(results, partial_or_type=False):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with precison and recall populated.
    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    actual = results["actual"]
    possible = results["possible"]
    partial = results['partial']
    correct = results['correct']

    if partial_or_type:

        precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall = (correct + 0.5 * partial) / possible if possible > 0 else 0

    else:
        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision != 0 and recall != 0) else 0
    results["precision"] = precision
    results["recall"] = recall
    results["f1"] = f1
    return results


def compute_precision_recall_wrapper(results):
    """
    Wraps the compute_precision_recall function and runs on a dict of results
    """

    results_a = {key: compute_precision_recall(value, True) for key, value in results.items() if
                 key in ['partial', 'ent_type']}
    results_b = {key: compute_precision_recall(value) for key, value in results.items() if
                 key in ['strict', 'exact']}

    results = {**results_a, **results_b}
    return results


def read_file(input_path):
    data = []
    sentence = []
    gold_label = []
    predict_label = []
    pos_label = []
    print(input_path, "##")
    for line in open(input_path, "r").readlines():
        if len(line.split()) == 2 or line.split(" ")[0].startswith("DOC") or line.split(" ")[0].startswith("-DOC") or line.strip() == "":
            if len(sentence) > 0:
                data.append((sentence, pos_label, gold_label, predict_label))
                sentence = []
                gold_label = []
                predict_label = []
                pos_label = []
            continue
        split = line.split(" ")

        sentence.append(split[0])
        gold_label.append(split[-2].strip())
        predict_label.append(split[-1][:-1])
        pos_label.append(split[1].strip())
    print("!!")
    if len(sentence) > 0:

        data.append((sentence, pos_label, gold_label, predict_label))
        sentence = []
        label = []
    return data


def eval_scores(test_path, raw_file_path=str(os.path.join("data/data_joint_model_0820", "tag_dev_0820.tsv"))):
    data = read_file(test_path)[:]
    dev = read_special_format(raw_file_path)
    test = read_special_format(raw_file_path)
    print(set([x[-1] for x in dev]))

    # onlyTest on positive sentence.
    if "dev" in test_path:
        indexes = [dev.index(y) for y in dev if y[-1] == "1"]
    else:
        #         indexes = []
        #         for i,y in enumerate(test):
        #             if len(set(y[1])) > 1:
        #                 indexes.append(i)
        indexes = [test.index(y) for y in dev if y[-1] == "1"]

    data = [y for index, y in enumerate(data) if index in indexes]
    print("HAVE DATA", len(data))

    data_all = [y[0] for y in data]
    y_true = [y[-2] for y in data]
    y_pred = [y[-1] for y in data]
    print(len(data_all))
#     print(y_true)
#     print(y_pred)
    post_change_y_true = []
    post_change_y_pred = []
    x = y_true[::]
    report = classification_report(y_true, y_pred, digits=5)

    for i in range(len(data_all)):

        post_change_y_true_instance, post_change_y_pred_instance = fix(
            data_all[i], y_true[i], y_pred[i])
        post_change_y_true.append(post_change_y_true_instance)
        post_change_y_pred.append(post_change_y_pred_instance)

    print(report)

    # write the file.
    import os
    with open(test_path + "conll_eval", "w") as fin:
        for i in range(len(data_all)):
            for index in range(len(data_all[i])):
                fin.write(data_all[i][index] + " " + post_change_y_true[i]
                          [index] + " " + post_change_y_pred[i][index] + '\n')
            fin.write("\n")

    # evaluate.
    exact_match = evaluate_conll_file(inputFile=test_path + "conll_eval")
    partial_match = evaluate_conll_file_partial(
        inputFile=test_path + "conll_eval")

    os.remove(test_path + "conll_eval")
    return exact_match, partial_match


#     metrics_results = {'correct': 0, 'incorrect': 0, 'partial': 0,
#                        'missed': 0, 'spurious': 0, 'possible': 0, 'actual': 0, 'precision': 0, 'recall': 0, 'f1':0}

#     # overall results
#     results = {'strict': deepcopy(metrics_results),
#                'ent_type': deepcopy(metrics_results),
#                'partial':deepcopy(metrics_results),
#                'exact':deepcopy(metrics_results)
#               }


#     # results aggregated by entity type
# #     print('start')
#     evaluation_agg_entities_type = {e: deepcopy(results) for e in ['bias']}

#     for true_ents, pred_ents in zip(post_change_y_true, post_change_y_pred):

#         # compute results for one message
#         tmp_results, tmp_agg_results = compute_metrics(
#             collect_named_entities(true_ents), collect_named_entities(pred_ents),  ['bias']
#         )

#         # aggregate overall results
#         for eval_schema in results.keys():
#             for metric in metrics_results.keys():

#                 results[eval_schema][metric] += tmp_results[eval_schema][metric]

#         # Calculate global precision and recall

#         results = compute_precision_recall_wrapper(results)


#         # aggregate results by entity type

#         for e_type in ['bias']:

#             for eval_schema in tmp_agg_results[e_type]:

#                 for metric in tmp_agg_results[e_type][eval_schema]:

#                     evaluation_agg_entities_type[e_type][eval_schema][metric] += tmp_agg_results[e_type][eval_schema][metric]

#          # Calculate precision recall at the individual entity level

#             evaluation_agg_entities_type[e_type] = compute_precision_recall_wrapper(evaluation_agg_entities_type[e_type])


#     for e_type in evaluation_agg_entities_type:
#         for category in evaluation_agg_entities_type[e_type]:
#             data = evaluation_agg_entities_type[e_type][category]
# #             print( "%s & %.2f & %.2f  & %.2f "%(category, data['precision']*100, data['recall']*100, data['f1']*100))

#     exact_tuple = (evaluation_agg_entities_type["bias"]['exact']['precision'] * 100,
#                    evaluation_agg_entities_type["bias"]['exact']['recall'] * 100,
#                    evaluation_agg_entities_type["bias"]['exact']['f1'] * 100)

#     partial_tuple = (evaluation_agg_entities_type["bias"]['partial']['precision'] * 100,
#                    evaluation_agg_entities_type["bias"]['partial']['recall'] * 100,
#                    evaluation_agg_entities_type["bias"]['partial']['f1'] * 100)

    return exact_tuple, partial_tuple


def eval_scores_fine_grained(test_path, raw_file_path=str(os.path.join("data/data_joint_model_0820", "tag_dev_0820.tsv"))):
    data = read_file(test_path)[:]
    dev = read_special_format(raw_file_path)
    test = read_special_format(raw_file_path)
    print(set([x[-1] for x in dev]))

    # onlyTest on positive sentence.
    if "dev" in test_path:
        indexes = [dev.index(y) for y in dev if y[-1] == "1"]
    elif "test" in test_path:
        indexes = [test.index(y) for y in test if y[-1] == "1"]

    print(data[0])
    print(len(indexes))

    data = [y for index, y in enumerate(data) if index in indexes]
    print("HAVE DATA", len(data))

    data_all = [y[0] for y in data]
    y_true = [y[-2] for y in data]
    y_pred = [y[-1] for y in data]
    print(len(data_all))
#     print(y_true)
#     print(y_pred)
    post_change_y_true = []
    post_change_y_pred = []
    x = y_true[::]
    report = classification_report(y_true, y_pred, digits=5)

    for i in range(len(data_all)):

        post_change_y_true_instance, post_change_y_pred_instance = fix_fine_grained(
            data_all[i], y_true[i], y_pred[i])
        post_change_y_true.append(post_change_y_true_instance)
        post_change_y_pred.append(post_change_y_pred_instance)

    print(post_change_y_pred)

    print(report)

    metrics_results = {'correct': 0, 'incorrect': 0, 'partial': 0,
                       'missed': 0, 'spurious': 0, 'possible': 0, 'actual': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    # overall results
    results = {'strict': deepcopy(metrics_results),
               'ent_type': deepcopy(metrics_results),
               'partial': deepcopy(metrics_results),
               'exact': deepcopy(metrics_results)
               }

    # results aggregated by entity type
#     print('start')
    evaluation_agg_entities_type = {e: deepcopy(
        results) for e in ['epistemological_bias', 'frame_bias', 'demographic_bias']}

    for true_ents, pred_ents in zip(post_change_y_true, post_change_y_pred):

        # compute results for one message
        tmp_results, tmp_agg_results = compute_metrics(
            collect_named_entities(true_ents), collect_named_entities(pred_ents), [
                'epistemological_bias', 'frame_bias', 'demographic_bias']
        )

        # aggregate overall results
        for eval_schema in results.keys():
            for metric in metrics_results.keys():

                results[eval_schema][metric] += tmp_results[eval_schema][metric]

        # Calculate global precision and recall

        results = compute_precision_recall_wrapper(results)

        # aggregate results by entity type

        for e_type in ['epistemological_bias', 'frame_bias', 'demographic_bias']:

            for eval_schema in tmp_agg_results[e_type]:

                for metric in tmp_agg_results[e_type][eval_schema]:

                    evaluation_agg_entities_type[e_type][eval_schema][metric] += tmp_agg_results[e_type][eval_schema][metric]

         # Calculate precision recall at the individual entity level

            evaluation_agg_entities_type[e_type] = compute_precision_recall_wrapper(
                evaluation_agg_entities_type[e_type])

    for e_type in evaluation_agg_entities_type:
        for category in evaluation_agg_entities_type[e_type]:
            data = evaluation_agg_entities_type[e_type][category]
#             print( "%s & %.2f & %.2f  & %.2f "%(category, data['precision']*100, data['recall']*100, data['f1']*100))
    for etype in evaluation_agg_entities_type:
        exact_tuple = (evaluation_agg_entities_type[etype]['exact']['precision'] * 100,
                       evaluation_agg_entities_type[etype]['exact']['recall'] * 100,
                       evaluation_agg_entities_type[etype]['exact']['f1'] * 100)

        partial_tuple = (evaluation_agg_entities_type[etype]['partial']['precision'] * 100,
                         evaluation_agg_entities_type[etype]['partial']['recall'] * 100,
                         evaluation_agg_entities_type[etype]['partial']['f1'] * 100)
        print(etype, exact_tuple, partial_tuple)
    print(evaluation_agg_entities_type)
    return exact_tuple, partial_tuple
