from eval_Score import *


def readfile(filename):
    '''
    read file
    '''
    f = open(filename, encoding="utf-8")
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue

        splits = line.split(' ')
        if splits[0] in ['``', "''"]:
            splits[0] = '"'
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    print(len(data))
    return data


def eval_score_(test_path, raw_file_path=str(os.path.join("../../data/data_joint_model_0820", "tag_dev_0820.tsv"))):

    dev = read_special_format(raw_file_path)
    indexes = [dev.index(y) for y in dev if y[-1] == "1"]

    in_file = readfile(test_path)
    data = [y for index, y in enumerate(in_file) if index in indexes]

    source_data = [y for index, y in enumerate(dev) if index in indexes]

    data_all = [y[0] for y in source_data]
    y_pred = [y[-2] for y in data]
    y_true = [y[-1] for y in data]

#     print(y_true)
#     print(y_pred)
    post_change_y_true = []
    post_change_y_pred = []
    x = y_true[::]

    report = classification_report(y_true, y_pred, digits=5)
    # post_change_y_true, post_change_y_pred = y_true, y_pred

    for i in range(len(data_all)):

        if len(data_all[i]) != len(y_true):
            print(i)
            print(len(data_all[i]))
            print(len(y_true[i]))
        post_change_y_true_instance, post_change_y_pred_instance = fix(
            data_all[i], y_true[i], y_pred[i])
        post_change_y_true.append(post_change_y_true_instance)
        post_change_y_pred.append(post_change_y_pred_instance)

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
    evaluation_agg_entities_type = {e: deepcopy(results) for e in ['bias']}

    for true_ents, pred_ents in zip(post_change_y_true, post_change_y_pred):

        # compute results for one message
        tmp_results, tmp_agg_results = compute_metrics(
            collect_named_entities(true_ents), collect_named_entities(
                pred_ents),  ['bias']
        )

        # aggregate overall results
        for eval_schema in results.keys():
            for metric in metrics_results.keys():

                results[eval_schema][metric] += tmp_results[eval_schema][metric]

        # Calculate global precision and recall

        results = compute_precision_recall_wrapper(results)

        # aggregate results by entity type

        for e_type in ['bias']:

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

    exact_tuple = (evaluation_agg_entities_type["bias"]['exact']['precision'] * 100,
                   evaluation_agg_entities_type["bias"]['exact']['recall'] * 100,
                   evaluation_agg_entities_type["bias"]['exact']['f1'] * 100)

    partial_tuple = (evaluation_agg_entities_type["bias"]['partial']['precision'] * 100,
                     evaluation_agg_entities_type["bias"]['partial']['recall'] * 100,
                     evaluation_agg_entities_type["bias"]['partial']['f1'] * 100)

    return exact_tuple, partial_tuple


a, b = eval_score_('dev_top1.txt')
print(a)
print(b)
