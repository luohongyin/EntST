import enum
import sys
import csv
import copy
import random

import json
import pandas as pd

import torch

from synthesis import synthesis_func

domain_dict = {
    'sst2': 'SST-2',
    'qqp': 'QQP',
    'mnli': 'MNLI',
    'qnli': 'QNLI',
    'rte': 'RTE',
    'cola': 'CoLA',
    'mrpc': 'MRPC'
}

ds_dict = {
    'sst2': {
        'sentence1': 'sentence',
        'sentence2': None,
        'sentence1_eval': 'sentence',
        'sentence2_eval': None,
        'label': 'label',
        'label_dict': {
            0: 0, 1: 1
        },
        'label_explain': {
            0: 'negative', 1: 'positive'
        }
    },
    'qqp': {
        'sentence1': 'question1',
        'sentence2': 'question2',
        'sentence1_eval': 'question1',
        'sentence2_eval': 'question2',
        'label': 'is_duplicate',
        'label_dict': {
            0: 0, 1: 1
        },
        'label_explain': {
            0: 'contradiction', 1: 'entailment'
        }
    },
    'mnli': {
        'sentence1': 'sentence2',
        'sentence2': 'sentence1',
        'sentence1_eval': 'hypothesis',
        'sentence2_eval': 'premise',
        # 'sentence2_eval': 'hypothesis',
        'label': 'gold_label',
        'label_dict': {
            'entailment': 0,
            'neutral': 1,
            'contradiction': 2
        },
        'label_explain': {
            0: 'entailment', 1: 'neutral', 2: 'contradiction'
        }
    },
    'qnli': {
        'sentence1': 'question',
        'sentence2': 'sentence',
        'sentence1_eval': 'question',
        'sentence2_eval': 'sentence',
        'label': 'label',
        'label_dict': {
            'entailment': 0,
            'not_entailment': 1
        },
        'label_explain': {
            0: 'entailment', 1: 'contradiction'
        }
    },
    'rte': {
        'sentence1': 'sentence2',
        'sentence2': 'sentence1',
        'sentence1_eval': 'sentence2',
        'sentence2_eval': 'sentence1',
        'label': 'label',
        'label_dict': {
            'entailment': 0,
            'not_entailment': 1
        },
        'label_explain': {
            0: 'entailment', 1: 'contradiction'
        }
    },
    'cola': {}
}

def coordinate(domain, mlm=True):

    if domain == 'sst2':
        cord_list = [
            'It is [MASK] that it is a good movie and I like the movie.',
            'It is [MASK] that it is a bad movie and I hate the movie.',
            'It is [MASK] that it is a good movie and I hate the movie.',
            'It is [MASK] that it is a bad movie and I like the movie.',
        ]
    elif domain == 'qnli':
        cord_list = [
            'It is [MASK] that the answer to question that can be answered by context is in the context.',
            'It is [MASK] that the answer to question that cannot be answered by context is not in context.',
            'It is [MASK] that the answer to question that cannot be answered by context is in the context.',
            'It is [MASK] that the answer to question that can be answered by context is not in context.',

            'It is [MASK] that the context that contains the answer to question can answer question.',
            'It is [MASK] that the context that does not contain the answer to question cannot answer question.',
            'It is [MASK] that the context that does not contain the answer to question can answer question.',
            'It is [MASK] that the context that contains the answer to question cannot answer question.',
        ]
    elif domain == 'qqp':
        cord_list = [
            'It is [MASK] that the questions are duplicated and they have no differnt answer.',
            'It is [MASK] that the questions are not duplicated and they have different answers.',
            'It is [MASK] that the questions are duplicated and they have different answers.',
            'It is [MASK] that the questions are not duplicated and they no different answers.'
        ]
    elif domain == 'rte':
        cord_list = [
            'It is [MASK] that the hypothesis is entailed by the premise and they can be both correct.',
            'It is [MASK] that the hypothesis contradicts the premise and they cannot be both correct.',
            'It is [MASK] that the hypothesis is entailed by the premise and they cannot be both correct.',
            'It is [MASK] that the hypothesis contradicts the premise and they can be both correct.'
        ]
    else:
        print(f'\nDomain {domain} is not supported in build_prompt_input()\n')
        sys.exit()
    if not mlm:
        prefix_len = len('It is [MASK] that ')
        cord_list = [x[prefix_len:] for x in cord_list]
    return cord_list, [1, 1, 0, 0, 1, 1, 0, 0]

def build_prompt_input(domain, sent1_list, sent2_list, mlm=True):

    def prompt_1(domain, sent1, sent2):
        if domain == 'sst2':
            # return f'It is [MASK] that I like the movie is true when {sent1}.'
            return f'It is [MASK] that the movie is good is entailed by {sent1}.'
        elif domain == 'qnli':
            return f'It is [MASK] that the answer to {sent1} is entailed by {sent2}.'
        elif domain == 'qqp':
            return f'It is [MASK] that the answer to {sent1} is entailed by {sent2}.'
        elif domain == 'rte':
            return f'It is [MASK] that {sent1} is entailed by {sent2}.'
        elif domain == 'mnli':
            return f'It is [MASK] that {sent1} is entailed by {sent2}.'
        elif domain == 'cola':
            return f'It is [MASK] that the sentence {sent1} is fluent.'
        else:
            print(f'\nDomain {domain} is not supported in build_prompt_input()\n')
            sys.exit()

    def prompt_2(domain, sent1, sent2):
        if domain == 'sst2':
            # return f'It is [MASK] that the movie is good cannot be entailed by {sent1}.'
            return f'It is [MASK] that I like the movie cannot be entailed by the comment {sent1}.'
        elif domain == 'qnli':
            return f'It is [MASK] that {sent1} cannot be answered by {sent2}.'
        elif domain == 'qqp':
            return f'It is [MASK] that {sent1} and {sent2} cannot be the same questions.'
        elif domain == 'rte':
            return f'It is [MASK] that {sent1} cannot be true when {sent2} is true.'
        elif domain == 'mnli':
            return f'It is [MASK] that {sent1} cannot be true when {sent2} is true.'
        elif domain == 'cola':
            return f'It is [MASK] that the grammar of {sent1} cannot be accept.'
        else:
            print(f'\nDomain {domain} is not supported in build_prompt_input()\n')
            sys.exit()

    label_rvs_map = {
        'sst2': (1, 0), 'qnli': (0, 1), 'qqp': (1, 0),
        'rte': (0, 1), 'mnli': (0, 1), 'cola': (1, 0)
    }

    prompt_list_1 = [prompt_1(domain, x, y) for x, y in zip(sent1_list, sent2_list)]
    prompt_list_2 = [prompt_2(domain, x, y) for x, y in zip(sent1_list, sent2_list)]
    # prompt_list_3 = [prompt_3(domain, x, y) for x, y in zip(sent1_list, sent2_list)]
    # prompt_list_4 = [prompt_4(domain, x, y) for x, y in zip(sent1_list, sent2_list)]

    prompt_list = prompt_list_1 + prompt_list_2 # + prompt_list_3 + prompt_list_4
    if not mlm:
        prefix_len = len('It is [MASK] that ')
        prompt_list = [x[prefix_len:] for x in prompt_list]
    return prompt_list, label_rvs_map[domain]


def shuffle_data(sent1_list, sent2_list, label_list):
    if sent2_list is None:
        data = list(zip(sent1_list, label_list))
    else:
        data = list(zip(sent1_list, sent2_list, label_list))

    random.shuffle(data)
    sent1_list = [x[0] for x in data]
    label_list = [x[-1] for x in data]

    if sent2_list is not None:
        sent2_list = [x[1] for x in data]
    return {
        'sent1_list': sent1_list,
        'sent2_list': sent2_list,
        'label_list': label_list
    }


def contrast_shuffle_data(sent1_list, sent2_list, label_list):
    if domain == 'sst2':
        batch_size = 32
    elif domain == 'mnli':
        batch_size = 12
    elif domain == 'rte':
        batch_size = 16
    elif domain == 'qqp':
        batch_size = 16
    elif domain == 'qnli':
        batch_size = 8
    else:
        batch_size = 32
    data_size = len(sent1_list) // 2
    prompt_batch_size = batch_size // 2
    
    if sent2_list is None:
        data = list(zip(sent1_list, label_list))
    else:
        data = list(zip(sent1_list, sent2_list, label_list))
    
    new_data_list = []
    for i in range(0, data_size, batch_size):
        p1_batch = data[i: i + prompt_batch_size]
        p2_batch = data[i + data_size: i + data_size + prompt_batch_size]
        new_data_list += p1_batch + p2_batch
    
    data = new_data_list

    sent1_list = [x[0] for x in data]
    label_list = [x[-1] for x in data]

    if sent2_list is not None:
        sent2_list = [x[1] for x in data]
    return {
        'sent1_list': sent1_list,
        'sent2_list': sent2_list,
        'label_list': label_list
    }


def build_ft_data(
        rvs_map, num_prompt_type, pseudo_label_list,
        label_list, prompt_list, ft_mode, train_mode, train_size
    ):
    ft_labels = []
    
    if ft_mode == 'st':
        if rvs_map[1] != rvs_map[0]:
            if num_prompt_type >= 1:
                ft_labels = pseudo_label_list + [1 - x for x in pseudo_label_list]
            else:
                ft_labels = [1 - x for x in pseudo_label_list] + pseudo_label_list
        else:
            ft_labels = pseudo_label_list + pseudo_label_list
    
    elif ft_mode == 'ft':
        if rvs_map[0] == 0:
            ft_labels = label_list # + [1 - x for x in label_list]
        else:
            ft_labels = [1 - x for x in label_list] # + label_list
        if rvs_map[1] == rvs_map[0]:
            ft_labels = ft_labels + ft_labels
        else:
            ft_labels = ft_labels + [1 - x for x in ft_labels]
    
    else:
        print(f'\nFT_MODE = {ft_mode} not supported.\n')
        abort()
    
    mask_label = (torch.rand(train_size) > 0.5).long().tolist()
    new_prompt_list = ['' for i in range(train_size)]
    new_ft_labels = [0 for i in range(train_size)]
    
    p1_data = {
        'sent1_list': prompt_list[:train_size],
        'sent2_list': None,
        'label_list': ft_labels[:train_size]
    }
    
    p2_data = {
        'sent1_list': prompt_list[train_size:],
        'sent2_list': None,
        'label_list': ft_labels[train_size:]
    }
    
    pj_data = {
        'sent1_list': prompt_list[:],
        'sent2_list': None,
        'label_list': ft_labels[:]
    }

    '''for i, l in enumerate(mask_label):
        if l == 0:
            new_prompt_list[i] = p1_data['sent1_list'][i]
            new_ft_labels[i] = p1_data['label_list'][i]
        else:
            new_prompt_list[i] = p2_data['sent1_list'][i]
            new_ft_labels[i] = p2_data['label_list'][i]
    
    pj_data = {
        'sent1_list': new_prompt_list,
        'sent2_list': None,
        'label_list': new_ft_labels
    }'''

    if train_mode == 'prompt_1':
        new_data = p1_data
    
    if train_mode == 'prompt_2':
        new_data = p2_data
    
    if train_mode == 'prompt_joint':
        new_data = pj_data
    
    # for x in new_data['sent1_list']:
    #     print(x)
    # abort()

    # '''
    new_data = shuffle_data(
        new_data['sent1_list'], new_data['sent2_list'], new_data['label_list']
    )
    # '''
    '''
    new_data = contrast_shuffle_data(
        new_data['sent1_list'], new_data['sent2_list'], new_data['label_list']
    )
    # '''
    
    return new_data


def pd_load_data(fn, dformat):
    df = pd.read_csv(fn, sep = '\t')
    s1_title = dformat['sentence1']
    s2_title = dformat['sentence2']
    label_title = dformat['label']

    sent1_list = list(df[s1_title])
    sent2_list = None

    print(f'Num. of sent1_list = {len(sent1_list)}')

    if s2_title:
        sent2_list = list(df[s2_title])
    else:
        sent2_list = None

    label_list = []
    outlier = 0

    for x in df[label_title]:
        try:
            label_list.append(dformat['label_dict'][x])
        except:
            outlier += 1
            label_list.append(1)

    print(f'Num. of outlier = {outlier}')

    return sent1_list, sent2_list, label_list


def csv_load_data(fn, dformat):
    tsv_lines = open(fn, encoding='utf-8').readlines()

    sent1_list = []
    sent2_list = []
    label_list = []

    num_fields = 0

    s1_title = dformat['sentence1']
    s2_title = dformat['sentence2']
    label_title = dformat['label']

    for row_id, row in enumerate(tsv_lines):
        if row_id == 0:
            fields = row.strip('\n').split('\t')
            num_fields = len(fields)
            field_dict = {t: i for i, t in enumerate(fields)}
        row = row.strip('\n').split('\t')

        if len(row) > num_fields:
            continue

        if row_id > 0:
            sent1_list.append(row[field_dict[s1_title]])
            sent2_list.append(row[field_dict[s2_title]])
            label_list.append(dformat['label_dict'][row[field_dict[label_title]]])

    print(f'Num. of sent1_list = {len(sent1_list)}')
    return sent1_list, sent2_list, label_list


def adv_load_data(dev_data, dformat):
    s1_title = dformat['sentence1_eval']
    s2_title = dformat['sentence2_eval']

    sent1_list = [x[s1_title] for x in dev_data]
    sent2_list = None
    if s2_title:
        sent2_list = [x[s2_title] for x in dev_data]

    label_list = [x['label'] for x in dev_data]
    return sent1_list, sent2_list, label_list


def enumerate_data(data_triple, sent2_none=False):
    sent1_list = [x[0] for x in data_triple]
    label_list = [x[-1] for x in data_triple]
    if sent2_none:
        sent2_list = None
    else:
        sent2_list = [x[1] for x in data_triple]
    return sent1_list, sent2_list, label_list


def dump_split(data_triple, dformat, domain, split_size=16, split_limit=100, sent2_none=True):
    split_id = 0

    for i in range(0, len(data_triple), split_size):
        split = data_triple[i: i + split_size]
        sent1_list, sent2_list, label_list = enumerate_data(split, sent2_none)

        split_dict = {
            'sent1_list': sent1_list,
            'sent2_list': sent2_list,
            'label_list': label_list,
            'dformat': dformat
        }

        json.dump(split_dict, open(
            f'data/glue_data/{domain}/splits/syn_sent2_proc_{split_id}.json', 'w'
        ))
        split_id += 1

        if split_id > split_limit:
            break

    return split_id


def load_cola(file_name):

    def proc_line(line):
        line_list = line.split('\t')
        label = line_list[1]
        sent1 = line_list[-1].strip('\n')
        return [sent1, int(label)]
    
    in_list = open(file_name, encoding='utf8')
    proc_list = [proc_line(x) for x in in_list]
    sent1_list = [x[0] for x in proc_list]
    label_list = [x[1] for x in proc_list]
    return sent1_list, None, label_list


if __name__ == '__main__':
    domain = sys.argv[1]
    split = sys.argv[2]

    if len(sys.argv) == 3:
        fs_rate = None
    else:
        fs_rate = int(sys.argv[3])
    
    exp_id = int(sys.argv[4])

    train_domain_name = domain_dict[domain]
    split_size = 32

    if split != 'adv':
        train_fn = f'data/glue_data/{train_domain_name}/{split}.tsv'
        if domain == 'cola':
            sent1_list, sent2_list, label_list = load_cola(train_fn)
        elif 'nli' not in domain:
            sent1_list, sent2_list, label_list = pd_load_data(
                train_fn, ds_dict[domain]
            )
        else:
            sent1_list, sent2_list, label_list = csv_load_data(
                train_fn, ds_dict[domain]
            )
    else:
        dev_adv = json.load(open('data/adv_dev/dev.json'))[domain]
        sent1_list, sent2_list, label_list = adv_load_data(dev_adv, ds_dict[domain])

    if split == 'train':
        # print(sent1_list[0])
        # print(sent2_list[0])

        syn_sent2 = {
            'sent1_list': sent1_list,
            'sent2_list': sent2_list,
            'label_list': label_list,
            'dformat': ds_dict[domain]
        }

        json.dump(syn_sent2, open(
            f'data/glue_data/{train_domain_name}/syn_sent2_proc.json', 'w'
        ))

    dformat = ds_dict[domain]

    if fs_rate is not None:
        num_case = len(sent1_list)
        num_sample = fs_rate * (exp_id + 1)

        if sent2_list is None:
            data_triple = list(zip(sent1_list, label_list))
            num_split = dump_split(
                data_triple, ds_dict[domain], train_domain_name,
                split_size = split_size, split_limit = 100, sent2_none = True
            )

            data_triple = random.sample(data_triple, num_sample)
            sent1_list = [x[0] for x in data_triple]
            label_list = [x[1] for x in data_triple]

        else:
            data_triple = list(zip(sent1_list, sent2_list, label_list))
            num_split = dump_split(
                data_triple, ds_dict[domain], train_domain_name,
                split_size = split_size, split_limit = 100, sent2_none = False
            )

            data_triple = random.sample(data_triple, num_sample)
            sent1_list = [x[0] for x in data_triple]
            sent2_list = [x[1] for x in data_triple]
            label_list = [x[2] for x in data_triple]

        dformat['num_split'] = num_split

    # print(sent2_list[0])
    dataset = {
        'sent1_list': sent1_list,
        'sent2_list': sent2_list,
        'label_list': label_list,
        'dformat': ds_dict[domain]
    }

    if fs_rate is not None:
        json.dump(dataset, open(
            f'data/glue_data/{train_domain_name}/{split}_proc_{exp_id}.json', 'w'
        ))
    else:
        json.dump(dataset, open(
            f'data/glue_data/{train_domain_name}/{split}_proc.json', 'w'
        ))

    json.dump(dformat, open(
        f'data/glue_data/{train_domain_name}/dformat.json', 'w'
    ))
