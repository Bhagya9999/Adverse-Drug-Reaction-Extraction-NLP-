# * Working on CADEC v.2 dataset in the root folder.

import os, glob
import re
import numpy as np
import pandas as pd
import pickle
import nltk
nltk.set_proxy('http://dfwproxy.ent.covance.com:80/')

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

text_files_path = "CADEC.v2/cadec/text"
ann_files_path  = "CADEC.v2/cadec/original"
model_data_path = "CADEC.v2/cadec/train"

tag_words_idx = 4
tag_idx       = 1


def list_files(path):
    files = []
    for file in sorted(os.listdir(path)):
        files.append(os.path.join(path, file))
    return files

def get_files(text_files_path, ann_files_path):
    text_files = list_files(text_files_path)
    ann_files  = list_files(ann_files_path)
    return text_files, ann_files

text_files, ann_files = get_files(text_files_path, ann_files_path)

def parse_ann_ner(ann_lines):
    ner_list = []
    for line in ann_lines:
        subs_count    = line.count(';')
        words_line    = re.split(';|\s+',line)
        ner_tag       = words_line[tag_idx]
        begin_ner_tag = 'B-'+ ner_tag
        inter_ner_tag = 'I-'+ ner_tag
        if(subs_count > 0):
            wrd_idx = tag_words_idx + 2*subs_count
        else:
            wrd_idx = tag_words_idx
        ner_words = words_line[wrd_idx:]
        for w in ner_words:
            tag = begin_ner_tag if w == ner_words[0] else inter_ner_tag
            ner_list.append ([w, tag])
    # ner_list.append(list(w, begin_ner_tag) if w == ner_words[0] else list(w, inter_ner_tag) for w in ner_words)
    return ner_list

def check_ann_parts(line):
    x = 0
    init_split = line.count(';')
    str_line = ''
    for t in line.split(";")[:-1]:
        str_line += t
        if(t.rstrip()[-1].isdigit()):
            str_line += ';'
            x += 1
    str_line += line.split(';')[-1]
    words_line = re.split(';|\s+',str_line)

    return x, words_line
    
    


def parse_ann(ann_lines, text):
    ner_list = []
    for line in ann_lines:
        subs_count, words_line = check_ann_parts(line)
        ner_tag       = words_line[tag_idx]
        begin_ner_tag = 'B-'+ ner_tag
        inter_ner_tag = 'I-'+ ner_tag
        init_word     = True
        if(subs_count > 0):
            wrd_idx = tag_words_idx + 2*subs_count
            for j in range(subs_count+1):
                ner_list_line = []
                idx_start     = int(words_line[2*(j+1)])
                idx_end       = int(words_line[2*(j+1) + 1])
                sub_text      = text[idx_start: idx_end]
                words_sub_text = sub_text.split()
                words_sub_pos  = nltk.pos_tag(words_sub_text)
                for (w, pos) in words_sub_pos:
                    tag = begin_ner_tag if init_word else inter_ner_tag
                    ner_list_line.append([w, pos, tag])
                    if init_word: init_word = False
                ner_list.append([idx_start, idx_end ,ner_list_line])
        else:
            wrd_idx       = tag_words_idx
            ner_words     = nltk.pos_tag(words_line[wrd_idx:])
            ner_list_line = []
            for (w, pos) in ner_words:
                tag = begin_ner_tag if init_word else inter_ner_tag
                ner_list_line.append([w, pos, tag])
                if init_word: init_word = False
            ner_list.append([int(words_line[wrd_idx-2]), int(words_line[wrd_idx-1]) ,ner_list_line])
    return ner_list

def parse_txt(txt, ann_ner, file_name, ann_check=False):
    idx_start_txt = 0
    ner_tags = []
    o_tag = 'O'
    
    print(file_name)
    for sent in txt.split('\n'):
        if(len(sent.strip())>0):
            idx_end_sent = idx_start_txt + len(sent)
            ner_tags_sent = []
            if(ann_check):
                for ann in ann_ner:
                    if(ann[0]>=idx_start_txt and ann[1]<=idx_end_sent):
                        idx_start_ner = ann[0]
                        idx_end_ner   = ann[1]
                        curr_tags = ann[-1]
                        curr_txt = txt[idx_start_txt: idx_start_ner]
                        curr_txt_words = nltk.word_tokenize(curr_txt)
                        curr_txt_pos = nltk.pos_tag(curr_txt_words)
                        for (w, pos) in curr_txt_pos:
                            ner_tags_sent.append([w, pos, o_tag])
                        ner_tags_sent.extend(curr_tags)
                        idx_start_txt = idx_end_ner
            curr_txt = txt[idx_start_txt:idx_end_sent]
            curr_txt_pos = nltk.pos_tag(nltk.word_tokenize(curr_txt))
            for (w, pos) in curr_txt_pos:
                ner_tags_sent.append([w, pos, o_tag])
            idx_start_txt = idx_end_sent + 1
            ner_tags.extend(ner_tags_sent)
    return ner_tags

def gen_data(txt, ann):
    ner_data = []
    empty_ann_files = []
    empty_txt_files = []
    for i in range(len(ann)):
        with open(txt[i]) as f:
            text = f.read()
        ann_lines      = [line.rstrip('\n') for line in open(ann[i]) if line[0] != '#']
        if(len(ann_lines)>0 and len(text)>0):
            ann_ner_list   = np.array(parse_ann(ann_lines, text))
            ann_ner_sorted = ann_ner_list[ann_ner_list[:,0].argsort()]
            unique_keys, indices = np.unique(ann_ner_sorted[:,0], return_index=True)
            ann_ner_unique = ann_ner_sorted[indices]
            txt_ner_data = parse_txt(text, ann_ner_unique, txt[i], True)
            ner_data.extend(txt_ner_data)
        elif(len(text)>0):
            txt_ner_data = parse_txt(text, [], txt[i], False)
            ner_data.extend(txt_ner_data)
            empty_ann_files.append(ann[i])
        else:
            empty_txt_files.append(ann[i])
    return ner_data, empty_ann_files, empty_txt_files


processed_data,empty_ann_files,empty_txt_files = gen_data(text_files, ann_files)
