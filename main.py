import random
import os
import pickle
import math
import re
from collections import defaultdict

DIR_BROWN_CORPUS = 'brown'
DIR_BROWN_TRAIN_MODEL = 'brown_train'
DIR_REFORMAT_BROWN_CORPUS = 'brown_reformat'

REFORMAT_BROWN_CORPUS = 'brown'
REFORMAT_BROWN_CORPUS_TEST = 'brown_test'

DIR_REUTERS = 'reuters'
DIR_REUTERS_TRAIN = 'reuters/training'
DIR_REUTERS_TEST = 'reuters/test'
DIR_REUTERS_TRAIN_MODEL = 'reuters_train'
DIR_REFORMAT_REUTERS = 'reuters_reformat'

REFORMAT_REUTERS = 'reuters'
REFORMAT_REUTERS_TEST = 'reuters_test'

UNIGRAM = 'unigram'
BIGRAM = 'bigram'
TRIGRAM = 'trigram'
A_BIGRAM = 'A_bigram'
A_TRIGRAM = 'A_trigram'

unigram = defaultdict(int)
bigram = defaultdict(int)
trigram = defaultdict(int)

A_bigram = defaultdict(list)
A_trigram = defaultdict(list)

COUNT = 0
DISCOUNT_VALUE = 0.5
NUM_OF_TRAIN_FILES = 490


def format_brown_corpus():
    list_of_filename = os.listdir(DIR_BROWN_CORPUS)
    os.makedirs(DIR_REFORMAT_BROWN_CORPUS)

    content = ''
    for file in list_of_filename[:NUM_OF_TRAIN_FILES]:
        if len(file) == 4:
            with open(DIR_BROWN_CORPUS + '/' + file, 'r') as file_content:
                lines = file_content.readlines()
                for line in lines:
                    if line and line.strip():
                        for word_tag in line.split():
                            current_word = word_tag.split('/', 1)[0]
                            content += current_word + ' '
                        content += '\n'
            file_content.close()

    file_path = DIR_REFORMAT_BROWN_CORPUS + '/' + REFORMAT_BROWN_CORPUS
    with open(file_path, 'wb') as file:
        file.write(str(content))
    file.close()

    content = ''
    for file in list_of_filename[NUM_OF_TRAIN_FILES:]:
        if len(file) == 4:
            with open(DIR_BROWN_CORPUS + '/' + file, 'r') as file_content:
                lines = file_content.readlines()
                for line in lines:
                    if line and line.strip():
                        for word_tag in line.split():
                            current_word = word_tag.split('/', 1)[0]
                            content += current_word + ' '
                        content += '\n'
            file_content.close()

    file_path = DIR_REFORMAT_BROWN_CORPUS + '/' + REFORMAT_BROWN_CORPUS_TEST
    with open(file_path, 'wb') as file:
        file.write(str(content))
    file.close()


def format_reuters():
    list_of_filename = os.listdir(DIR_REUTERS_TRAIN)
    os.makedirs(DIR_REFORMAT_REUTERS)

    content = ''
    for file in list_of_filename:
        with open(DIR_REUTERS_TRAIN + '/' + file, 'r') as file_content:
            lines = file_content.readlines()
            for line in lines:
                if line and line.strip():
                    for word_tag in line.split():
                        current_word = word_tag.split('/', 1)[0]
                        content += current_word + ' '
                    content += '\n'
        file_content.close()

    file_path = DIR_REFORMAT_REUTERS + '/' + REFORMAT_REUTERS
    with open(file_path, 'wb') as file:
        file.write(str(content))
    file.close()

    list_of_filename = os.listdir(DIR_REUTERS_TEST)
    content = ''
    for file in list_of_filename[:250]:
        with open(DIR_REUTERS_TEST + '/' + file, 'r') as file_content:
            lines = file_content.readlines()
            for line in lines:
                if line and line.strip():
                    for word_tag in line.split():
                        current_word = word_tag.split('/', 1)[0]
                        content += current_word + ' '
                    content += '\n'
        file_content.close()

    file_path = DIR_REFORMAT_REUTERS + '/' + REFORMAT_REUTERS_TEST
    with open(file_path, 'wb') as file:
        file.write(str(content))
    file.close()


def save_trained_data(data, filename, brown_corpus):
    if brown_corpus:
        dir_path = DIR_BROWN_TRAIN_MODEL
    else:
        dir_path = DIR_REUTERS_TRAIN_MODEL

    file_path = dir_path + '/' + filename
    with open(file_path, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    file.close()
    print ("Finish save trained file: %s" % file_path)


def get_trained_data(filename, brown_corpus):
    if brown_corpus:
        dir_path = DIR_BROWN_TRAIN_MODEL
    else:
        dir_path = DIR_REUTERS_TRAIN_MODEL

    file_path = dir_path + '/' + filename
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    file.close()
    print ("Finish get trained file: %s" % file_path)
    return data


def compute_perplexity(brow_corpus):
    log_probability = 0.0
    M = 0

    if brow_corpus:
        file_path = DIR_REFORMAT_BROWN_CORPUS + '/' + REFORMAT_BROWN_CORPUS_TEST
    else:
        file_path = DIR_REFORMAT_REUTERS + '/' + REFORMAT_REUTERS_TEST

    with open(file_path, 'rb') as file_content:
        index = 0
        lines = file_content.readlines()
        for line in lines:
            if line and line.strip() and len(line) < 500:
                index += 1
                line += ' STOP'
                p_sentence, num_word = get_p_sentence(line)
                log_probability += math.log(p_sentence, 2)
                M += num_word
                print ("Complete: %d / %d" % (index, len(lines)))
    file_content.close()

    l = log_probability / M
    perplexity = math.pow(2, (-1.0 * l))

    print ("=============================")
    print ("Vocabulary size: %d" % len(unigram))
    print ("Number of sentences test: %d" % index)
    print ("Number of words test: %d" % M)
    print ("Log Probability: %f" % log_probability)
    print ("Perplexity: %f" % perplexity)


def get_p_sentence(sentence):
    sentence = sentence.split()
    n = len(sentence)
    p_sentence = 1.0
    for k in range(1, n + 1):
        word = get_word(sentence, k - 1)
        last_word = get_word(sentence, k - 2)
        penult_word = get_word(sentence, k - 3)

        p_sentence *= get_q_trigram(penult_word, last_word, word)

    return (p_sentence, n)


def get_word(sentence, k):
    if k < 0:
        return '*'
    else:
        if sentence[k] not in unigram:
            print("<Warning: '%s' is not exist in the vocabulary>" % sentence[k])
            return '<unk>'
        return sentence[k]


def get_q_bigram(last_word, word):
    if len(A_bigram[last_word]) > 0:
        if word in A_bigram[last_word]:
            q = (float(bigram[last_word, word] - DISCOUNT_VALUE) / unigram[last_word])
        else:
            alpha = alpha_unigram(last_word)
            c_wi = unigram[word]
            total_c_w_in_A_bigram = 0
            for pos in A_bigram[last_word]:
                total_c_w_in_A_bigram += unigram[pos]
            total_c_w_in_B_bigram = COUNT - total_c_w_in_A_bigram

            q = alpha * (float(c_wi) / total_c_w_in_B_bigram)
    else:
        q = float(unigram[word]) / COUNT

    return q


def get_q_trigram(penult_word, last_word, word):
    if len(A_trigram[penult_word, last_word]) > 0:
        if word in A_trigram[penult_word, last_word]:
            q = (float(trigram[penult_word, last_word, word] - DISCOUNT_VALUE) / bigram[penult_word, last_word])
        else:
            aplpha = alpha_bigram(penult_word, last_word)
            q_bo_of_word = get_q_bigram(last_word, word)
            total_q_bo_in_A_trigram = 0.0
            for pos in A_trigram[penult_word, last_word]:
                total_q_bo_in_A_trigram += \
                    get_q_bigram(last_word, pos)

            q = aplpha * (float(q_bo_of_word) / (1 - total_q_bo_in_A_trigram))

    else:
        q = get_q_bigram(last_word, word)

    return q


def alpha_unigram(last_word):
    alpha = float((DISCOUNT_VALUE * len(A_bigram[last_word]))) / \
            unigram[last_word]

    return alpha


def alpha_bigram(penult_word, last_word):
    alpha = float(DISCOUNT_VALUE * len(A_trigram[penult_word, last_word])) / \
            bigram[penult_word, last_word]

    return alpha


def gen_random_sentence():
    sentence = ''

    new = defaultdict(list)
    for pos in unigram:
        if re.match("^[A-Za-z_-]*$", pos):
            new[pos] = unigram[pos]

    word_3 = ''
    word_1 = random.choice(new.keys())
    word_2 = random.choice(new.keys())

    sentence += word_1 + ' '
    sentence += word_2 + ' '

    print sentence

    index = 0
    while word_3 != 'STOP' and index < 20:
        if (word_1, word_2) in A_trigram:
            # print 1
            prob, word_3 = max((get_q_trigram(word_1, word_2, w), w) for w in A_trigram[word_1, word_2])
        elif word_1 in A_bigram:
            # print 2
            prob, word_3 = max((get_q_trigram(word_1, word_2, w), w) for w in A_bigram[word_1])
        else:
            # print 3
            prob, word_3 = max((get_q_trigram(word_1, word_2, w), w) for w in new)

        index += 1

        sentence += word_3 + ' '
        word_1 = word_2
        word_2 = word_3

    print '\n'
    print sentence


if not os.path.isdir(DIR_REFORMAT_BROWN_CORPUS) or not os.path.isdir(DIR_REFORMAT_REUTERS):
    if not os.path.isdir(DIR_REFORMAT_BROWN_CORPUS):
        format_brown_corpus()
    if not os.path.isdir(DIR_REFORMAT_REUTERS):
        format_reuters()

elif not os.path.isdir(DIR_BROWN_TRAIN_MODEL) or not os.path.isdir(DIR_REUTERS_TRAIN_MODEL):
    if not os.path.isdir(DIR_BROWN_TRAIN_MODEL):
        with open(DIR_REFORMAT_BROWN_CORPUS + '/' + REFORMAT_BROWN_CORPUS, 'r') as file_content:
            lines = file_content.readlines()
            count_word = 0
            index = 0
            for line in lines:
                if line and line.strip():
                    index += 1
                    line += " STOP"
                    penult_word = '*'
                    last_word = '*'

                    unigram[penult_word] += 2
                    bigram[penult_word, last_word] += 1

                    if last_word not in A_bigram[penult_word]:
                        A_bigram[penult_word].append(last_word)

                    for word in line.split():
                        count_word += 1
                        current_word = word

                        unigram[current_word] += 1
                        bigram[last_word, current_word] += 1
                        trigram[penult_word, last_word, current_word] += 1

                        if word not in A_bigram[last_word]:
                            A_bigram[last_word].append(word)
                        if word not in A_trigram[penult_word, last_word]:
                            A_trigram[penult_word, last_word].append(word)

                        penult_word = last_word
                        last_word = current_word

                    print ("Complete: %d / %d" % (index, len(lines)))

            unigram['<unk>'] = len(unigram)

        file_content.close()

        # for last_word in unigram:
        #     for word in unigram:
        #         if (last_word, word) in bigram:
        #             A_bigram[last_word].append(word)
        #             # else:
        #             #     B_bigram[last_word].append(word)
        #     print ("A_bigram: Complete %d. Number of unigram: %d" % (len(A_bigram), len(unigram)))

        # for (penult_word, last_word) in bigram:
        #     for word in unigram:
        #         if (penult_word, last_word, word) in trigram:
        #             A_trigram[penult_word, last_word].append(word)
        #             # else:
        #             #     B_trigram[penult_word, last_word].append(word)
        #     print ("A_trigram: Complete %d. Number of bigram: %d" % (len(A_trigram), len(bigram)))

        os.makedirs(DIR_BROWN_TRAIN_MODEL)
        save_trained_data(unigram, UNIGRAM, True)
        save_trained_data(bigram, BIGRAM, True)
        save_trained_data(trigram, TRIGRAM, True)

        save_trained_data(A_bigram, A_BIGRAM, True)
        save_trained_data(A_trigram, A_TRIGRAM, True)

    if not os.path.isdir(DIR_REUTERS_TRAIN_MODEL):
        unigram = defaultdict(int)
        bigram = defaultdict(int)
        trigram = defaultdict(int)
        A_bigram = defaultdict(list)
        A_trigram = defaultdict(list)

        with open(DIR_REFORMAT_REUTERS + '/' + REFORMAT_REUTERS, 'r') as file_content:
            lines = file_content.readlines()
            count_word = 0
            index = 0
            for line in lines:
                if line and line.strip():
                    index += 1
                    line += " STOP"
                    penult_word = '*'
                    last_word = '*'

                    unigram[penult_word] += 2
                    bigram[penult_word, last_word] += 1

                    if last_word not in A_bigram[penult_word]:
                        A_bigram[penult_word].append(last_word)

                    for word in line.split():
                        count_word += 1
                        current_word = word

                        unigram[current_word] += 1
                        bigram[last_word, current_word] += 1
                        trigram[penult_word, last_word, current_word] += 1

                        if word not in A_bigram[last_word]:
                            A_bigram[last_word].append(word)
                        if word not in A_trigram[penult_word, last_word]:
                            A_trigram[penult_word, last_word].append(word)

                        penult_word = last_word
                        last_word = current_word

                    print ("Complete: %d / %d" % (index, len(lines)))

            unigram['<unk>'] = len(unigram)

        file_content.close()

        os.makedirs(DIR_REUTERS_TRAIN_MODEL)
        save_trained_data(unigram, UNIGRAM, False)
        save_trained_data(bigram, BIGRAM, False)
        save_trained_data(trigram, TRIGRAM, False)

        save_trained_data(A_bigram, A_BIGRAM, False)
        save_trained_data(A_trigram, A_TRIGRAM, False)

else:
    choice = raw_input("\n"
                       "== Enter your choice: "
                       "\n 1: load trigram model"
                       "\n 2: test perplexity"
                       "\n 3: generate random sentence"
                       "\n")

    while choice != 'exit':
        if choice == '1':
            COUNT = 0

            choice_2 = raw_input("\n"
                                 "== Choose model type: "
                                 "\n 1: brown corpus"
                                 "\n 2: reuters"
                                 "\n")
            if choice_2 == '1':
                get_brown = True

            elif choice_2 == '2':
                get_brown = False

            unigram = get_trained_data(UNIGRAM, get_brown)
            bigram = get_trained_data(BIGRAM, get_brown)
            trigram = get_trained_data(TRIGRAM, get_brown)

            A_bigram = get_trained_data(A_BIGRAM, get_brown)
            A_trigram = get_trained_data(A_TRIGRAM, get_brown)

            for word in unigram:
                if word != '*' and word != '<unk>':
                    COUNT += unigram[word]

        elif choice == '2':
            choice_2 = raw_input("\n"
                                 "== Choose file test type: "
                                 "\n 1: brown corpus"
                                 "\n 2: reuters"
                                 "\n")
            if choice_2 == '1':
                compute_perplexity(True)

            elif choice_2 == '2':
                compute_perplexity(False)

        elif choice == '3':
            gen_random_sentence()

        elif choice == 'help':
            print ("\n"
                   "\n 1: load trigram model"
                   "\n 2: test perplexity"
                   "\n 3: generate random sentence"
                   "\n")

        else:
            print 'wrong option'

        choice = raw_input("\n== Enter your choice: ")
