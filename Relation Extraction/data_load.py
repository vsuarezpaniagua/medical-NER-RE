import numpy as np
import re
import unicodedata
import os
import sys
from copy import deepcopy
from nltk import pos_tag
from nltk import word_tokenize
#word_tokenize = list # char-level
import networkx as nx
import spacy
nlp = spacy.load("es_dep_news_trf")
from sklearn.metrics import confusion_matrix


ENTITY1 = "1"
ENTITY2 = "2"
ENTITY0 = "0"
NUM = "#"


def clean_str(sentence):
    """
    Adapted from:
    https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # Keep characters, numbers and special characteres
    sentence = ''.join([c for c in unicodedata.normalize('NFD',sentence) if unicodedata.category(c) != 'Mn'])
    sentence = re.sub(u"[^A-Za-z0-9(),]", " ", sentence)
    # Replace special characters
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    # Replace numbers
    sentence = re.sub(r"\d", " " + NUM + " ", sentence)
    sentence = re.sub(r"(" + re.escape(NUM) + "(\s)*)+" + re.escape(NUM), " " + NUM + " ", sentence)
    # Replace the entities
    sentence = re.sub('entitya', ENTITY1, sentence)
    sentence = re.sub('entityb', ENTITY2, sentence)
    sentence = re.sub('entityx', ENTITY0, sentence)
    # Replace multiple spaces
    sentence = re.sub(r"\s{2,}", " ", sentence)
    return sentence.strip().lower() + " ."


def deoverlapping(overlapped, discarded = [], discarded_overlapped = []):
    """
    Get the all possible discarded overlapped in a list to have non-overlap
    overlapped = list of lists with overlapped [[overlap, [whom overlap]], ...]
    discarded = temporal discarded overlapped
    discarded_overlapped = complete discarded list to have a non-overlap
    """
    # all overlapped are None (No overlap) => keep the discarded
    if all(value[1] is None for value in overlapped):
        discarded.sort()
        if discarded not in discarded_overlapped:
            discarded_overlapped.append(discarded)
    # OVERLAP!
    else:
        for i in range(len(overlapped)):
            if not overlapped[i][1] == None:
                # Copy to pass-by-value
                overlapped_copy = deepcopy(overlapped)
                discarded_copy = deepcopy(discarded)
                for j in overlapped[i][1]:
                    if j not in discarded:
                        # Add to discarded the new whom overlap
                        discarded_copy.append(j)
                        # Find the index
                        for k in range(len(overlapped)):
                            if overlapped[k][0] == j:
                                break
                        # Remove the new whom overlap to continue
                        overlapped_copy[k][1] = None
                # Remove the overlap for the recursion
                overlapped_copy[i][1] = None
                discarded_overlapped = deoverlapping(overlapped_copy, discarded_copy, discarded_overlapped)
    return discarded_overlapped


def load_data(sentence_paths, entities_paths, relations_paths):
    """
    Load instances and labels from a list of paths
    sentence_paths = the list of paths which contain the sentences files
    entities_paths = the list of paths which contain the entities files
    relations_paths = the list of paths which contain the relation files (optional = [])
    """
    total_instances = []
    total_labels = []
    for path in range(len(sentence_paths)):
        
        # Read the sentences file
        with open(sentence_paths[path], 'rb') as f:
            sentences = f.read().decode("utf-8")
        
        # Read the entities file
        with open(entities_paths[path], 'rb') as f:
            fileread = f.readlines()
        #print([entity for entity in (entity[:-1].decode("utf8").split("\t") for entity in fileread) for x in entity[1].split(" ") if ";" in x and not int(x.split(";")[0])+1 == int(x.split(";")[1])]) # print all the disconituous entities
        entities = [[entity[0], int(entity[1].split()[0]), int(entity[1].split()[-1])] for entity in (entity[:-1].decode("utf8").split("\t") for entity in fileread)]
        types = {etype[0]: etype[2] for etype in (etype[:-1].decode("utf8").split("\t") for etype in fileread)}
        ind_sentence = [0] + [ind+1 for ind, word in enumerate(sentences) if word == "\n"]
        entities = [[[entity[0], entity[1]-ind_sentence[ind], entity[2]-ind_sentence[ind]] for entity in entities if entity[1] > ind_sentence[ind] and entity[2] < ind_sentence[ind+1]] for ind in range(len(ind_sentence)-1)]
        
        # Read the relation file
        if os.path.isfile(relations_paths[path]):
            with open(relations_paths[path], 'rb') as f:
                relations = [[relation[0], relation[1], relation[2]] for relation in (relation.decode("utf-8").split() for relation in f.readlines())]
            relations = [[relation for relation in relations if relation[1] in list(zip(*sentence))[0] and relation[2] in list(zip(*sentence))[0]] for sentence in entities]
        # Create the instances and labels for the current path
        sentences = sentences.split("\n")[:-1]
        instances, labels = [], []
        for idx in range(len(entities)):
            # Create different sentences with the different overlapping entities
            overlaps = [[e1[0], [e2[0] for e2 in entities[idx] if not e1==e2 and e1[1]<e2[2] and e1[2]>e2[1]]] for e1 in entities[idx]]
            #print(overlaps) # print all the overlapping entities
            overlaps = [[o[0], o[1]] if o[1] else [o[0], None] for o in overlaps]
            overlaps = deoverlapping(overlaps, [], [])
            for overlap in overlaps:
                for e1_idx in range(len(entities[idx])):
                    for e2_idx in range(len(entities[idx])):
                        if not e1_idx == e2_idx and entities[idx][e1_idx][0] not in overlap and entities[idx][e2_idx][0] not in overlap:
                            
                            # Name entity blinding
                            text = sentences[idx]
                            entities_idx = [e for e in entities[idx] if e[0] not in overlap]
                            for e_idx in range(len(entities_idx)):
                                e_id, offset1, offset2 = entities_idx[e_idx]
                                name = (e_id==entities[idx][e1_idx][0])*" entitya " + (e_id==entities[idx][e2_idx][0])*" entityb " + (not e_id==entities[idx][e1_idx][0] and not e_id==entities[idx][e2_idx][0])*" entityx "#text[offset1:offset2]
                                text = text[:offset1] + name + text[offset2:]
                                # Replace offsets
                                diff = len(name) + offset1 - offset2
                                entities_idx = [[e[0], e[1] + (offset1 < e[1])*diff, e[2] + (offset2 < e[2])*diff] for e in entities_idx]
                            
                            # Label the sentence
                            relation = 'None'
                            if os.path.isfile(relations_paths[path]):
                                for relation_id, e1_id, e2_id in relations[idx]:
                                    if entities[idx][e1_idx][0] == e1_id and entities[idx][e2_idx][0] == e2_id:
                                        #if not relation == 'None':
                                        #    print(relation, relation_id, e1_id, e2_id) # print all the multilabel relations (comment break)
                                        relation = relation_id
                                        break # if there are equal entities in different relation we keep the first one!
                                        
                            #if types[entities[idx][e1_idx][0]] == b'Concept' and types[entities[idx][e2_idx][0]] == b'Concept':#Relationships
                            #if not types[entities[idx][e1_idx][0]] == b'Concept' and types[entities[idx][e2_idx][0]] == b'Concept':#Roles
                            instances.append(clean_str(text))
                            labels.append([relation, entities[idx][e1_idx][0], entities[idx][e2_idx][0], types[entities[idx][e1_idx][0]], types[entities[idx][e2_idx][0]]])
        total_instances += instances
        total_labels += labels
    return np.array(total_instances), np.array(total_labels)


def SDP(sentence):
    """
    Reduce the sentence to the Shortest Dependency Path
    """
    try:
        sentence = re.sub(ENTITY1, 'entity1', sentence)
        sentence = re.sub(ENTITY2, 'entity2', sentence)
        sentence = re.sub(ENTITY0, 'entity0', sentence)
        edges = [('{0}'.format(token.lower_,token.i), '{0}'.format(child.lower_,child.i)) for token in nlp(sentence) for child in token.children]
        sentence = " ".join(nx.shortest_path(nx.Graph(edges), source='entity1', target='entity2'))
        sentence = re.sub('entity1', ENTITY1, sentence)
        sentence = re.sub('entity2', ENTITY2, sentence)
        sentence = re.sub('entity0', ENTITY0, sentence)
    except nx.NetworkXNoPath:
        pass
    return sentence


def load_features(instances):
    distances = [[word_tokenize(instance).index(ENTITY1), word_tokenize(instance).index(ENTITY2)] for instance in instances]
    pos = [" ".join(['NN' for p in word_tokenize(instance)]) for instance in instances]#[" ".join([p[1] if not (p[0] == ENTITY1 or p[0] == ENTITY2 or p[0] == ENTITY0) else 'NN' for p in pos_tag(word_tokenize(instance))]) for instance in instances]
    sdp = [instance for instance in instances]#[SDP(instance) for instance in instances]
    return np.array(distances), np.array(pos), np.array(sdp)


def print_results(y_true, y_pred, labelsnames, verbose = False):
    labels = labelsnames[:]
    idx = labels.index("None")
    labels[idx] = "Other"
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    eps = sys.float_info.epsilon
    # Classification Results
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    fmeasure = 2 * precision * recall / (precision + recall + eps)
    support = TP + FN
    # Micro-Average without Other-class
    m_TP = np.sum(TP) - TP[idx]
    m_FP = np.sum(FP) - FP[idx]
    m_FN = np.sum(FN) - FN[idx]
    m_precision = m_TP / (m_TP + m_FP + eps)
    m_recall = m_TP / (m_TP + m_FN + eps)
    m_fmeasure = 2 * m_precision * m_recall / (m_precision + m_recall + eps)
    m_support = np.sum(support) - support[idx]
    # Macro-Average without Other-class
    M_TP = (np.mean(TP) * len(TP) - TP[idx]) / (len(TP) - 1)
    M_FP = (np.mean(FP) * len(FP) - FP[idx]) / (len(FP) - 1)
    M_FN = (np.mean(FN) * len(FN) - FN[idx]) / (len(FN) - 1)
    M_precision = (np.mean(precision) * len(precision) - precision[idx]) / (len(precision) - 1)
    M_recall = (np.mean(recall) * len(recall) - recall[idx]) / (len(recall) - 1)
    M_fmeasure = 2 * M_precision * M_recall / (M_precision + M_recall + eps)
    M_support = (np.mean(support) * len(support) - support[idx]) / (len(support) - 1)
    if verbose:
        max_l = max([len(l) for l in labels])
        string = "CLASSIFICATION ANALYTICS\n"
        class_mat = np.zeros((len(labels)+3, 8), dtype="S" + str(max(max_l, 12)))
        class_mat[0,:] = ["Classes", "TP", "FP", "FN", "Precision", "Recall", "F-measure", "Support"]
        class_mat[1:,0] = labels + ["MicroAverage", "MacroAverage"]
        class_mat[1:,1] = np.concatenate([TP, [m_TP, str(np.around(M_TP, decimals=2))]])
        class_mat[1:,2] = np.concatenate([FP, [m_FP, str(np.around(M_FP, decimals=2))]])
        class_mat[1:,3] = np.concatenate([FN, [m_FN, str(np.around(M_FN, decimals=2))]])
        class_mat[1:,4] = np.around(100*np.concatenate([precision, [m_precision, M_precision]]), decimals=2)
        class_mat[1:,5] = np.around(100*np.concatenate([recall, [m_recall, M_recall]]), decimals=2)
        class_mat[1:,6] = np.around(100*np.concatenate([fmeasure, [m_fmeasure, M_fmeasure]]), decimals=2)
        class_mat[1:,7] = np.concatenate([support, [m_support, str(np.around(M_support, decimals=2))]])
        string += "\n".join(["".join([("%-" + str(max(max_l,12)+4)*(column==0)+str(len(class_mat[0][column])+4)*(column>0) + "s") % (class_mat[row][column].decode("utf-8")) for column in range(class_mat.shape[1])]) for row in range(class_mat.shape[0])])
        string += "\nCONFUSION MATRIX\n"
        con_mat = np.zeros((len(labels)+1, len(labels)+1), dtype="S" + str(max(max_l, 5)))
        con_mat[0,:] = ["Classes"] + labels
        con_mat[1:,0] = labels
        con_mat[1:len(labels)+1, 1:len(labels)+1] = cm
        string += "\n".join(["".join([("%-" + str(max(max_l,12)+4)*(column==0)+str(len(con_mat[0][column])+4)*(column>0) + "s") % (con_mat[row][column].decode("utf-8")) for column in range(con_mat.shape[1])]) for row in range(con_mat.shape[0])])
        print(string)
    return [["MicroAverage", 100*m_precision, 100*m_recall, 100*m_fmeasure], ["MacroAverage", 100*M_precision, 100*M_recall, 100*M_fmeasure]]