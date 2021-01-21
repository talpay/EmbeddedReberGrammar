#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:39:06 2015

@author: tay
"""
import argparse
import os
import random

import numpy as np
import csv

class Reber(object):
    def __init__(self, seed=42, prediction=True):

        self.seed = seed
        random.seed(self.seed)

        # decide Prediction vs Classification:
        self.prediction = prediction

        self.vocabSize = 8

        self.UNK = '#'
        self.PAD = ' ' #' '

        # based on: https://github.com/danvk/lstm-examples
        self.transitions = [
            [('T', 1), ('P', 2)],
            [('X', 3), ('S', 1)],
            [('V', 4), ('T', 2)],
            [('X', 2), ('S', 5)],
            [('P', 3), ('V', 5)],
            [('E', -1)]]


        self.atoi = {
            self.PAD: 0,
            'T': 1,
            'P': 2,
            'S': 3,
            'X': 4,
            'V': 5,
            'E': 6,
            'B': 7,
        }

        self.itoa = {v: k for k, v in self.atoi.items()}

        self.currentSet = []
        self.currentSetInt = []
        self.currentInputs = []
        self.currentTargets = []
        self.outputLabels = []

    @staticmethod
    def get_labels(data):
        """
            data: list of intStr-coded symbols
            generates labels instead of output vectors
        """

        Y = []
        for seq in data:
            Ys = []
            for sym in seq:
                Ys.append(int(sym))
            Y.append(Ys)

        return np.asarray(Y).astype('i')

    @staticmethod
    def remove_last(currentInputs, currentTargets, outputAsLabels=False):

        axis = -1 if outputAsLabels else 1

        currentInputs = np.delete(currentInputs, -1, axis=1)
        currentTargets = np.delete(currentTargets, -1, axis=axis)

        return currentInputs, currentTargets

    def make_reber(self, minLen, maxLen):
        idx = 0
        out = 'B'
        while idx != -1:
            ts = self.transitions[idx]
            symbol, idx = random.choice(ts)
            out += symbol
        return out

    def make_embedded_reber(self, minLen, maxLen):
        c = random.choice(['T', 'P'])
        return 'B%s%s%sE' % (c, self.make_reber(minLen, maxLen), c)

    def make_sequences(self, num, minLen, maxLen, embedded=True, patience=1000000):
        """ Generates human-readable sequences in a list
            e.g. ['BPTTVPXTVPXVVE', 'BPTTVPXTTTTVVE']

            input and output are lists
            patience:1000000
        """

        lexicon = []

        if embedded:
            generate = self.make_embedded_reber
            minSymbols = 9
        else:
            generate = self.make_reber
            minSymbols = 5

        if maxLen < minSymbols:
            raise NameError('maxLen too small, require maxLen > %s' % minSymbols)

        # raise value to minimum bound
        if minLen < minSymbols:
            minLen = minSymbols

        print('Generating {} Reber sequences of length [{},{}] with seed {} (embedded={})'.format(
            num, minLen, maxLen, self.seed, embedded))

        x = -1
        while True:
            x += 1
            if x > patience:
                print("WARNING: Patience of Reber Generator elapsed!")
                print("Note that there are a limited amount of Reber-words for a given length.")
                break

            word = generate(minLen, maxLen)
            if len(word) >= minLen:
                # padding:
                word += self.PAD * (maxLen - len(word))
                if len(word) > maxLen or word in lexicon:
                    continue

                lexicon.append(word)
                x = 0

            if len(lexicon) == num:
                break

        print("Collected # of samples:", len(lexicon))

        return lexicon

    def seq_str_to_int(self, s):
        """ accepts symbol-string and returns integer-string """

        res = ''
        for char in s:
            res += str(self.atoi.get(str(char), self.UNK))

        if self.UNK in res:
            raise Exception('unknown character could not be encoded within sequence: %s' % res)
        return res

    def str_to_int(self, data):
        """ input and output is list of str
            (no arrays/ints due to misinterpreting leading/trailing 0)
        """

        X = []
        for seq in data:
            X.append(self.seq_str_to_int(seq))

        return X

    def batch_to_str(self, batch):
        """ accepts batch and returns symbol-strings.
            preserves dimensionality.
        """

        dim = len(batch.shape)
        b = []

        # input batch with shape (BATCH-SIZE, NUM, 7):
        if dim == 3:
            for seq in batch:
                seqL = []
                for symVec in seq:
                    # decode from int to reberSym
                    if type(symVec) is not np.array:
                        symVec = np.array(symVec)
                    if np.all((symVec == 0)):
                        symStr = self.PAD
                    else:
                        idx = np.where(symVec == 1)[0]
                        symStr = self.itoa.get(int(idx))
                    seqL.append(symStr)
                b.append(seqL)

        # output/prediction batch with shape (BATCH-SIZE, NUM)
        # e.g.    [[1 0 1 1 5 5 6 1 6 0], [1 0 1 3 5 3 6 1 6 0]]
        elif dim == 2:
            for seq in batch:
                seqStr = ""
                for symInt in seq:
                    symInt = int(round(symInt))  # in case of float
                    sym = self.itoa.get(symInt)
                    seqStr += sym
                b.append(seqStr)

        return b

    def generate_targets(self, inputs, shift=1, outputAsLabels=False):
        """ shift targets by shift steps
            shift=1 for 1 step-ahead-prediction
        """

        if self.prediction:
            axis = -1 if outputAsLabels else 1
            Y = np.roll(inputs, -shift, axis=axis).astype('i' if outputAsLabels else 'f')
        else:
            Y = inputs

        return Y

    def one_hot_encoding(self, data):
        """
            data: list of intStr-coded symbols
            generates np.array-batches ready for processing
        """

        X = []
        for seq in data:
            Xs = []
            for sym in seq:
                inp = np.zeros(self.vocabSize)
                if int(sym) != self.atoi[self.PAD]:
                    inp[int(sym)] = 1
                Xs.append(inp)
            X.append(Xs)

        return np.asarray(X).astype('f')

    def get_output_labels(self, data=None, outputAsLabels=False, asSymbols=False):
        """ get the output labels for the current batches """

        inp = data if data is not None else self.currentSetInt

        result = self.generate_targets(self.get_labels(inp), outputAsLabels=outputAsLabels)
        if len(self.outputLabels) == 0 and data is None:
            self.outputLabels = result

        if asSymbols:
            b1 = []
            for seq in self.outputLabels:
                b2 = ''
                for sym in seq:
                    b2 += self.itoa.get(sym)
                b1.append(b2)
            return b1

        return result

    def get_batches(self, num, minLen=10, maxLen=30, embedded=True, outputAsLabels=None, patience=1000000):
        """ Generate input batches and targets
            and return them as a list: [inputs, outputs]
        """

        assert maxLen >= minLen, "Error: maxLen can't be smaller than minLen"
        if minLen != maxLen:
            print("Varying sequence lengths will be padded with 0-vector and symbol: " + r.PAD)

        if outputAsLabels is None:
            outputAsLabels = not self.prediction

        # make human-readable set
        self.currentSet = self.make_sequences(num=num, minLen=minLen, maxLen=maxLen, embedded=embedded,
                                              patience=patience)

        # make int-encoding
        self.currentSetInt = self.str_to_int(self.currentSet)

        # make unit-encoding (local input-output representation)
        self.currentInputs = self.one_hot_encoding(self.currentSetInt)

        if outputAsLabels:  # output vector only has labels
            self.currentTargets = self.get_output_labels()
        else:  # output vector has unit-coding
            self.currentTargets = self.generate_targets(self.currentInputs, outputAsLabels=outputAsLabels)

        if self.prediction:
            # remove last element (inp and target) since we're not predicting it
            self.currentInputs, self.currentTargets = self.remove_last(self.currentInputs, self.currentTargets,
                                                                       outputAsLabels=outputAsLabels)

        # print("Generated the following Reber set: ", self.currentSet)
        return [self.currentInputs, self.currentTargets]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, help="number of samples", default=100000)
    parser.add_argument("--minLen", type=int, help="min sequence length", default=5)
    parser.add_argument("--maxLen", type=int, help="max sequence length", default=50)
    parser.add_argument('--seed', type=int, help="random seed", default=42)
    parser.add_argument('--prediction', type=lambda x: (str(x).lower() == 'true'), help="Prediction: shift targets by 1 timestep", default=False)
    parser.add_argument('--embedded', type=lambda x: (str(x).lower() == 'true'), help="Embedded Reber Grammar or Reber Grammar", default=True)
    parser.add_argument('--patience', type=int, help="Script aborts after 'patience' iterations without success of generating a new token", default=1000000)
    parser.add_argument("--results_dir", type=str, help="directory for results", default="./data")

    args = parser.parse_args()

    r = Reber(seed=args.seed, prediction=args.prediction)

    x, y = r.get_batches(num=args.num, minLen=args.minLen, maxLen=args.maxLen, embedded=args.embedded, patience=args.patience)
    s = r.batch_to_str(x)

    subdir = str(args.num)+"_"+str(args.minLen)+"_"+str(args.maxLen)

    if not os.path.exists(args.results_dir+"/"+subdir):
        os.makedirs(args.results_dir+"/"+subdir)

    with open(args.results_dir+"/"+subdir+"/raw_"+str(args.num)+"_"+str(args.minLen)+"_"+str(args.maxLen)+".txt", "w", newline="") as f:
        f.write("\n".join(r.currentSet))

    with open(args.results_dir+"/"+subdir+"/token_"+str(args.num)+"_"+str(args.minLen)+"_"+str(args.maxLen)+".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(s)

    np.save(args.results_dir+"/"+subdir+"/x_"+str(args.num)+"_"+str(args.minLen)+"_"+str(args.maxLen), x)
    np.save(args.results_dir+"/"+subdir+"/y_"+str(args.num)+"_"+str(args.minLen)+"_"+str(args.maxLen), y)

    print("Generated data in "+args.results_dir+"/"+subdir)
