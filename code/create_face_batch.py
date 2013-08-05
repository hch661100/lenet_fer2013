import sys
import os
import theano
import theano.tensor as T
import numpy as np
import cPickle
import random
import csv
import numpy
from math import ceil
from time import time

SIZE_IMG = 48

class CreateFaceBatch:
     def __init__(self, fcsv, path_data):
        self.trn_data, self.trn_label, self.tst_data, self.tst_label = self.load_csv(fcsv)
        self.path_data = path_data

     def load_csv(self, fcsv):
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        reader = csv.reader(file(fcsv, 'r')) 
        for line in reader:
           if reader.line_num == 1:
              continue
           tmp1 = line[1].split()
           tmp2 = [int(p) for p in tmp1]
           if reader.line_num <= 28710: 
             train_label.append(int(line[0]))
             train_data.append(tmp2)
           else:
             test_label.append(int(line[0]))
             test_data.append(tmp2)

        return train_data, train_label, test_data, test_label

     def build_batch(self):
          f = open(self.path_data, 'wb')
          data = {}
          data['train_data'] = self.trn_data
          data['train_label'] = self.trn_label
          data['test_data'] = self.tst_data
          data['test_label'] = self.tst_label
          cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
          f.close()
          
          
def generate_shared_data(data):
         data=numpy.asarray(data)
         shared_data = theano.shared(numpy.asarray(data,dtype=theano.config.floatX),borrow=True)
         print len(data)
         return shared_data

def load_data(data):
         f = open(data,'r')
         dataset = cPickle.load(f)
         trn_data = dataset['train_data']
         trn_label = dataset['train_label']
         tst_data = dataset['test_data']
         tst_label = dataset['test_label']

         train_data = generate_shared_data(trn_data)
         train_label = generate_shared_data(trn_label)
         test_data = generate_shared_data(tst_data)
         test_label = generate_shared_data(tst_label)

         return  train_data, T.cast(train_label,'int32'), test_data, T.cast(test_label, 'int32')

if __name__ == "__main__":
    fcsv = sys.argv[1]
    path_data = sys.argv[2]
    model = CreateFaceBatch(fcsv, path_data)
    model.build_batch()
             
