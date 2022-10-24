import argparse
import numpy as np
import random as r
import math as mt

from sklearn.svm import SVC as SVM
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import pickle

import os
from os import listdir
from os.path import isfile, join

class Sushifier:

    def distance_to_cost(self,x):
        y = 1 if (x==0) else x
        return round(max(10/y,1))

    def __init__(self, usr = None, dataset='USER'):
        if dataset not in ['USER', 'SUSHI']:
            self.path_mode = 'USER'
        else:
            self.path_mode = dataset
        
        self.path = {
            'USER':{'dataset':'./users_preferences/', 
                    'models':'./users_svm_models/'},
            'SUSHI':{'dataset':'./sushi_characteristics/', 
                     'models':'./sushi_svm_models/'}
        }
        
        if usr != None and (usr < 1 or usr > 5000):
            print('Wrong user ID. The classifier won\'t work.')
        self.user = usr
        self.users_list = []
        
        # models_path = './users_svm_models/'
        directory_content = [f for f in listdir(self.path[self.path_mode]['models']) if isfile(join(self.path[self.path_mode]['models'], f))]
        if self.user != None and 'sushi_svm_user'+str(self.user)+'.sav' in directory_content:
            filename = self.path[self.path_mode]['models'] + 'sushi_svm_user' + str(self.user)+'.sav'
            self.svm = pickle.load(open(filename, 'rb'))
            print ('SVM model successfully loaded.')
        else:
            self.svm = self.build_svm_model()
            self.train_model(True)
            print ('No SVM model has been found. New model has been instantiated.')
        
    def build_svm_model(self):
        svm = SVM(probability=True, kernel='poly')
        return svm
    
    def set_user(self, usr):
        if usr != None and (usr < 1 or usr > 5000):
            print('Wrong user ID. The classifier won\'t work.')
        else:
            self.user = usr
    
    def get_user(self):
        return 'User'+str(self.user)
    
    def save_model(self, svm, user):
        pickle.dump(svm, open(self.path[self.path_mode]['models'] + 'sushi_svm_user' + str(user) + '.sav', 'wb'))
        
    def load_model(self):
        directory_content = [f for f in listdir(self.path[self.path_mode]['models']) if isfile(join(self.path[self.path_mode]['models'], f))]
        if self.user != None and 'sushi_svm_user'+str(self.user)+'.sav' in directory_content:
            self.svm = pickle.load(open(self.path[self.path_mode]['models'] + 'sushi_svm_user' + str(self.user) + '.sav', 'rb'))
            print ('SVM model successfully loaded.')
        else:
            self.svm = self.build_svm_model()
            print ('No SVM model has been found. New model has been instantiated.')
    
    def get_users_models_list(self):
        directory_content = [f for f in listdir(self.path[self.path_mode]['models']) if isfile(join(self.path[self.path_mode]['models'], f))]
        l = [u[14:-4] for u in directory_content]
        return l
    
    def train_model(self, save = False):
        X = np.load(self.path[self.path_mode]['dataset'] + 'user' + str(self.user) + '_x.npy')
        Y = np.load(self.path[self.path_mode]['dataset'] + 'user' + str(self.user) + '_y.npy')
        self.svm.fit(X, Y)
        if save:
            self.save_model(self.svm, self.user)
            
    def train_users(self, random = False):
        USERS = [u for u in range(1,61)]
        if random == True:
            import random
            USERS = random.sample(range(1, 5001), 60)
        for u in USERS:
            print(u)
            X = np.load(self.path[self.path_mode]['dataset'] + 'user' + str(u) + '_x.npy')
            Y = np.load(self.path[self.path_mode]['dataset'] + 'user' + str(u) + '_y.npy')
            svm = self.build_svm_model()
            svm.fit(X, Y)
            self.save_model(svm, u)
    
    def evaluate_single_user(self, user = None):
        if user == None:
            user = self.user
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        X = np.load(self.path[self.path_mode]['dataset']+'user'+str(user)+'_x.npy')
        Y = np.load(self.path[self.path_mode]['dataset']+'user'+str(user)+'_y.npy')
        scores = []
        for train, test in kfold.split(X, Y):
            x_train, y_train = X[train], Y[train]
            x_test, y_test = X[test], Y[test]

            svm = SVM(probability=False, kernel='poly')
            svm.fit(x_train, y_train)
            svm_score = svm.score(x_test, y_test)
            scores.append(svm_score)
            
        mn_s = np.mean(scores)
        sd_s = np.std(scores)
        print('mean acc user ' + str(user) + ': ' + str(mn_s) + ' (+/-' + str(sd_s) + ')')
        return mn_s, sd_s
        
    def evaluate_multiple_users(self, random = False):
        USERS = [u for u in range(1,61)]
        if random == True:
            USERS = random.sample(range(1, 5001), 60)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        user_scores = []
        user_stds = []
        for user in USERS:
            X = np.load(self.path[self.path_mode]['dataset']+'user'+str(user)+'_x.npy')
            Y = np.load(self.path[self.path_mode]['dataset']+'user'+str(user)+'_y.npy')
            scores = []
            count = 1
            for train, test in kfold.split(X, Y):
                x_train, y_train = X[train], Y[train]
                x_test, y_test = X[test], Y[test]

                svm = SVM(probability=False, kernel='poly')
                svm.fit(x_train, y_train)
                svm_score = svm.score(x_test, y_test)
                scores.append(svm_score)
                
            mn_s = np.mean(scores)
            sd_s = np.std(scores)
            user_scores.append(mn_s)
            user_stds.append(sd_s)
        mn_s = np.mean(user_scores)
        sd_s = np.mean(user_stds)
        print('mean acc:' + str(mn_s) + ' (+/-' + str(sd_s) + ')')
        return mn_s, sd_s
    
    def get_items_features(self):
        covered = [0, 1, 2, 3, 4, 5, 6, 7, 26, 29]
        not_covered = []
        items = {}
        for i in range (0, 100):
            items[i] = {}
            if i not in covered:
                not_covered.append(i)
        with open('./sushi3.idata.csv', encoding='utf-8') as csv_file:
            import csv
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            for row in csv_reader:
                items[line_count]['name'] = row[1]
                items[line_count]['style'] = int(row[2])
                items[line_count]['major_group'] = int(row[3])
                items[line_count]['minor_group'] = int(row[4])
                items[line_count]['oiliness'] = float(row[5].replace(',', '.'))
                items[line_count]['freq_eat'] = float(row[6].replace(',', '.'))
                items[line_count]['price'] = float(row[7].replace(',', '.'))
                items[line_count]['freq_sold'] = float(row[8].replace(',', '.'))
                line_count += 1
        return items, not_covered, covered
    
    def generate_input(self, greater = None, than = None):
        items, not_covered, covered = self.get_items_features()
        if greater == None:
            greater, than = r.sample(covered, 2)
        
        if self.path_mode == 'USER':
            test_input = np.array([self.user, 
                                items[greater]['style'], items[than]['style'], 
                                items[greater]['major_group'], items[than]['major_group'], 
                                items[greater]['minor_group'], items[than]['minor_group'], 
                                items[greater]['oiliness'], items[than]['oiliness'], 
                                items[greater]['freq_eat'], items[than]['freq_eat'], 
                                items[greater]['price'], items[than]['price'], 
                                items[greater]['freq_sold'], items[than]['freq_sold']])
        elif self.path_mode == 'SUSHI':
            test_input = np.array([self.user, 
                               items[greater]['style'], items[than]['style'], 
                               items[greater]['major_group'], items[than]['major_group'], 
                               items[greater]['minor_group'], items[than]['minor_group'], 
                               items[greater]['oiliness'], items[than]['oiliness']])
        return test_input, greater, than, items
        
    def predict_random_input(self, greater = None, than = None):
        test_input, greater, than, items = self.generate_input(greater, than)
        return self.svm.predict(np.reshape(test_input,(1,15))), greater, items[greater]['name'], than, items[than]['name']
    
    def generate_ilasp_preferences(self, cardinality = 45):
        ilasp_path = '../../../ILASP/sushi/sushify_based_input.las'
        
        f = open('./base_ilasp_knowledge.txt', 'r')
        base_knowledge = f.read()
        f.close()
        
        preferences = ''
        times = cardinality
        already_added = []
        
        while times > 0:
            pred, it1, gr_name, it2, th_name = sushi.predict_random_input()
            present = False
            for x in already_added:
                if (x[0] == it1 and x[1] == it2) or (x[1] == it1 and x[0] == it2):
                    present = True

            if not present:
                print('Does '+self.get_user()+' prefer '+gr_name+'('+str(it1)+') over '+th_name+'('+str(it2)+')? '+str(pred[0]))
                if pred[0] == 1:
                    preferences = preferences + '#brave_ordering(o'+str(times)+'@1, item'+str(it1)+', item'+str(it2)+').\n'
                    already_added.append((it1, it2))
                else:
                    preferences = preferences + '#brave_ordering(o'+str(times)+'@1, item'+str(it2)+', item'+str(it1)+').\n'
                    already_added.append((it2, it1))
                times -= 1
        
        f = open(ilasp_path, 'w')
        f.write(base_knowledge)
        f.write(preferences)
        f.close()
    
    def create_new_item(self, id_it):
        minor_group = [1, 3, 5, 6, 7, 8, 9, 11]
        style = ['maki.', '']
        major_group = ['seafood.', '']
        oiliness = [1, 2, 3, 4]
        freq = [0, 1, 2]
        price = [1, 2, 3, 4]
        freq2 = [0, 1]
        
        ret = {
            'id':id_it,
            'style':r.choice(style), 
            'major_group':r.choice(major_group),
            'minor_group':r.choice(minor_group),
            'oil':r.choice(oiliness),
            'freq':r.choice(freq),
            'price':r.choice(price),
            'freq2':r.choice(freq2)
        }
        return ret
    
    def generate_in_neighborhood(self, item, N = 1000, ret = 10):
        n = N
        if ret > n:
            ret = n
        ids = item['id']+1
        new_items = []
        f_new_items = []
        items_distances = []
        f_item = self.item_to_features(item)
        while n > 0:
            n_item = self.create_new_item(ids)
            # CHANGE NEXT LINE IF YOU WANT TO INCLUDE/EXCLUDE THE ITEM ITSELF
            if n_item != item: # True
                new_item = self.item_to_features(n_item)
                new_items.append(n_item)
                f_new_items.append(new_item)
                items_distances.append(self.items_distance(f_item, new_item))
                n -= 1
                ids += 1
        #print('distance:',items_distances[0:10])
        #print('items', new_items[0:10])
        i = [x for x in range(0,N)]
        index_items = sorted(zip(items_distances, i))
        
        ordered_items = []
        f_ordered_items = []
        dist_ordered_items = []
        
        for x in index_items:
            ordered_items.append(new_items[x[1]])
            f_ordered_items.append(f_new_items[x[1]])
            dist_ordered_items.append(items_distances[x[1]])

#            print(new_items[x[1]], " ", items_distances[x[1]])
        
        #ordered_items = [it for _, it in sorted(zip(new_items, items_distances))]
        #f_ordered_items = [it for _, it in sorted(zip(f_new_items, items_distances))]

        return ordered_items[0:ret], f_ordered_items[0:ret], dist_ordered_items[0:ret]
    
    def item_to_features(self, item):
        ret = {
            'id':item['id'],
            'style':0 if item['style']=='maki.' else 1,
            'major_group':0 if item['major_group']=='seafood.' else 1,
            'minor_group':item['minor_group'],
            'oiliness':item['oil'],
            'freq_eat':item['freq'],
            'price':item['price'],
            'freq_sold':item['freq2']
        }
        return ret

    def get_svm_input(self, item1, item2):
        test_input = np.array([self.user, 
                                item1['style'], item2['style'], 
                                item1['major_group'], item2['major_group'], 
                                item1['minor_group'], item2['minor_group'], 
                                item1['oiliness'], item2['oiliness'], 
                                item1['freq_eat'], item2['freq_eat'], 
                                item1['price'], item2['price'], 
                                item1['freq_sold'], item2['freq_sold']])
        return test_input
    
    def items_distance(self, item1, item2):
        style = (3*abs(item1['style'] - item2['style']))**2
        major = (3*abs(item1['major_group'] - item2['major_group']))**2
        minor = (0 if item1['minor_group'] == item2['minor_group'] else 3)**2
        oil = abs(item1['oiliness'] - item2['oiliness'])**2
        freq = abs(item1['freq_eat'] - item2['freq_eat'])**2
        price = abs(item1['price'] - item2['price'])**2
        freq2 = abs(item1['freq_sold'] - item2['freq_sold'])**2
        distance = mt.sqrt(style + major + minor + oil + freq + price + freq2)
        return distance
    
    def ilasp_train_test(self, dim_train=10, dim_test=1, neighbor = False):
        ilasp_path = './train_ilasp_with_svm.las'
        f = open('./ilasp_items_generation.txt', 'r')
        base_knowledge = f.read()
        f.close()

        test = []

        if not neighbor:
            new_items = []
            times = 0
            dim = dim_train + dim_test

            # items generation
            pos_ilasp = ''
            while times < dim:
                i = self.create_new_item(times)
                if i not in new_items:
                    new_items.append(i)
                    generic_item = '{} {} minor_group({}). value(oil,{}). value(freq,{}). value(price,{}). value(freq2,{}).'
                    item = generic_item.format(i['style'], i['major_group'], i['minor_group'], i['oil'], i['freq'], i['price'], i['freq2'])
                    pos_ilasp = pos_ilasp + '#pos(item' + str(times) + ', {}, {}, {' + item + '}).\n'
                    times += 1
            
            # feature-ized items
            items = []
            for f in new_items[0:len(new_items)]:
                ft = self.item_to_features(f)
                items.append(ft)
            
            # ilasp brave ordering generation
            train_items_features, train_items = [], []
            if dim_train > 1:
                train_items_features = items[0:dim_train]
                train_items = new_items[0:dim_train]
            elif dim_train == 1:
                train_items_features = [items[0:dim_train]]
                train_items = [new_items[0:dim_train]]
            
            test_items = []
            if dim_test > 1:
                test_items = new_items[dim_train:dim]
            elif dim_test == 1:
                test_items = [new_items[dim_train:dim]]
            
            brave_orderings = ''
            n_bo = 0
            for i in range(0, len(train_items_features)):
                for j in range(i+1, len(train_items_features)):
                    svm_input = self.get_svm_input(train_items_features[i], train_items_features[j])
                    svm_order = self.svm.predict(np.reshape(svm_input, (1,15)))
                    if svm_order == 1:
                        order = str(train_items_features[i]['id']) + ', item' + str(train_items_features[j]['id'])
                    else:
                        order = str(train_items_features[j]['id']) + ', item' + str(train_items_features[i]['id'])
                    brave_orderings = brave_orderings + '#brave_ordering(o' + str(n_bo) + '@1, item' + order + ').\n'
                    n_bo += 1
            
            # save ilasp input
            f = open(ilasp_path, 'w')
            f.write(pos_ilasp+'\n')
            f.write(base_knowledge+'\n')
            f.write(brave_orderings)
            f.close()
            
            # building input test to compare ilasp with svm
            for i in range(0, len(test_items)):
                for j in range(0, len(train_items)):
                    svm_input = self.get_svm_input(self.item_to_features(test_items[i]), self.item_to_features(train_items[j]))
                    
                    svm_order = self.svm.predict(np.reshape(svm_input, (1,15)))
                    generic_item = '{} {} minor_group({}). value(oil,{}). value(freq,{}). value(price,{}). value(freq2,{}).'
                    
                    item_ts = generic_item.format(test_items[i]['style'], test_items[i]['major_group'], test_items[i]['minor_group'], test_items[i]['oil'], test_items[i]['freq'], test_items[i]['price'], test_items[i]['freq2'])
                    
                    item_tr = generic_item.format(train_items[j]['style'], train_items[j]['major_group'], train_items[j]['minor_group'], train_items[j]['oil'], train_items[j]['freq'], train_items[j]['price'], train_items[j]['freq2'])
                    
                    if svm_order == 1:
                        order = (item_ts, item_tr)
                    else:
                        order = (item_tr, item_ts)
                    test.append(order)
        else:
            pos_ilasp = ''
            n=1000
            test_item1 = self.create_new_item(0)
            test_item2 = self.create_new_item(n+1)

            train_items1, train_items_features1, distances1 = self.generate_in_neighborhood(test_item1, N=n, ret = int(dim_train))
            train_items2, train_items_features2, distances2 = self.generate_in_neighborhood(test_item2, N=n, ret = int(dim_train))
            generic_item = '{} {} minor_group({}). value(oil,{}). value(freq,{}). value(price,{}). value(freq2,{}).'

            for i in train_items1:
                    item1 = generic_item.format(i['style'], i['major_group'], i['minor_group'], i['oil'], i['freq'], i['price'], i['freq2'])
                    pos_ilasp = pos_ilasp + '#pos(item' + str(i['id']) + ', {}, {}, {' + item1 + '}).\n'
            for j in train_items2:
                    item2 = generic_item.format(j['style'], j['major_group'], j['minor_group'], j['oil'], j['freq'], j['price'], j['freq2'])
                    pos_ilasp = pos_ilasp + '#pos(item' + str(j['id']) + ', {}, {}, {' + item2 + '}).\n'

            item1 = generic_item.format(test_item1['style'], test_item1['major_group'], test_item1['minor_group'], test_item1['oil'], test_item1['freq'], test_item1['price'], test_item1['freq2'])
            item2 = generic_item.format(test_item2['style'], test_item2['major_group'], test_item2['minor_group'], test_item2['oil'], test_item1['freq'], test_item1['price'], test_item2['freq2'])

            pos_ilasp = pos_ilasp + '#pos(item' + str(test_item1['id']) + ', {}, {}, {' + item1 + '}).\n'
            pos_ilasp = pos_ilasp + '#pos(item' + str(test_item2['id']) + ', {}, {}, {' + item2 + '}).\n'
            
            brave_orderings = ''
            n_bo = 0
            for i in range(0, len(train_items_features1)):
                for j in range(i, len(train_items_features2)):
                    svm_input = self.get_svm_input(train_items_features1[i], train_items_features2[j])
                    svm_order = self.svm.predict(np.reshape(svm_input, (1,15)))
                    if svm_order == 1:
                        order = str(train_items_features1[i]['id']) + ', item' + str(train_items_features2[j]['id'])
                    else:
                        order = str(train_items_features2[j]['id']) + ', item' + str(train_items_features1[i]['id'])
                    brave_orderings = brave_orderings + '#brave_ordering(o' + str(n_bo) + '@' + str(self.distance_to_cost(distances1[i]+distances2[j])) + ', item' + order + ').\n'
#                    brave_orderings = brave_orderings + '#brave_ordering(o' + str(n_bo) + '@' + str(1) + ', item' + order + ').\n'
                    n_bo += 1
            
            # save ilasp input
            f = open(ilasp_path, 'w')
            f.write(pos_ilasp+'\n')
            f.write(base_knowledge+'\n')
            f.write(brave_orderings)
            f.close()
            
#            for j in range(0, len(train_items)):
#                svm_input = self.get_svm_input(self.item_to_features(test_item), self.item_to_features(train_items[j]))
#                
#                svm_order = self.svm.predict(np.reshape(svm_input, (1,15)))
#                
#                item_ts = generic_item.format(test_item['style'], test_item['major_group'], test_item['minor_group'], test_item['oil'], test_item['freq'], test_item['price'], test_item['freq2'])
#                
#                item_tr = generic_item.format(train_items[j]['style'], train_items[j]['major_group'], train_items[j]['minor_group'], train_items[j]['oil'], train_items[j]['freq'], train_items[j]['price'], train_items[j]['freq2'])
#                
#                if svm_order == 1:
#                    order = (item_ts, item_tr)
#                else:
#                    order = (item_tr, item_ts)
#                test.append(order)
            svm_input = self.get_svm_input(self.item_to_features(test_item1), self.item_to_features(test_item2))
            svm_order = self.svm.predict(np.reshape(svm_input, (1,15)))
            item_ts = generic_item.format(test_item1['style'], test_item1['major_group'], test_item1['minor_group'], test_item1['oil'], test_item1['freq'], test_item1['price'], test_item1['freq2'])
            item_tr = generic_item.format(test_item2['style'], test_item2['major_group'], test_item2['minor_group'], test_item2['oil'], test_item2['freq'], test_item2['price'], test_item2['freq2'])
            
            if svm_order == 1:
                order = (item_ts, item_tr)
            else:
                order = (item_tr, item_ts)
            test.append(order)
            
        np.save('./svm_test.npy', test)
        return test
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', type=int, default=1, help='User preferences.')
    parser.add_argument('--prefers', type=int, default=-1, help='prefers')
    parser.add_argument('--to', type=int, default=-1, help='to')
    parser.add_argument('--mode', type=str, default='prediction', help='mode')
    parser.add_argument('--data', type=str, default='USER', help='user')
    args = parser.parse_args()

    sushi = Sushifier(args.user, args.data)
    test = sushi.ilasp_train_test(neighbor=True)
    for x in test:
        print(x)

    # if args.mode == 'prediction':
    #     if args.prefers == -1 or args.to == -1:
    #         pred, greater, gr_name, than, th_name = sushi.predict_random_input()
    #     else:
    #         pred, greater, gr_name, than, th_name = sushi.predict_random_input(args.prefers, args.to)
    #     print('Does ' + sushi.get_user() + ' prefer ' + gr_name + '('+str(greater)+') over ' + th_name + '('+str(than)+')? ' + str(pred[0]))
    # elif args.mode == 'generate':
    #     sushi.generate_ilasp_preferences()
    # elif args.mode == 'run_ilasp':
    #     cmd = 'docker exec ilasp /home/ILASP --clingo "/home/clingo-4.3.0-x86_64-linux/clingo" --version=2  sushi/sushi.las'
    #     cmd = '/home/ILASP --clingo "/home/clingo-4.3.0-x86_64-linux/clingo" --version=2  sushi/sushi.las'
    #     os.system(cmd)
    # elif args.mode == 'train_user':
    #     sushi.train_model(True)
    # elif args.mode == 'train_all':
    #     sushi.train_users()
    # elif args.mode == 'evaluate':
    #     sushi.evaluate_single_user()