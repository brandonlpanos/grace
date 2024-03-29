import os
import torch
import pandas 
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import rcParams
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torchvision.transforms as T
from matplotlib.ticker import MultipleLocator

'''
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
Code trains a Mutual Information Neural Estimator (MINE-network) on each month of GRACE data.
Returns monthly csv files with a new column for the point-wise mutual information,
associated training curve, as well as the saved model. 

File structure
--------------
- datasets 
    -2019-01 --> res files per day
    -2019-02 --> res files per day
    -2019-12 --> res files per day
- midata
- models
- plots
- grace_pmi.py

 PMI can take positive or negative values, but is zero if X and Y are independent. 
 Note that even though PMI may be negative or positive, its expected outcome over 
 all joint events (MI) is positive. (wiki)
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
'''


def res_file_to_df(file):
    '''
    converts a res file into a pandas DataFrame for a single day
    '''
    # header information
    columns = ['MJD', 
               'frac of a day', 
               'GPS range AB [m]', 
               'GPS range rate AB [m/s]', 
               'Kband range [m]',
               'Kband range rate [m/s]',
               'O-C range rate [m/s]',
               'Latitude [deg]',
               'Longitude [deg]',
               'Arg. of lat. [deg]',
               'beta [deg]']
    # collect data
    data_array = [] # empty container 
    with open(file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i < 12: continue # skip header information
            line_entries = line.split(' ') # split lines with space delimiter
            line_entries = np.array(line_entries) # convert to numpy array
            line_entries = np.delete(line_entries, np.argwhere(line_entries=='')).astype(float) # convert to floats
            data_array.append(line_entries) # collect vectors
    data_array = np.vstack(data_array) # stack vectors into an array 
    df = pandas.DataFrame(data=data_array,columns=columns) # convert stack to pandas DataFrame
    return df

class MINEDataLoader(torch.utils.data.Dataset):
    '''
    data loader that generates samples of joints and marginals 
    '''
    def __init__(self, data1, data2, n_samples):
        self.n_samples = n_samples
        self.data1 = torch.from_numpy(data1).type(torch.FloatTensor)
        self.data2  = torch.from_numpy(data2).type(torch.FloatTensor)

    def __len__(self):
        return len(self.data2)

    def __getitem__(self, indx):
        rand_index = np.random.randint(0, len(self.data1)-1, self.n_samples)
        x_sample = self.data1[rand_index]
        y_sample = self.data2[rand_index]
        y_shuffle = self.data2[np.random.randint(0, len(self.data2)-1, len(y_sample))]
        return torch.squeeze(x_sample), torch.unsqueeze(y_sample, -1), torch.unsqueeze(y_shuffle, -1)

class MINEnetwork(nn.Module):
    '''
    MINE-network using PyTorch
    '''
    def __init__(self, d1, dz):
        super(MINEnetwork, self).__init__()
        self.fc1 = nn.Linear(d1, dz)
        self.fc2 = nn.Linear(dz + 1, 2)
        self.fc3 = nn.Linear(2, 1)

    def forward(self, x, y):
        x = self.fc1(x)
        x = F.relu(x)
        out = torch.cat((x, y), -1)
        out = self.fc2(out) 
        out = F.relu(out)
        out = self.fc3(out)
        return torch.squeeze(out) # output single scalar

def PMI(x_sample, y_sample, y_shuffle):
    '''
    returns the pmi of a single data pair
    '''
    with torch.no_grad():            
        # Calculate joint output
        pred_xy = model(x_sample, y_sample)
        D_xy = pred_xy
        # Calculate marginal output
        pred_x_y = model(x_sample, y_shuffle)
        D_x_y = torch.log(torch.mean(torch.exp(pred_x_y)))
        # Calculate pmi
        pmi = (D_xy[0] - D_x_y)
        pmi = pmi.numpy()
        return pmi


if __name__ ==  '__main__':

    # iterate over data in months
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    for month in months:
        if month == '01': continue
        print(month)
        X = None 
        y = None
        df_month = None
        # if month != '05': continue
        year = '2019'
        root_path  = f'/Users/brandonlpanos/Desktop/grace/datasets/{year}-{month}'
        for day in os.listdir(root_path):
            path = root_path + '/' + day
            try: df = res_file_to_df(path) # convert res file into pandas DataFrame
            except:
                print(path + ' is empty') 
                continue # some datasets are empty, if any problems just skip        
            y_day = np.array(df['O-C range rate [m/s]']) # target random variable residuals 
            df2 = df.copy()
            df2 = df2.drop( ['MJD', 
               'frac of a day', 
               'GPS range rate AB [m/s]', 
               'Kband range rate [m/s]',
               'O-C range rate [m/s]',
               'beta [deg]'], axis=1) # drop target variable from df
            
            try: df_month = pandas.concat([df_month, df], ignore_index=True)
            except: 
                print('here')
                df_month = df 
            
            # concatenate x matrices
            X_day = df2.to_numpy() # construct matrix out of remaining columns
            try: X = np.concatenate( (X, X_day), axis=0 )
            except: X = X_day

            # concatenate y vectors 
            try: y = np.concatenate( (y, y_day) )
            except: y = y_day

            print(len(df_month), X.shape, y.shape)

            # error catching
            assert len(X) == len(y), f'{path} dimensions dont match'
            # if month != '01' and X <= X_day: print(path + ' is broken and reset the dataset, matrix issue')
            # if month != '01' and df_month <= df: print(path + ' is broken and reset the dataset, dataframe issue')

        # standerdize
        X_scal = preprocessing.StandardScaler().fit_transform(X)
        y_scal = preprocessing.StandardScaler().fit_transform(y.reshape(len(y),1))
        y_scal = np.squeeze(y_scal)

        dataset = MINEDataLoader(X_scal, y_scal, 1000)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)

        n_epoch = 10000
        model = MINEnetwork(5,3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        mi_hist = []
        for epoch in range(n_epoch):
            x_sample, y_sample, y_shuffle = next(iter(dataloader))
            pred_xy = model(x_sample, y_sample)
            pred_x_y = model(x_sample, y_shuffle)
            loss = - (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))
            mi_hist.append(-1 * loss.data.numpy())
            model.zero_grad()
            loss.backward()
            optimizer.step()

        # save model
        root_to_save_model = '/Users/brandonlpanos/Desktop/grace/models/'
        torch.save(model, f'{root_to_save_model}/{year}_{month}.pt')

        # plot results
        fig, ax = plt.subplots(figsize=(17,5))
        rcParams['font.size'] = 14
        plt.title('Mutual Information learning curves for grace residuals using the neural estimator (MINE)', fontsize=18)
        from scipy.signal import savgol_filter
        mi_smoother = savgol_filter(mi_hist, 200, 3) # window size 200, polynomial order 3
        plt.plot(mi_smoother, c='k', label='post-fit residuals vs orbital location')
        plt.axhline(np.mean(mi_hist[1000::]), c='#8dd8f8', linestyle='--', linewidth=2, label='converged MI')
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(0.005))
        ax.yaxis.set_minor_locator(MultipleLocator(0.001))
        ax.tick_params(which='major', length=10,width=1)
        ax.tick_params(which='minor', length=7,width=1)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('MI', fontsize=18)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'/Users/brandonlpanos/Desktop/grace/plots/{year}_{month}.pdf')
        plt.close(fig)

        # calculate PMI 
        months_pmis = []
        for x, y in zip(X_scal,y_scal):
            _, _, y_shuffle = next(iter(dataloader))
            y = y.reshape(1, 1, 1)
            y = torch.Tensor(y).type(torch.FloatTensor)
            y = y.repeat(1, y_shuffle.shape[1], 1)
            x = x.reshape(1, 1, x.shape[0])
            x = torch.Tensor(x).type(torch.FloatTensor)
            x = x.repeat(1, y_shuffle.shape[1], 1)
            pmi = PMI(x, y, y_shuffle)
            months_pmis.append(pmi)
        months_pmis = np.array(months_pmis)

        # append new information into csv file for the month
        assert len(months_pmis) == len(df_month), f'{path} dimensions dont match, pmi issue'
        df_month['MI'] = months_pmis
        save_new_csv = '/Users/brandonlpanos/Desktop/grace/midata/'
        df_month.to_csv(f'{save_new_csv}/{year}_{month}.csv',sep='\t')
