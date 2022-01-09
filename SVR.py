import warnings
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from hmmlearn import hmm
warnings.filterwarnings("ignore", category=DeprecationWarning)
NUM_TEST = 100
K = 50
NUM_ITERS=10000
p = r'C:\Users\Sanjib Kundu\PycharmProjects\StockPrediction'
ex='.csv'
e = '\\'
s = input("Enter stock name: ")
path = p+e+s
os.chdir(path)
#STOCKS = glob.glob('*.{}'.format(extension))
STOCK=input("Enter company name: ")
stock=STOCK+ex
#labels = ['4th Max','3rd Max','2nd Max','Max']
likelihood_vect = np.empty([0,1])
aic_vect = np.empty([0,1])
bic_vect = np.empty([0,1])
STATE_SPACE = range(2,5)
def calc_mape(predicted_data, true_data):
    return np.divide(np.sum(np.divide(np.absolute(predicted_data - true_data), true_data), 0), true_data.shape[0])
dataset = np.genfromtxt(stock, delimiter=',')
predicted_stock_data = np.empty([0,dataset.shape[1]])
likelihood_vect = np.empty([0,1])
aic_vect = np.empty([0,1])
bic_vect = np.empty([0,1])
for states in STATE_SPACE:
    num_params = states**2 + states
    dirichlet_params_states = np.random.randint(1,50,states)
    model = hmm.GaussianHMM(n_components=states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS)
    model.fit(dataset[NUM_TEST:,:])
    if model.monitor_.iter == NUM_ITERS:
        print('Increase number of iterations')
        sys.exit(1)
    likelihood_vect = np.vstack((likelihood_vect, model.score(dataset)))
    aic_vect = np.vstack((aic_vect, -2 * model.score(dataset) + 2 * num_params))
    bic_vect = np.vstack((bic_vect, -2 * model.score(dataset) +  num_params * np.log(dataset.shape[0])))
opt_states = np.argmin(bic_vect) + 2
print('Optimum number of states are {}'.format(opt_states))
for idx in reversed(range(NUM_TEST)):
    train_dataset = dataset[idx + 1:,:]
    test_data = dataset[idx,:]
    num_examples = train_dataset.shape[0]
    if idx == NUM_TEST - 1:
        model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS, init_params='stmc')
    else:
        model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS, init_params='')
        model.transmat_ = transmat_retune_prior
        model.startprob_ = startprob_retune_prior
        model.means_ = means_retune_prior
        model.covars_ = covars_retune_prior
    model.fit(np.flipud(train_dataset))
    transmat_retune_prior = model.transmat_
    startprob_retune_prior = model.startprob_
    means_retune_prior = model.means_
    covars_retune_prior = model.covars_
    if model.monitor_.iter == NUM_ITERS:
        print('Increase number of iterations')
        sys.exit(1)
    iters = 1
    past_likelihood = []
    curr_likelihood = model.score(np.flipud(train_dataset[0:K - 1, :]))
    while iters < num_examples / K - 1:
        past_likelihood = np.append(past_likelihood, model.score(np.flipud(train_dataset[iters:iters + K - 1, :])))
        iters = iters + 1
    likelihood_diff_idx = np.argmin(np.absolute(past_likelihood - curr_likelihood))
    predicted_change = train_dataset[likelihood_diff_idx,:] - train_dataset[likelihood_diff_idx + 1,:]
    predicted_stock_data = np.vstack((predicted_stock_data, dataset[idx + 1,:] + predicted_change))
#np.savetxt('{}_forecast.csv'.format(stock),predicted_stock_data,delimiter=',',fmt='%.2f')
mape = calc_mape(predicted_stock_data, np.flipud(dataset[range(100),:]))
print('MAPE for '+STOCK,s+' is'.format(STOCK),mape)
hdl_p = plt.plot(range(100), predicted_stock_data)
plt.title('Predicted stock prices')
plt.legend(iter(hdl_p), ('4th Max','3rd Max','2nd Max','Max'))
plt.xlabel('Time steps')
plt.ylabel('Price')
plt.grid(True)
plt.figure()
hdl_a = plt.plot(range(100),np.flipud(dataset[range(100),:]))
plt.title('Actual stock prices')
plt.legend(iter(hdl_p), ('4th Max','3rd Max','2nd Max','Max'))
plt.xlabel('Time steps')
plt.ylabel('Price')
plt.grid(True)
plt.show()