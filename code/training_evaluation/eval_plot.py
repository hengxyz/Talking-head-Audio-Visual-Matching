import numpy as np
import matplotlib.pyplot as plt
import os

eval_path = '/data/zming/logs/lip-reading-deeplearning/150epoch_25000train_40test_bs256_newlabels'

def eval_plot():

	EER = np.load(os.path.join(eval_path, 'EER_test_epochs.npy'))
	AUC = np.load(os.path.join(eval_path, 'AUC_test_epochs.npy'))
	AP = np.load(os.path.join(eval_path, 'AP_test_epochs.npy'))

	f = plt.figure()
	plt.plot(EER)
	plt.xlabel('epochs')
	plt.ylabel('EER')
	plt.show()
	f.savefig(os.path.join(eval_path, 'EER_eval')) 

	f = plt.figure()
	plt.plot(AUC)
	plt.xlabel('epochs')
	plt.ylabel('AUC')
	plt.show()
	f.savefig(os.path.join(eval_path, 'AUC_eval')) 

	f = plt.figure()
	plt.plot(AP)
	plt.xlabel('epochs')
	plt.ylabel('AP')
	plt.show()
	f.savefig(os.path.join(eval_path, 'AP_eval')) 


if __name__ == '__main__':
	eval_plot()