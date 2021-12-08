import pickle
from project_libs.project.models import test_and_plot_bpnn

epoch = 15
# Load models
with open(f'../data/bpnn/model_train_{epoch}.pickle', 'rb') as handle:
    model = pickle.load(handle)
with open(f'../data/bpnn/accuracies_train_{epoch}.pickle', 'rb') as handle:
    accuracies = pickle.load(handle)
with open(f'../data/bpnn/losses_train_{epoch}.pickle', 'rb') as handle:
    losses = pickle.load(handle)
with open(f'../data/bpnn/times_train_{epoch}.pickle', 'rb') as handle:
    times = pickle.load(handle)

test_and_plot_bpnn(title='_',
                   model=model,
                   accuracies=accuracies,
                   times=times,
                   losses=losses,
                   subsample=1)
