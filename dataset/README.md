from dataset import get_dataset, update_dataset

loaders, datasets = get_dataset(opt)
val_loader_iter = iter(loaders['val'])


# Update parts of the train loader, e.g., 200 samples, this update will replace half of the sample in the loader, i.e., 100 samples.
loaders['train'] = update_dataset(opt, datasets['train'], loaders['train'])

