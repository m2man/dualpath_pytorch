dictionary_path = 'Flickr30k_dictionary.pickle'
dataset_path = 'Flickr30k_dataset.pickle'
numb_epochs = 3
last_epoch = 0
last_index = 0
batch_size = 32 #increase 32 for each 20 epoch, max is 256 --> need huge amount of GPU RAM
seeds = [1309 + x for x in range(numb_epochs)] # seed to generate batch
save_path = 'dualpath_torch_model/'

image_folders = ['flickr30k_images/']

stage_2 = False 
alpha = 1
lamb_0 = 0
lamb_1 = 1
lamb_2 = 1

if stage_2 == False:
  lamb_0 = 0
