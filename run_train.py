import time
import random
import datetime
from model import *
import numpy as np
import pickle
import json
import mylibrary as mylib
from tqdm import tqdm
import os
import config
import torch
torch.set_default_dtype(torch.float64)

def main():
  
  # Argument parsing
  # args = parse_arguments()
  have_cuda = torch.cuda.is_available()
  if have_cuda:
    #cudnn.benchmark = True
    device = torch.device('cuda:0')
  else:
    device = torch.device('cpu')
  print(device)

  if config.stage_2:
    log_filename = os.path.join('report/dualpath_pytorch/','dualpath_pytorch_stage_2.log')
  else:
    log_filename = os.path.join('report/dualpath_pytorch/','dualpath_pytorch_stage_1.log')
  if not os.path.exists('report/dualpath_pytorch/'):
    os.mkdir('report/dualpath_pytorch/')

  images_names = list(dataset_flickr30k.keys())

  index_flickr30k = [i for i in range(len(dataset_flickr30k))]
  index_train, index_validate, index_test = mylib.generate_3_subsets(index_flickr30k, ratio = [0.93, 0.035, 0.035])
  print("Number of samples in 3 subsets are {}, {} and {}". format(len(index_train), len(index_validate), len(index_test)))

  images_names_train = [images_names[i] for i in index_train]
  images_names_val = [images_names[i] for i in index_validate]
  images_names_test = [images_names[i] for i in index_test]

  list_dataset = []
  img_classes = [int(x[0:-4]) for x in images_names_train]
  img_classes = sorted(img_classes)
  for idx, img_name in enumerate(images_names_train):
    img_class = int(img_name[0:-4]) # remove '.jpg'
    for des in dataset_flickr30k[img_name]:
      temp = [img_name, des, img_classes.index(img_class)]
      list_dataset.append(temp)

  model = DualPath(nimages = len(images_names_train), 
                   nwords = len(my_dictionary), 
                   backbone_trainable = False, 
                   initial_word2vec = True)
  if have_cuda:
    model.to(device)

  pretrained_path = config.save_path + 'checkpoint_{}.pth'.format(config.last_epoch)
  optimizer = torch.optim.Adam(model.parameters())
  if os.path.exists(pretrained_path):
    print("Train from {}".format(pretrained_path))
    checkpoint = torch.load(pretrained_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
  else:
    print("Train form sratch")
  
  model.train()

  last_index = config.last_index
  
  # Training
  for current_epoch in range(config.last_epoch, config.numb_epochs):
    print("Generate Batch")
    batch_dataset = mylib.generate_batch_dataset(list_dataset, config.batch_size, 
                                                 seed=config.seeds[current_epoch],
                                                 fill_remain=True)

    image_loss = np.zeros(len(batch_dataset))
    text_loss = np.zeros(len(batch_dataset))
    total_loss = np.zeros(len(batch_dataset))
    if config.stage_2:
      rank_loss = np.zeros(len(batch_dataset))
      lamb_2 = config.lamb_2
    else:
      lamb_2 = 0

    print("Start Training")
    for index in tqdm(range(last_index, len(batch_dataset))):
      batch_data = batch_dataset[index]
      img_ft, txt_ft, lbl = mylib.get_feature_from_batch(batch_data, 
                                                         image_folders=config.image_folders,
                                                         dictionary=my_dictionary,
                                                         resize=224, max_len=32)
      img_ft = torch.tensor(img_ft) # in the correct axis from transform
      txt_ft = torch.tensor(txt_ft).permute(0, 3, 1, 2)
      lbl = torch.tensor(lbl, dtype=torch.long)

      if have_cuda:
        img_ft = img_ft.to(device)
        txt_ft = txt_ft.to(device)
        lbl = lbl.to(device)


      optimizer.zero_grad()
      img_class, txt_class, fimg, ftxt = model(img_ft, txt_ft)                                                   
      loss_img, loss_txt, loss_rank = loss_func(img_class, txt_class, lbl, fimg, ftxt, 
                                                lamb_0=config.lamb_0, lamb_1=config.lamb_1,
                                                lamb_2=lamb_2)
      loss = config.lamb_0*loss_img + config.lamb_1*loss_txt + lamb_2*loss_rank
      loss.backward()
      optimizer.step()

      # Track mean loss in current epoch    
      image_loss[index] = loss_img
      text_loss[index] = loss_txt
      total_loss[index] = loss_img + loss_txt  
      if config.stage_2:
        rank_loss[index] = loss_rank
        total_loss[index] += loss_rank
        info = "Epoch {}/{}: Iter batch {}/{}\nLoss_Visual: {:.6f}\nLoss_Text: {:.6f}\nLoss_Ranking: {:.6f}\nLoss_Total: {:.6f}\n-----".format(current_epoch+1, config.numb_epochs, 
                                                                                                                                                index+1, len(batch_dataset),
                                                                                                                                                np.mean(image_loss[last_index:index+1]),
                                                                                                                                                np.mean(text_loss[last_index:index+1]),
                                                                                                                                                np.mean(rank_loss[last_index:index+1]),
                                                                                                                                                np.mean(total_loss[last_index:index+1]))
      else:
        info = "Epoch {}/{}: Iter batch {}/{}\nLoss_Visual: {:.6f}\nLoss_Text: {:.6f}\nLoss_Total: {:.6f}\n-----".format(current_epoch+1, config.numb_epochs, 
                                                                                                                        index+1, len(batch_dataset),
                                                                                                                        np.mean(image_loss[last_index:index+1]),
                                                                                                                        np.mean(text_loss[last_index:index+1]),
                                                                                                                        np.mean(total_loss[last_index:index+1]))
  
      if (index+1) % 20 == 0 or index <= 9:
          print(info)
      if (index+1) % 20 == 0:
        with open(log_filename, 'a') as f:
          f.write('{}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')))
          f.write(info+'\n')
        print("Saving model ...")
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                  }, config.save_path + 'checkpoint_{}.pth'.format(current_epoch))
            
    last_index = 0
    print(info)
    with open(log_filename, 'a') as f:
      f.write('{}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')))
      f.write(info+'\n')

    # Save
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
              }, config.save_path + 'checkpoint_{}.pth'.format(current_epoch))

if __name__ == "__main__":
  main()
