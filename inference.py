import os
import torch
import theconf
from theconf import Config as C
import logging
import sys
from sklearn.metrics import classification_report
from tqdm import tqdm
import time

import trainer
from train import evaluation

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

def infer(flags):
    device = torch.device(type= 'cuda', index=0)

    #basic checkpoint
    
    model = trainer.model.create(C.get()['architecture'])
    model.to(device=device, non_blocking=True)
    try:
        checkpoint = torch.load(C.get()['inference']['checkpoint'], map_location=device)['model']
        for key in list(checkpoint.keys()):
            if 'module.' in key:
                checkpoint[key.replace('module.', '')] = checkpoint[key]
                del checkpoint[key]
        model.load_state_dict(checkpoint)
        

    except Exception as e:
        raise (e)
    
    #quantization only cpu
    #model = torch.jit.load(C.get()['inference']['quantization'])
    #logging.info(f"[Model] | Load from {C.get()['inference']['quantization']}")
    #model = torch.jit.load(checkpoint)

    #pruning
    #logging.info(f"[Model] | Load from {C.get()['inference']['prune']}")
    #model = torch.load(C.get()['inference']['prune'])
    #model.to(device=device, non_blocking=True)

    
    
    
    

    test_loader, _ = trainer.dataset.create(C.get()['dataset'],
                                                    mode='test')

    # logging.info(f'[Dataset] | test_examples: {len(test_loader)}')
    # criterion = trainer.loss.create(C.get()['loss']).to(device=device, non_blocking=True)

    model.eval()
    inference(model, test_loader, device, flags)

   

@torch.no_grad()
def inference(model, dataloader, device, flags):
    one_epoch_loss = 0
    test_total = 0
    test_hit = 0
    
    tp, tn, fp, fn = 0, 0, 0, 0

    y_true_total = []
    y_pred_total = []

    for step, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Test", disable=not flags.is_master):
        image = image.to(device=device, non_blocking=True)
        label = label.to(device=device, non_blocking=True)

        y_pred = model(image)             
        _, y_pred = y_pred.max(1)
        y_true_total.extend(label.cpu().tolist())
        y_pred_total.extend(y_pred.cpu().tolist())

        test_hit += y_pred.detach().eq(label).sum()
        test_total += image.shape[0]

        for i in range(y_pred.shape[0]):
            if y_pred[i] == 0:
                if label[i] == 0:
                    tp += 1
                else:
                    fp += 1
            else:
                if label[i] == 0:
                    fn += 1
                else:
                    tn += 1        

    if flags.is_master:
        logging.info(f'[Test] Test Case : {test_total} Acc: {test_hit / test_total}')
        logging.info(f'[Tailing] TP: {tp} FP: {fp} FN: {fn} TN: {tn}')
        print(f'{classification_report(y_true_total, y_pred_total, target_names=["tailing", "normal"], digits=4)}')
  
    return one_epoch_loss / len(dataloader), test_hit / test_total, classification_report(y_true_total, y_pred_total, target_names=["tailing", "normal"])

if __name__ == '__main__':
    parser = theconf.ConfigArgumentParser(conflict_handler='resolve')
    flags = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', stream=sys.stderr)
    
    flags.is_master = True
    infer(flags)