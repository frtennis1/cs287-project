from tqdm import tqdm_notebook, tqdm
import numpy as np
from util_funcs import *
from data_processors import *
from callback import *

def _null_progress_bar(x, **kwargs):
    return x

def get_progress_bar(option=None):
    if option == "notebook":
        progress_bar = tqdm_notebook
    elif option == "text":
        progress_bar = tqdm
    else:
        progress_bar = _null_progress_bar
    return progress_bar

class Trainer:
    def __init__(self, model, data, loss_fn, optimizer, output_dir, num_labels,
                 writer=None,
                 val_data=None, 
                 device=torch.device("cuda"),
                 **kwargs):
        self.model = model
        self.data = data        
        self.val_data = val_data
        self.tb_writer = writer
        self.output_dir = output_dir
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.num_labels = num_labels
        
    def train_loop(self, *args, **kwargs):
        raise NotImplementedError
        
    def validate(self, *args, **kwargs):
        raise NotImplementedError
        
    def report(self, report_validation, **kwargs):
        writer_callback(self.counter, self.total_period_loss / self.report_frequency,
                        self.tb_writer, run_name=self.output_dir.replace("/", "_"))
        if report_validation:
            writer_callback(self.counter, self.total_period_val_loss / self.report_frequency,
                            self.tb_writer, run_name=self.output_dir.replace("/", "_"), variable="validation_loss")
            writer_callback(self.counter, self.total_period_val_acc,
                            self.tb_writer, run_name=self.output_dir.replace("/", "_"), variable="validation_accuracy")
        self.total_period_loss = 0
        self.total_period_val_loss = 0
        
    def train(self, num_train_epochs,
              progress_bar="notebook",
              patience=2, **kwargs):
        self.counter = 0        
        self.total_period_loss = 0
        self.total_period_val_loss = 0
        device = self.device
        opt_progress_bar = progress_bar
        progress_bar = get_progress_bar(option=progress_bar)
        best_val_loss = np.inf
        best_state_dict = None
        best_epoch = 0
        for epoch in progress_bar(range(num_train_epochs), desc="Epoch"):
            try:
                val_loss = self.train_loop(progress_bar=opt_progress_bar, **kwargs)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = self.model.cpu().state_dict()
                    self.model.to(device)
                    best_epoch = epoch
                elif epoch > best_epoch + patience:
                    self.model.cpu().load_state_dict(best_state_dict)
                    self.model.to(device)
                    return self.model
                else:
                    pass

            except KeyboardInterrupt:
                break
        save_model(self.model, self.output_dir)
        if self.tb_writer is not None:
            self.tb_writer.close()

            
class DeepTwistTrainer(Trainer):
    def __init__(self, model, data, 
                 output_dir, num_labels,
                 twist_frequency,
                 loss_fn, optimizer,
                 distort,
                 twist_args=dict(),
                 writer=None,
                 val_data=None, 
                 device=torch.device("cuda"),
                 **kwargs):
        super().__init__(model, data, loss_fn, optimizer, output_dir, num_labels,
                         writer=writer,
                         val_data=val_data, 
                         device=torch.device("cuda"),
                         **kwargs)
        self.twist_frequency = twist_frequency
        self.distort = distort 
        self.twist_args = twist_args
        self.train_opts = kwargs
        
        
    def train_loop(self, *args, progress_bar='notebook', report_frequency=5, report_validation=True, **kwargs):
        opt_progress_bar = progress_bar
        self.report_frequency = report_frequency
        num_labels = self.num_labels
        device = self.device
        self.model.train()
        gradient_accumulation_steps = self.train_opts['gradient_accumulation_steps']
        
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        progress_bar = get_progress_bar(option=progress_bar)
        
        for step, batch in enumerate(progress_bar(self.data, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            
            # define a new function to compute loss values for both output_modes
            logits = self.model(input_ids, segment_ids, input_mask, labels=None)
            
            # No regression support
            loss = self.loss_fn(logits.view(-1, num_labels), label_ids.view(-1))
                        
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()            
            
            if self.tb_writer is not None and self.counter >= report_frequency and self.counter % report_frequency == 0:
                self.total_period_loss += loss.item()
                if report_validation:
                    self.total_period_val_loss, self.total_period_val_acc = self.validate()
                self.report(report_validation, **kwargs)
            self.counter += 1

                
            # Deep twist
            if self.counter % self.twist_frequency == 0 and self.counter > 0 and self.twist_frequency > 0:
                state_dict = self.distort(self.model.cpu().state_dict(), **self.twist_args)
                self.model.load_state_dict(state_dict)
                self.model.cuda()
        
        # Twist at epoch end 
        state_dict = self.distort(self.model.cpu().state_dict(), **self.twist_args)
        self.model.load_state_dict(state_dict)
        self.model.cuda()
        return self.validate()[0]
    
    def validate(self, progress_bar=None, **kwargs):
        num_labels = self.num_labels
        progress_bar = get_progress_bar(progress_bar)
        device = self.device
        self.model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        correct = 0
        total = 0
        
        if self.val_data is None:
            return None
        for step, batch in enumerate(progress_bar(self.val_data, desc="Validation", leave=False)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            
            # define a new function to compute loss values for both output_modes
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask, labels=None)
            
            # No regression support
            loss = self.loss_fn(logits.view(-1, num_labels), label_ids.view(-1))
            eval_loss += loss.mean().item()   
            nb_eval_steps += 1
            
            correct += (np.argmax(logits.detach().cpu().numpy(), axis=1) == 
                        label_ids.detach().cpu().numpy().flatten()).sum()
            total += len(label_ids)
            
        eval_loss = eval_loss / nb_eval_steps
        return eval_loss, correct / total
        self.model.train()
            
        