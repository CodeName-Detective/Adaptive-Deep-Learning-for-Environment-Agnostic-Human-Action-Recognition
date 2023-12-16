import os
import torch
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy

# Modules for Distributed Data Paralleling
from torch.multiprocessing import spawn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from .data_loader import UCF101_Dataset_Loader

if torch.cuda.is_available():
    DEVICE = torch.device(device='cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device(device='mps')
else:
    DEVICE = torch.device(device='cpu')


class Single_Core_Trainer:
    def __init__(self: 'Single_Core_Trainer', num_classes: int):
        self.loss_criteria = torch.nn.CrossEntropyLoss()
        self.multiclass_accuracy  = MulticlassAccuracy(num_classes=num_classes).to(device=DEVICE)
    
    def _one_epoch_train(self: 'Single_Core_Trainer', model: torch.nn.Module, data_loader_train: torch.utils.data.DataLoader, 
                         optim_alog: torch.optim) -> tuple:
        """Function that trains the model for one epoch.

        Args:
            model (torch.nn.Module): Pytorch model we want to train.
            data_loader_train (torch.utils.data.DataLoader): Pytorch dataloader that carries training data.
            optim_alog (torch.optim): Opimiztion algoritham that we use to update model weights.

        Returns:
            tuple: Output tensor carrying predicted probability of each class.
        """
        batch_loss_train = []
        batch_accuracy_train = []
        batch_counter = 0
        for inputs, labels in tqdm(data_loader_train):
            inputs = inputs.to(device=DEVICE)
            labels = labels.to(device=DEVICE)
            
            # Enabling model training.
            model.train(True)
            
            #Setting gradients to zero to prevent gradient accumulation.
            optim_alog.zero_grad()
            
            # Forward pass.
            y_pred_prob = model(inputs)
            loss = self.loss_criterion(y_pred_prob, labels)
            
            batch_loss_train.append(loss.item())
            
            # Back Propagation
            loss.backward()
            
            # Updating weights
            optim_alog.step()
            
            # Calculating training accuracy.
            with torch.inference_mode():
                accuracy = self.multiclass_accuracy(y_pred_prob, labels)
                batch_accuracy_train.append(accuracy.item())
            batch_counter += 1
            
            del(inputs)
            del(labels)
            
        return sum(batch_loss_train)/batch_counter, sum(batch_accuracy_train)/batch_counter
    
    def training_loop(self: 'Single_Core_Trainer', model: torch.nn.Module, data_loader_train: torch.utils.data.DataLoader, data_loader_test: torch.utils.data.DataLoader, 
                  epochs:int, optim_alog: torch.optim, learning_rate_scheduler:torch.optim =None)-> dict:
        """Function that trains the model for the given number of epochs

        Args:
            model (torch.nn.Module): Pytorch model we want to train.
            data_loader_train (torch.utils.data.DataLoader): Pytorch dataloader that carries training data.
            data_loader_test (torch.utils.data.DataLoader): Pytorch dataloader that carries testing data.
            epochs (int): Count of EPOCHS
            optim_alog (torch.optim): Opimiztion algoritham that we use to update model weights.
            learning_rate_scheduler (torch.optim, optional): Learning rate scheduler to decrease the learning rate. Defaults to None.

        Returns:
            dict: A dictionary that carries the output metrics.
        """
        
        loss_train = []
        loss_test = []
        
        accuracy_train = []
        accuracy_test = []
        
        # Loop that iterates over each EPOCH
        for epoch in range(epochs):
            
            #Train the model for one EPOCH
            epoch_loss, epoch_accuracy = self._one_epoch_train(model, data_loader_train, optim_alog)
            loss_train.append(epoch_loss)
            accuracy_train.append(epoch_accuracy)
            
            model.train(False)
            # Making a forward pass of Test data
            batch_loss_test = []
            batch_accuracy_test = []
            batch_counter = 0
            with torch.inference_mode():
                for inputs, labels in tqdm(data_loader_test):
                    inputs = inputs.to(device=DEVICE)
                    labels = labels.to(device=DEVICE)
                    y_pred_prob = model(inputs)
                    #Calculate the test loss.
                    loss_batch = self.loss_criteria(y_pred_prob, labels)
                    batch_loss_test.append(loss_batch.item())
                    # Calculate Test Accuracy.
                    accuracy_batch = self.multiclass_accuracy(y_pred_prob, labels)
                    batch_accuracy_test.append(accuracy_batch.item())
                    batch_counter += 1
                    del(inputs)
                    del(labels)
            loss = sum(batch_loss_test)/batch_counter
            accuracy = sum(batch_accuracy_test)/batch_counter
            loss_test.append(loss)
            accuracy_test.append(accuracy)
            
            if (epoch+1)%5 == 0:
                print('For Epoch {} We Train Loss:{}, Test Loss:{}, Train Accuracy:{}, Test Accuracy:{}'.format(epoch+1, epoch_loss,
                                                                                                            loss,
                                                                                                            epoch_accuracy,
                                                                                                            accuracy))
        return {'training_loss':loss_train, 'testing_loss':loss_test, 'training_accuracy':accuracy_train, 'testing_accuracy':accuracy_test}
 

# Mixed Precision Training Using Single Core GPU        
class Mixed_Precision_Single_Core_Trainer(Single_Core_Trainer):
    def __init__(self: 'Mixed_Precision_Single_Core_Trainer',num_classes: int):
        super(Mixed_Precision_Single_Core_Trainer, self).__init__(num_classes)
        
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available. A CUDA-compatible device is required.')
        else:
            #  Maximum size of a tensor that can be split during a parallel operation on the GPU.
            torch.backends.cuda.max_split_size = 512
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _one_epoch_train(self: 'Mixed_Precision_Single_Core_Trainer', model: torch.nn.Module, data_loader_train: torch.utils.data.DataLoader, 
                         optim_alog: torch.optim) -> tuple:
        """Function that trains the model for one epoch.

        Args:
            model (torch.nn.Module): Pytorch model we want to train.
            data_loader_train (torch.utils.data.DataLoader): Pytorch dataloader that carries training data.
            loss_criterion (torch.nn): Pytorch loss criteria on which we calculate loss.
            optim_alog (torch.optim): Opimiztion algoritham that we use to update model weights.

        Returns:
            tuple: Output tensor carrying predicted probability of each class.
        """
        batch_loss_train = []
        batch_accuracy_train = []
        batch_counter = 0
        for inputs, labels in tqdm(data_loader_train):
            inputs = inputs.to(device=DEVICE)
            labels = labels.to(device=DEVICE)
            
            # Enabling model training.
            model.train(True)
            
            #Setting gradients to zero to prevent gradient accumulation.
            optim_alog.zero_grad()
            
            # Forward pass with Mixed Precision
            with torch.cuda.amp.autocast():
                y_pred_prob = model(inputs)
                loss = self.loss_criteria(y_pred_prob, labels)
            
            batch_loss_train.append(loss.item())
            
            # Back Propagation with Mixed Precision
            self.scaler.scale(loss).backward()
            
            # Updating weights with Mixed Precision
            self.scaler.step(optim_alog)
            self.scaler.update()
            
            # Calculating training accuracy.
            with torch.inference_mode():
                accuracy = self.multiclass_accuracy(y_pred_prob, labels)
                batch_accuracy_train.append(accuracy.item())
            batch_counter += 1
            
            del(inputs)
            del(labels)
            
        return sum(batch_loss_train)/batch_counter, sum(batch_accuracy_train)/batch_counter
    
    
    def training_loop(self: 'Mixed_Precision_Single_Core_Trainer', model: torch.nn.Module, data_loader_train: torch.utils.data.DataLoader, data_loader_test: torch.utils.data.DataLoader, 
                  epochs:int, optim_alog: torch.optim, learning_rate_scheduler:torch.optim =None)-> dict:
        """Function that trains the model for the given number of epochs

        Args:
            model (torch.nn.Module): Pytorch model we want to train.
            data_loader_train (torch.utils.data.DataLoader): Pytorch dataloader that carries training data.
            data_loader_test (torch.utils.data.DataLoader): Pytorch dataloader that carries testing data.
            epochs (int): Count of EPOCHS
            optim_alog (torch.optim): Opimiztion algoritham that we use to update model weights.
            learning_rate_scheduler (torch.optim, optional): Learning rate scheduler to decrease the learning rate. Defaults to None.

        Returns:
            dict: A dictionary that carries the output metrics.
        """
    
        loss_train = []
        loss_test = []
        
        accuracy_train = []
        accuracy_test = []
        
        # Loop that iterates over each EPOCH
        for epoch in range(epochs):
            
            #Train the model for one EPOCH
            epoch_loss, epoch_accuracy = self._one_epoch_train(model, data_loader_train, optim_alog)
            loss_train.append(epoch_loss)
            accuracy_train.append(epoch_accuracy)
            
            model.train(False)
            model.train(False)
            # Making a forward pass of Test data
            batch_loss_test = []
            batch_accuracy_test = []
            batch_counter = 0
            with torch.inference_mode():
                for inputs, labels in tqdm(data_loader_test):
                    inputs = inputs.to(device=DEVICE)
                    labels = labels.to(device=DEVICE)
                    y_pred_prob = model(inputs)
                    #Calculate the test loss.
                    loss_batch = self.loss_criteria(y_pred_prob, labels)
                    batch_loss_test.append(loss_batch.item())
                    # Calculate Test Accuracy.
                    accuracy_batch = self.multiclass_accuracy(y_pred_prob, labels)
                    batch_accuracy_test.append(accuracy_batch.item())
                    batch_counter += 1
                    del(inputs)
                    del(labels)
            loss = sum(batch_loss_test)/batch_counter
            accuracy = sum(batch_accuracy_test)/batch_counter
            loss_test.append(loss)
            accuracy_test.append(accuracy)
            
            if (epoch+1)%1 == 0:
                print('For Epoch {} We Train Loss:{}, Test Loss:{}, Train Accuracy:{}, Test Accuracy:{}'.format(epoch+1, epoch_loss,
                                                                                                            loss,
                                                                                                            epoch_accuracy,
                                                                                                            accuracy))
        return {'training_loss': loss_train, 'testing_loss': loss_test, 'training_accuracy': accuracy_train, 'testing_accuracy': accuracy_test}
    
    

# Data Parallel Distributed and Mixed Precision Training 
class Data_Parallel_Distributed_Trainer:
    def __init__(self, num_classes, model_name, case):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available. A CUDA-compatible device is required.')
        
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.multiclass_accuracy = MulticlassAccuracy(num_classes=num_classes).to(device='cuda')
        self.world_size = torch.cuda.device_count()
        self.model_name = model_name
        self.case = case
        self.scaler = GradScaler()  # For mixed precision training
    
    def _ddp_setup(self, rank):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='nccl', rank=rank, world_size=self.world_size)
        torch.cuda.set_device(rank)
    
    def _one_epoch_train(self, model, data_loader_train, optim_alg, rank, epoch):
        model.train()
        total_loss, total_accuracy = 0.0, 0.0
        for batch_idx, (inputs, labels) in enumerate(data_loader_train):
            inputs, labels = inputs.to(rank), labels.to(rank)

            optim_alg.zero_grad()
            with autocast():  # Mixed precision
                y_pred_prob = model(inputs)
                loss = self.loss_criterion(y_pred_prob, labels)

            self.scaler.scale(loss).backward()  # Scaled backpropagation
            self.scaler.step(optim_alg)         # Scaled optimizer step
            self.scaler.update()                # Update the scale for next iteration

            total_loss += loss.item()
            total_accuracy += self.multiclass_accuracy(y_pred_prob, labels).item()
            
            del(inputs)
            del(labels)

        avg_loss = total_loss / len(data_loader_train)
        avg_accuracy = total_accuracy / len(data_loader_train)
        return avg_loss, avg_accuracy
    
    def _training_loop(self: 'Data_Parallel_Distributed_Trainer', model: torch.nn.Module, data_loader_train: torch.utils.data.DataLoader, data_loader_test: torch.utils.data.DataLoader, 
                  epochs:int, optim_alog: torch.optim, gpu_id)-> dict:
        """Function that trains the model for the given number of epochs

        Args:
            model (torch.nn.Module): Pytorch model we want to train.
            data_loader_train (torch.utils.data.DataLoader): Pytorch dataloader that carries training data.
            data_loader_test (torch.utils.data.DataLoader): Pytorch dataloader that carries testing data.
            epochs (int): Count of EPOCHS
            optim_alog (torch.optim): Opimiztion algoritham that we use to update model weights.
            learning_rate_scheduler (torch.optim, optional): Learning rate scheduler to decrease the learning rate. Defaults to None.

        Returns:
            dict: A dictionary that carries the output metrics.
        """
        
        loss_train = []
        loss_test = []
        
        accuracy_train = []
        accuracy_test = []
        
        model = DDP(model, device_ids=[gpu_id])
        
        # Loop that iterates over each EPOCH
        for epoch in range(epochs):
            
            #Train the model for one EPOCH
            epoch_loss, epoch_accuracy = self._one_epoch_train(model, data_loader_train, optim_alog, epoch)
            loss_train.append(epoch_loss)
            accuracy_train.append(epoch_accuracy)
            
            if  gpu_id == 0:
                 
                batch_loss_test = 0
                batch_accuracy_test = 0
                model.train(False)
                for inputs, labels in data_loader_test:
                    inputs = inputs.to(device=gpu_id)
                    labels = labels.to(device=gpu_id)
                    
                    # Disabling model training.
                    model.train(False)  # Alternatively, you can use model.eval()

                    # Forward pass.
                    with torch.inference_mode():  # Disable gradient computation for efficiency
                        y_pred_prob = model(inputs)
                        loss = self.loss_criterion(y_pred_prob, labels)

                        batch_loss_test += loss.item()
                        accuracy = self.multiclass_accuracy(y_pred_prob, labels)
                        batch_accuracy_test += accuracy.item()
                    
                    del(inputs)
                    del(labels)
                    
                epoch_loss_test = batch_loss_test/len(data_loader_test) 
                epoch_accuracy_test =  batch_accuracy_test/len(data_loader_test)
                loss_test.append(epoch_loss_test)
                accuracy_test.append(epoch_accuracy_test)
                
                if (epoch+1)%1 == 0:
                    print('For Epoch {} We Train Loss:{}, Test Loss:{}, Train Accuracy:{}, Test Accuracy:{}'.format(epoch+1, epoch_loss,
                                                                                                            epoch_loss_test,
                                                                                                            epoch_accuracy,
                                                                                                            epoch_accuracy_test))
                    self._save_checkpoint(model, epoch+1)
                
        return model, {'training_loss':loss_train, 'testing_loss':loss_test, 'training_accuracy':accuracy_train, 'testing_accuracy':accuracy_test}
    
    def _save_checkpoint(self, model, epoch):
        ckp = model.module.state_dict()
        PATH = "models/"+self.model_name+"_"+self.case+"_"+"checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def _main(self, model_class, epochs, optim_alg, rank, ucf_data_dir, ucf_label_dir, frames_per_clip, step_between_clips, height, width, batch_size):
        self._ddp_setup(rank)
        ucf101 = UCF101_Dataset_Loader(ucf_data_dir, ucf_label_dir, frames_per_clip, step_between_clips, height, width, batch_size, data_parallel_distributed_training=True)
        model = model_class((frames_per_clip, height, width), len(ucf101.classes))
        model = DDP(model.to(rank), device_ids=[rank])
        self._training_loop(model, ucf101.train_loader, ucf101.test_loader, epochs, optim_alg, rank)
        dist.destroy_process_group()

    def trainer(self, model_class, epochs, optim_alg, ucf_data_dir, ucf_label_dir, frames_per_clip, step_between_clips, height, width, batch_size):
        spawn(self._main, args=(model_class, epochs, optim_alg, ucf_data_dir, ucf_label_dir, frames_per_clip, step_between_clips, height, width, batch_size), nprocs=self.world_size)
                       