import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



# Set device: use MPS if available, otherwise fallback to CPU or CUDA
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TimeSeriesSeqDataset(Dataset):
    def __init__(self, symptoms, inputs, seq_len_hist, seq_len_future):
        """
        Args:
            symptoms (np.ndarray or torch.Tensor): Array of shape (N_days*time_step, 2).
            inputs (np.ndarray or torch.Tensor): Array of shape (N_days*time_step, 2).
            seq_len_hist (int): Number of timesteps for the history.
            seq_len_future (int): Number of timesteps for the future target.
            
        This dataset slides a window over the time-series data.
        For each sample, it returns:
          - hist_symptoms: symptoms for the past period (shape: (seq_len_hist, 2))
          - combined_inputs: inputs for history+future (shape: (seq_len_hist + seq_len_future, 2))
          - target: future symptoms (shape: (seq_len_future, 2))
        """
        # Convert to torch tensors if not already
        if isinstance(symptoms, np.ndarray):
            self.symptoms = torch.from_numpy(symptoms)
        else:
            self.symptoms = symptoms

        if isinstance(inputs, np.ndarray):
            self.inputs = torch.from_numpy(inputs)
        else:
            self.inputs = inputs
        
        self.seq_len_hist = seq_len_hist
        self.seq_len_future = seq_len_future
        self.total_seq_len = seq_len_hist + seq_len_future
        
        # The number of samples is the number of sliding windows which can be extracted.
        self.length = self.symptoms.shape[0] - self.total_seq_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # idx runs from 0 to (total_data_len - total_seq_len)
        # Extract historical symptoms
        hist_symptoms = self.symptoms[idx : idx + self.seq_len_hist]  # shape: (seq_len_hist, 2)
        # Extract combined inputs (history + future)
        combined_inputs = self.inputs[idx : idx + self.total_seq_len]  # shape: (seq_len_hist + seq_len_future, 2)
        # Extract future target symptoms
        target = self.symptoms[idx + self.seq_len_hist : idx + self.total_seq_len]  # shape: (seq_len_future, 2)
        return hist_symptoms, combined_inputs, target

    
class SeqClassifier(nn.Module):
    def __init__(self, 
                 symptom_vocab_size=10,   # classes 0-9 representing original 1-10
                 embed_dim=8,
                 food_ppi_dim=2,
                 encoder_hidden_size=128,
                 decoder_hidden_size=128,
                 num_layers=1,
                 dropout=0.3):
        super(SeqClassifier, self).__init__()
        
        # Embedding layers for the two symptom channels
        self.embed_reflux = nn.Embedding(symptom_vocab_size, embed_dim)
        self.embed_digestive = nn.Embedding(symptom_vocab_size, embed_dim)
        
        # Encoder: input per timestep is concatenation of:
        # [embedding(reflux), embedding(digestive), food/PPI input]
        self.encoder_input_dim = 2 * embed_dim + food_ppi_dim
        self.encoder = nn.LSTM(input_size=self.encoder_input_dim,
                               hidden_size=encoder_hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        
        # Decoder: takes future food/PPI inputs (without symptom info)
        self.decoder = nn.LSTM(input_size=food_ppi_dim,
                               hidden_size=decoder_hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        
        # Explicit dropout applied after decoder output.
        self.explicit_dropout = nn.Dropout(dropout)
        
        # Two classification heads: one for each symptom.
        self.head_reflux = nn.Linear(decoder_hidden_size, symptom_vocab_size)
        self.head_digestive = nn.Linear(decoder_hidden_size, symptom_vocab_size)
    
    def forward(self, hist_symptoms, combined_inputs):
        """
        Args:
            hist_symptoms: Tensor of shape (batch, seq_len_hist, 2) with discrete labels (1-10).
            combined_inputs: Tensor of shape (batch, seq_len_hist + seq_len_future, food_ppi_dim)
                             with continuous values.
        Returns:
            logits_reflux: Tensor of shape (batch, seq_len_future, symptom_vocab_size)
            logits_digestive: Tensor of shape (batch, seq_len_future, symptom_vocab_size)
        """
        batch_size = hist_symptoms.size(0)
        seq_len_hist = hist_symptoms.size(1)
        total_seq_len = combined_inputs.size(1)
        seq_len_future = total_seq_len - seq_len_hist
        
        # Extract historical portion of inputs
        past_inputs = combined_inputs[:, :seq_len_hist, :]  # shape: (batch, seq_len_hist, 2)
        
        # Convert historical symptoms from 1-10 to 0-9 indices.
        hist_symptoms_idx = hist_symptoms - 1  # now values in 0-9
        
        # Separate the two channels
        reflux_hist = self.embed_reflux(hist_symptoms_idx[:, :, 0])       # shape: (batch, seq_len_hist, embed_dim)
        digestive_hist = self.embed_digestive(hist_symptoms_idx[:, :, 1])   # shape: (batch, seq_len_hist, embed_dim)
        
        # Concatenate embeddings with corresponding past inputs
        encoder_input = torch.cat([reflux_hist, digestive_hist, past_inputs], dim=2)
        # Shape: (batch, seq_len_hist, 2*embed_dim + 2)
        
        # Encode historical sequence
        _, (h, c) = self.encoder(encoder_input)
        
        # Future inputs for decoder: remaining timesteps from combined_inputs
        future_inputs = combined_inputs[:, seq_len_hist:, :]  # shape: (batch, seq_len_future, 2)
        
        # Decode: use future inputs and initial state from encoder
        decoder_output, _ = self.decoder(future_inputs, (h, c))  # shape: (batch, seq_len_future, decoder_hidden_size)

        # Explicitly apply dropout to decoder outputs.
        decoder_output = self.explicit_dropout(decoder_output)
        
        # Classification heads: produce logits for each future timestep
        logits_reflux = self.head_reflux(decoder_output)       # shape: (batch, seq_len_future, symptom_vocab_size)
        logits_digestive = self.head_digestive(decoder_output) # shape: (batch, seq_len_future, symptom_vocab_size)
        
        return logits_reflux, logits_digestive
    
# -------------------------------------------------
# Define the Regression Sequence-to-Sequence Model
# -------------------------------------------------
class SeqRegressor(nn.Module):
    def __init__(self, symptom_dim=2, input_dim=2,
                 encoder_hidden_size=128, decoder_hidden_size=128,
                 num_layers=1, dropout=0.3):
        """
        Sequence-to-sequence regressor.
        - Encoder takes historical symptoms (continuous, shape: (batch, seq_len_hist, 2))
          concatenated with historical food/PPI inputs (shape: (batch, seq_len_hist, 2)).
        - The combined input per timestep is of dimension 4.
        - Decoder takes future food/PPI inputs (shape: (batch, seq_len_future, 2)).
        - The decoder's output is mapped to continuous symptom values via a linear layer.
        """
        super(SeqRegressor, self).__init__()
        
        # Encoder input: concatenation of historical symptoms and past food/PPI inputs.
        self.encoder_input_dim = symptom_dim + input_dim  # 2 + 2 = 4
        self.encoder = nn.LSTM(input_size=self.encoder_input_dim,
                               hidden_size=encoder_hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        
        # Decoder: input is future food/PPI inputs (dimension=2)
        self.decoder = nn.LSTM(input_size=input_dim,
                               hidden_size=decoder_hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        
        # Explicit dropout after decoder output to enable MC dropout uncertainty estimation.
        self.explicit_dropout = nn.Dropout(dropout)
        
        # Regression head: map decoder hidden state to symptom predictions (continuous)
        self.fc = nn.Linear(decoder_hidden_size, symptom_dim)
        
    def forward(self, hist_symptoms, combined_inputs):
        """
        Args:
            hist_symptoms: Tensor of shape (batch, seq_len_hist, 2)
            combined_inputs: Tensor of shape (batch, seq_len_hist + seq_len_future, 2)
        Returns:
            predictions: Tensor of shape (batch, seq_len_future, 2) -- continuous symptom predictions.
        """
        batch_size = hist_symptoms.size(0)
        seq_len_hist = hist_symptoms.size(1)
        total_seq_len = combined_inputs.size(1)
        seq_len_future = total_seq_len - seq_len_hist
        
        # Extract past inputs from combined_inputs (corresponding to historical period)
        past_inputs = combined_inputs[:, :seq_len_hist, :]  # (batch, seq_len_hist, 2)
        
        # Encoder input: concatenate historical symptoms and past inputs along feature dim.
        encoder_input = torch.cat([hist_symptoms, past_inputs], dim=2)  # (batch, seq_len_hist, 4)
        
        # Pass through encoder LSTM.
        _, (h, c) = self.encoder(encoder_input)
        
        # Future inputs: remaining timesteps from combined_inputs for decoder.
        future_inputs = combined_inputs[:, seq_len_hist:, :]  # (batch, seq_len_future, 2)
        
        # Decode using future inputs and encoder final states.
        decoder_output, _ = self.decoder(future_inputs, (h, c))  # (batch, seq_len_future, decoder_hidden_size)
        
        # Apply explicit dropout (active during training and MC dropout inference)
        decoder_output = self.explicit_dropout(decoder_output)
        
        # Regression head: map to continuous symptom predictions.
        predictions = self.fc(decoder_output)  # (batch, seq_len_future, 2)
        return predictions




if __name__=="__main__":

    data_path = "synthetic_data"
    # load parameters
    file_name = "parameters.pt"
    file_path = os.path.join(data_path, file_name)
    parameters = torch.load(file_path, weights_only=False)
    # store acid traj
    file_name = "acid_traj.pt"
    file_path = os.path.join(data_path, file_name)
    acid_traj = torch.load(file_path)
    # ppi traj
    file_name = "ppi_traj.pt"
    file_path = os.path.join(data_path, file_name)
    ppi_traj = torch.load(file_path)
    # food traj
    file_name = "food_traj.pt"
    file_path = os.path.join(data_path, file_name)
    food_traj = torch.load(file_path)
    # time traj
    file_name = "time_traj.pt"
    file_path = os.path.join(data_path, file_name)
    t_data = torch.load(file_path)
    # symptom traj
    file_name = "symptom_traj.pt"
    file_path = os.path.join(data_path, file_name)
    symptom_traj = torch.load(file_path)
    # constraints
    file_name = "constraints_traj.pt"
    file_path = os.path.join(data_path, file_name)
    constraints = torch.load(file_path)
    print("!!!synthetic training data is loaded!!!")

    # Training and validation data
    # Extract training and validation data from the file
    t_data = t_data.cpu().numpy()
    acid_obs= acid_traj.cpu().numpy()[0]  
    ppi_arr = ppi_traj.cpu().numpy()[0]
    meal_arr= food_traj.cpu().numpy()[0]
    reflux_obs = symptom_traj.cpu().numpy()[0,0,:]
    digestive_obs = symptom_traj.cpu().numpy()[0,1,:]
    constraints_obs = constraints.cpu().numpy()[0]

    # Training and validation datasets split
    time_step = 1/24
    train_start = int(0/time_step)
    train_data_num = int(200/time_step)
    train_end = train_start + train_data_num
    t_data_train = t_data[train_start:train_end]
    acid_obs_train = acid_obs[train_start:train_end]
    ppi_arr_train = ppi_arr[train_start:train_end]
    meal_arr_train = meal_arr[train_start:train_end]
    digestive_obs_train = digestive_obs[train_start:train_end]
    reflux_obs_train = reflux_obs[train_start:train_end]

    t_data_val = t_data[train_end:] #- t_data[train_end]
    acid_obs_val = acid_obs[train_end:]
    ppi_arr_val = ppi_arr[train_end:]
    meal_arr_val = meal_arr[train_end:]
    digestive_obs_val = digestive_obs[train_end:]
    reflux_obs_val = reflux_obs[train_end:]

    # Define hyperparameters for sequence lengths
    N_pre = 10    # past days
    N_next = 5   # future days to predict
    seq_len_hist = int(N_pre/time_step)        
    seq_len_future = int(N_next/time_step)
    REGRESSION = True

    if REGRESSION:
        train_data = torch.tensor(np.stack((meal_arr_train, ppi_arr_train), axis=1)).to(torch.float32)
        train_labels = torch.tensor(np.stack((reflux_obs_train, digestive_obs_train), axis=1)).to(torch.float32)
        valid_data = torch.tensor(np.stack((meal_arr_val, ppi_arr_val), axis=1)).to(torch.float32)
        valid_labels = torch.tensor(np.stack((reflux_obs_val, digestive_obs_val), axis=1)).to(torch.float32)
    else:
        train_data = torch.tensor(np.stack((meal_arr_train, ppi_arr_train), axis=1)).to(torch.float32)
        train_labels = torch.tensor(np.stack((reflux_obs_train, digestive_obs_train), axis=1)).to(torch.int)
        valid_data = torch.tensor(np.stack((meal_arr_val, ppi_arr_val), axis=1)).to(torch.float32)
        valid_labels = torch.tensor(np.stack((reflux_obs_val, digestive_obs_val), axis=1)).to(torch.int)

    # Load training and validation datasets
    train_dataset = TimeSeriesSeqDataset(train_labels, train_data, seq_len_hist, seq_len_future)
    valid_dataset = TimeSeriesSeqDataset(valid_labels, valid_data, seq_len_hist, seq_len_future)

    # Create DataLoaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Create the model and move it to the appropriate device.
    if REGRESSION:
        model = SeqRegressor(symptom_dim=2, input_dim=2,
                     encoder_hidden_size=128,
                     decoder_hidden_size=128,
                     num_layers=1,
                     dropout=0.3).to(device) 
    else:
        model = SeqClassifier(symptom_vocab_size=10,  # classes 0-9
                    embed_dim=8,
                    food_ppi_dim=2,
                    encoder_hidden_size=128,
                    decoder_hidden_size=128,
                    num_layers=1,
                    dropout=0.2).to(device)
        
    

    ##########################################
    # Define the Loss Function and Optimizer
    ##########################################
    lambda_smooth = 0.1   # scaling factor for the smoothing loss term
    weight_decay_value = 1e-4
    if REGRESSION:
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay_value)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay_value)

    ##########################################
    # Training Loop
    ##########################################
    num_epochs = 100
    best_val_loss = float('inf')
    train_loss_total = []
    valid_loss_total = []
    if REGRESSION:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for hist_symptoms, combined_inputs, target in train_loader:
                hist_symptoms = hist_symptoms.to(device)  # (batch, seq_len_hist, 2)
                combined_inputs = combined_inputs.to(device).float()  # (batch, total_seq_len, 2)
                target = target.to(device).float()  # (batch, seq_len_future, 2)
                
                optimizer.zero_grad()
                predictions = model(hist_symptoms, combined_inputs)  # (batch, seq_len_future, 2)
                loss = criterion(predictions, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            train_loss_total.append(avg_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")
            
            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for hist_symptoms, combined_inputs, target in valid_loader:
                    hist_symptoms = hist_symptoms.to(device)
                    combined_inputs = combined_inputs.to(device).float()
                    target = target.to(device).float()
                    predictions = model(hist_symptoms, combined_inputs)
                    loss = criterion(predictions, target)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(valid_loader)
            valid_loss_total.append(avg_val_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

            # Save checkpoint if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }
                torch.save(checkpoint, 'best_checkpoint.pth')
                print(f"Checkpoint saved at epoch {epoch+1} with Validation Loss: {avg_val_loss:.4f}")

    else:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for hist_symptoms, combined_inputs, target in train_loader:
                # Move data to device.
                hist_symptoms = hist_symptoms.to(device)  # (batch, seq_len_hist, 2), values in 1-10
                combined_inputs = combined_inputs.to(device).float()  # (batch, total_seq_len, 2)
                target = target.to(device)  # (batch, seq_len_future, 2), values in 1-10
                
                # Convert target labels from 1-10 to 0-9.
                target_idx = target - 1
                
                optimizer.zero_grad()
                logits_reflux, logits_digestive = model(hist_symptoms, combined_inputs)
                # logits: (batch, seq_len_future, 10)
                batch_size, seq_len_future, _ = logits_reflux.shape
                
                # Reshape for CrossEntropyLoss: predictions to (batch*seq_len_future, 10) and targets to (batch*seq_len_future)
                loss_reflux = criterion(logits_reflux.view(batch_size * seq_len_future, -1),
                                        target_idx[:, :, 0].view(-1))
                loss_digestive = criterion(logits_digestive.view(batch_size * seq_len_future, -1),
                                        target_idx[:, :, 1].view(-1))
                ce_loss = loss_reflux + loss_digestive
            
                # Compute smoothing loss:
                # For each sample, penalize the squared difference between logits at consecutive timesteps.
                smooth_loss_reflux = torch.mean(torch.sum((logits_reflux[:, 1:, :] - logits_reflux[:, :-1, :]) ** 2, dim=2))
                smooth_loss_digestive = torch.mean(torch.sum((logits_digestive[:, 1:, :] - logits_digestive[:, :-1, :]) ** 2, dim=2))
                smooth_loss = lambda_smooth * (smooth_loss_reflux + smooth_loss_digestive)
                
                # Total loss
                loss = ce_loss + smooth_loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            train_loss_total.append(avg_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            
            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for hist_symptoms, combined_inputs, target in valid_loader:
                    hist_symptoms = hist_symptoms.to(device)
                    combined_inputs = combined_inputs.to(device).float()
                    target = target.to(device)
                    target_idx = target - 1
                    logits_reflux, logits_digestive = model(hist_symptoms, combined_inputs)
                    batch_size, seq_len_future, _ = logits_reflux.shape
                    loss_reflux = criterion(logits_reflux.view(batch_size * seq_len_future, -1),
                                            target_idx[:, :, 0].view(-1))
                    loss_digestive = criterion(logits_digestive.view(batch_size * seq_len_future, -1),
                                            target_idx[:, :, 1].view(-1))
                    ce_loss_val = loss_reflux + loss_digestive
                    smooth_loss_reflux_val = torch.mean(torch.sum((logits_reflux[:, 1:, :] - logits_reflux[:, :-1, :]) ** 2, dim=2))
                    smooth_loss_digestive_val = torch.mean(torch.sum((logits_digestive[:, 1:, :] - logits_digestive[:, :-1, :]) ** 2, dim=2))
                    smooth_loss_val = lambda_smooth * (smooth_loss_reflux_val + smooth_loss_digestive_val)
                    loss_val = ce_loss_val + smooth_loss_val
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(valid_loader)
            valid_loss_total.append(avg_val_loss)
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # Save checkpoint if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }
                torch.save(checkpoint, 'best_checkpoint.pth')
                print(f"Checkpoint saved at epoch {epoch+1} with Validation Loss: {avg_val_loss:.4f}")
    
    result_data_path = "training_results"
    os.makedirs(result_data_path, exist_ok=True)
    file_name = "train_loss.pt"
    file_path = os.path.join(result_data_path, file_name)
    torch.save(train_loss_total, file_path)

    file_name = "valid_loss.pt"
    file_path = os.path.join(result_data_path, file_name)
    torch.save(valid_loss_total, file_path)

    plt.figure(figsize=(12, 5))
    t_plot = range(1,len(train_loss_total)+1)
    # Plot for Reflux (first symptom channel)
    plt.subplot(1, 1, 1)
    plt.plot(t_plot, train_loss_total, label='Training')
    plt.plot(t_plot, valid_loss_total, label='Validation')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    plt.close()