import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from BNN import TimeSeriesSeqDataset, SeqClassifier, SeqRegressor


# Set device: use MPS if available, otherwise fallback to CPU or CUDA
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def predict_with_uncertainty_reg(model, x_hist, x_comb, num_samples=50):
    """
    Performs MC dropout for regression by doing multiple forward passes.
    
    Args:
        model: Trained regression model.
        x_hist: Historical symptoms tensor, shape (batch, seq_len_hist, 2) with continuous values.
        x_comb: Combined inputs tensor, shape (batch, total_seq_len, 2).
        num_samples: Number of stochastic forward passes.
        
    Returns:
        mean_preds, std_preds: Each of shape (batch, seq_len_future, 2)
    """
    model.train()  # Enable dropout for MC dropout inference.
    preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            # Model output shape: (batch, seq_len_future, 2)
            pred = model(x_hist.to(device), x_comb.to(device).float())
            preds.append(pred.unsqueeze(0))
    # Concatenate along a new dimension for the samples.
    preds = torch.cat(preds, dim=0)  # shape: (num_samples, batch, seq_len_future, 2)
    mean_preds = preds.mean(dim=0)
    std_preds = preds.std(dim=0)
    return mean_preds, std_preds

def receding_horizon_MC_reg(model, dataset, device, num_samples=50, horizon_interval=1):
    """
    Runs MC dropout regression predictions on every sample in the dataset
    (or every 'horizon_interval'-th sample) so that you can compare the ground truth 
    and predicted symptom trajectories for the full dataset.
    
    Args:
        model: Trained regression model.
        dataset: A PyTorch Dataset (e.g., TimeSeriesSeqDataset) containing samples.
        device: The torch.device to use.
        num_samples: Number of stochastic forward passes.
        horizon_interval: Only process every `horizon_interval`-th sample.
        
    Returns:
        all_gt: List of ground truth target arrays (each shape: (seq_len_future, 2))
        all_mean_preds: List of mean prediction arrays (each shape: (seq_len_future, 2))
        all_std_preds: List of uncertainty arrays (each shape: (seq_len_future, 2))
    """
    from torch.utils.data import DataLoader

    # Use a DataLoader with batch_size=1 to process one sample at a time.
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_gt = []
    all_mean_preds = []
    all_std_preds = []
    sample_index = 0
    
    with torch.no_grad():
        for hist_symptoms, combined_inputs, target in loader:
            if sample_index % horizon_interval == 0:
                # Move tensors to device.
                hist_symptoms = hist_symptoms.to(device)   # shape: (1, seq_len_hist, 2)
                combined_inputs = combined_inputs.to(device).float()  # shape: (1, total_seq_len, 2)
                target = target.to(device).float()         # shape: (1, seq_len_future, 2)
                
                # Get MC dropout predictions.
                mean_preds, std_preds = predict_with_uncertainty_reg(model, hist_symptoms, combined_inputs, num_samples=num_samples)
                
                # Remove the batch dimension and convert to numpy.
                gt_sample = target.squeeze(0).cpu().numpy()         # (seq_len_future, 2)
                mean_sample = mean_preds.squeeze(0).cpu().numpy()     # (seq_len_future, 2)
                std_sample = std_preds.squeeze(0).cpu().numpy()       # (seq_len_future, 2)
                
                all_gt.append(gt_sample)
                all_mean_preds.append(mean_sample)
                all_std_preds.append(std_sample)
            sample_index += 1
    
    return all_gt, all_mean_preds, all_std_preds



def predict_with_uncertainty(model, x_hist, x_comb, num_samples=50):
    """
    Performs MC dropout by doing multiple forward passes.
    Args:
        model: Trained model.
        x_hist: Historical symptoms tensor, shape (batch, seq_len_hist, 2) with values in 1-10.
        x_comb: Combined inputs tensor, shape (batch, total_seq_len, 2).
        num_samples: Number of stochastic forward passes.
    Returns:
        mean_prob_reflux, std_prob_reflux: (batch, seq_len_future, 10)
        mean_prob_digestive, std_prob_digestive: (batch, seq_len_future, 10)
    """
    model.train()  # enable dropout
    preds_reflux = []
    preds_digestive = []
    
    # Ensure x_hist is 0-indexed for embedding.
    x_hist = x_hist - 1
    
    with torch.no_grad():
        for _ in range(num_samples):
            logits_reflux, logits_digestive = model(x_hist.to(device), x_comb.to(device).float())
            # Compute softmax probabilities for each head
            prob_reflux = torch.softmax(logits_reflux, dim=2)  # shape: (batch, seq_len_future, 10)
            prob_digestive = torch.softmax(logits_digestive, dim=2)  # shape: (batch, seq_len_future, 10)
            preds_reflux.append(prob_reflux.unsqueeze(0))
            preds_digestive.append(prob_digestive.unsqueeze(0))
    
    preds_reflux = torch.cat(preds_reflux, dim=0)  # shape: (num_samples, batch, seq_len_future, 10)
    preds_digestive = torch.cat(preds_digestive, dim=0)
    
    mean_prob_reflux = preds_reflux.mean(dim=0)
    std_prob_reflux = preds_reflux.std(dim=0)
    mean_prob_digestive = preds_digestive.mean(dim=0)
    std_prob_digestive = preds_digestive.std(dim=0)
    
    return mean_prob_reflux, std_prob_reflux, mean_prob_digestive, std_prob_digestive

# ------------------------------------------------------------
# Define a Function to Run Receding Horizon Prediction on a Dataset
# ------------------------------------------------------------
def receding_horizon_MC(model, dataset, device, num_samples=50, horizon_interval=1):
    """
    Run MC dropout predictions on a dataset.
    Returns:
        gt_symptoms: list of ground truth targets (numpy arrays) for each sample,
                     each of shape (seq_len_future, 2)
        mean_preds: list of mean predictions (numpy arrays) for each sample,
                     each of shape (seq_len_future, 2)
        std_preds: list of uncertainties (numpy arrays) for each sample,
                     each of shape (seq_len_future, 2)
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    gt_symptoms = []
    mean_preds = []
    std_preds = []  # Combine uncertainties from both heads into a (seq_len_future, 2) array
    sample_index = 0
    with torch.no_grad():
        for hist_symptoms, combined_inputs, target in loader:
            if sample_index % horizon_interval == 0:
                hist_symptoms = hist_symptoms.to(device)  # shape: (1, seq_len_hist, 2)
                combined_inputs = combined_inputs.to(device).float()  # shape: (1, total_seq_len, 2)
                target = target.to(device)  # shape: (1, seq_len_future, 2)
                
                # Get MC dropout predictions.
                mean_prob_reflux, std_prob_reflux, mean_prob_digestive, std_prob_digestive = predict_with_uncertainty(model, hist_symptoms, combined_inputs, num_samples=num_samples)
                
                # Get predicted classes for each future timestep: argmax + 1
                pred_reflux = mean_prob_reflux.argmax(dim=2) + 1  # shape: (1, seq_len_future)
                pred_digestive = mean_prob_digestive.argmax(dim=2) + 1  # shape: (1, seq_len_future)
                
                # Combine predictions into a (seq_len_future, 2) array.
                pred_combined = torch.cat([pred_reflux, pred_digestive], dim=0).view(2, -1).transpose(0, 1)
                
                # Similarly, combine uncertainties (we'll take the std of the argmax probabilities is less meaningful;
                # instead, we can compute the std for each head and keep them separate).
                uncertainty_reflux = std_prob_reflux.mean(dim=2)  # average std over classes, shape: (1, seq_len_future)
                uncertainty_digestive = std_prob_digestive.mean(dim=2)  # shape: (1, seq_len_future)
                uncertainty_combined = torch.cat([uncertainty_reflux, uncertainty_digestive], dim=0).view(2, -1).transpose(0, 1)
                
                gt = target.squeeze(0).cpu().numpy()  # shape: (seq_len_future, 2)
                pred_np = pred_combined.squeeze(0).cpu().numpy()  # shape: (seq_len_future, 2)
                uncert_np = uncertainty_combined.squeeze(0).cpu().numpy()  # shape: (seq_len_future, 2)
                
                gt_symptoms.append(gt)
                mean_preds.append(pred_np)
                std_preds.append(uncert_np)
            sample_index += 1
    return gt_symptoms, mean_preds, std_preds


# ------------------------------------------------------------
# Plotting Function with Uncertainty Shading
# ------------------------------------------------------------
def plot_sample_with_uncertainty(gt, pred, uncert, title, ppi, food, acid_gt, constraints):
    """
    Plot ground truth and predicted symptom trajectories with uncertainty.
    gt, pred, uncert: arrays of shape (seq_len_future, 2).
                     For uncert, each value represents the average std for that timestep.
    """
    seq_len_future = gt.shape[0]
    t = np.arange(seq_len_future)/24
    
    
    plt.figure(figsize=(15, 7))
    
    # Plot for Reflux (first symptom channel)
    plt.subplot(2, 2, 1)
    plt.plot(t, gt[:, 0], color='r', linestyle='--', label='GT Reflux')
    plt.plot(t, pred[:, 0], color='b', linestyle=':', label='Predicted Reflux')
    # Fill uncertainty (predicted ± uncertainty)
    plt.fill_between(t, pred[:, 0] - uncert[:, 0], pred[:, 0] + uncert[:, 0], color='gray', alpha=0.3, label='Uncertainty')
    plt.xlim([0, 380]) 
    plt.xlabel('Time (days)')
    plt.ylabel('Reflux Symptom Score')
    # plt.title(f'{title} - Reflux')
    plt.legend()
    
    # Plot for Digestive (second symptom channel)
    plt.subplot(2, 2, 3)
    plt.plot(t, gt[:, 1], color='r', linestyle='--', label='GT Digestive')
    plt.plot(t, pred[:, 1], color='b', linestyle=':', label='Predicted Digestive')
    plt.fill_between(t, pred[:, 1] - uncert[:, 1], pred[:, 1] + uncert[:, 1], color='gray', alpha=0.3, label='Uncertainty')
    plt.xlim([0, 380]) 
    plt.xlabel('Time (days)')
    plt.ylabel('Digestive Symptom Score')
    # plt.title(f'{title} - Digestive')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(t, food[0:t.shape[0]], color='r', linestyle='--', label='Meal')
    plt.plot(t, ppi[0:t.shape[0]], color='b', linestyle='-', label='PPI')
    # # Fill uncertainty (predicted ± uncertainty)
    # plt.fill_between(t, pred[:, 0] - uncert[:, 0], pred[:, 0] + uncert[:, 0], color='gray', alpha=0.3, label='Uncertainty')
    plt.xlim([0, 380]) 
    plt.xlabel('Time (days)')
    plt.ylabel('Food and PPI inputs')
    # plt.title(f'{title} - Inputs')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(t, acid_gt[0:t.shape[0]], color='r', linestyle='--', label='Acid level')
    # plt.plot(t, pred[:, 1], color='b', linestyle=':', label='Predicted Digestive')
    # plt.fill_between(t, pred[:, 1] - uncert[:, 1], pred[:, 1] + uncert[:, 1], color='gray', alpha=0.3, label='Uncertainty')
    plt.axhline(constraints[0, 0], color="k", linestyle="--", label="reflux constraint")
    plt.axhline(constraints[0, 1], color="b", linestyle="--", label="digestive constraint")
    plt.xlim([0, 380]) 
    plt.xlabel('Time (days)')
    plt.ylabel('Acid')
    # plt.title(f'{title} - Digestive')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("BNN_validation.jpg", dpi=600, bbox_inches='tight', transparent=True)
    # plt.show()
    plt.close()


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

    # training/validation loss data
    result_data_path = "training_results"
    # training loss
    file_name = "train_loss.pt"
    file_path = os.path.join(result_data_path, file_name)
    train_loss = torch.load(file_path)
    # validation loss
    file_name = "valid_loss.pt"
    file_path = os.path.join(result_data_path, file_name)
    valid_loss = torch.load(file_path)

    


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
    train_data_num = int(400/time_step)
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

    # Define model structure
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

    # Load the best checkpoint
    checkpoint = torch.load('best_checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 
    if REGRESSION:
        gt_train, pred_train, uncert_train = receding_horizon_MC_reg(model, train_dataset, device, num_samples=150, horizon_interval=int(N_next/time_step))
    else:
        gt_train, pred_train, uncert_train = receding_horizon_MC(model, train_dataset, device, num_samples=150, horizon_interval=int(N_next/time_step))

    gt_all = np.concatenate(gt_train, axis=0)       
    pred_all = np.concatenate(pred_train, axis=0)   
    uncert_all = np.concatenate(uncert_train, axis=0) 

    gt_down = gt_all
    pred_down = pred_all
    uncert_down = uncert_all

    plot_sample_with_uncertainty(gt_down, pred_down, uncert_down, title="Validation", ppi=ppi_arr_train, food=meal_arr_train, acid_gt=acid_obs_train, constraints=constraints)

    print("!!!MC dropout validation!!!")
