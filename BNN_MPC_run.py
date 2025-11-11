import numpy as np
import itertools
import torch
import matplotlib.pyplot as plt
from BNN_validate import predict_with_uncertainty_reg
import os
from models import Detailed_Model
from data_generation import noisy_acid_reflux_score, noisy_digestive_issue_score, rk4_solve
from BNN import SeqRegressor


def generate_ppi_trajectory(base_ppi, actions, horizon_days, timesteps_per_day=24):
    """
    Generate a PPI profile over a horizon based on a sequence of daily actions.
    
    Args:
        base_ppi (float): The current (or last known) PPI level in [0, 1].
        actions (list or tuple of str): Daily actions for the horizon: "same", "increase", "decrease".
        horizon_days (int): Number of days in the prediction horizon.
        timesteps_per_day (int): Number of timesteps per day.
    
    Returns:
        ppi_profile (np.array): An array of shape (horizon_days * timesteps_per_day,) with the PPI levels.
    """
    ppi_profile = []
    current_ppi = base_ppi
    for day in range(horizon_days):
        action = actions[day]
        if action == "increase":
            current_ppi = min(current_ppi + 0.2, 1.0)
        elif action == "decrease":
            current_ppi = max(current_ppi - 0.2, 0.0)
        # "same" means no change.
        day_profile = [current_ppi] * timesteps_per_day
        ppi_profile.extend(day_profile)
    return np.array(ppi_profile)

def mpc_search(model, base_ppi, food_forecasts, current_hist_inputs, food_hist, ppi_hist, prediction_horizon_days,
               timesteps_per_day=24, threshold=3, candidate_actions=["same", "increase", "decrease"],
               lambda_soft=100.0, beta=1.28, num_samples=50):
    """
    MPC search for the best PPI profile over the prediction horizon using multiple food forecasts
    and soft chance constraints.
    
    Args:
        model: Trained NN regressor (with MC dropout for uncertainty estimation).
        base_ppi: Current PPI level (float).
        food_forecasts: A list of food forecast arrays. Each array has shape (H, F), where 
                        H = prediction_horizon_days*timesteps_per_day.
        current_hist_inputs: Historical data required by the NN model, tensor of shape (1, seq_len_hist, 2).
        prediction_horizon_days: Number of days in the prediction horizon.
        timesteps_per_day: Number of timesteps per day.
        threshold: Symptom threshold (e.g. 3).
        candidate_actions: List of daily actions for PPI.
        lambda_soft: Penalty weight for constraint violations.
        beta: Factor for chance constraint (e.g., 1.28 for 90% probability).
        num_samples: Number of MC dropout forward passes.
        
    Returns:
        best_candidate_actions: The selected candidate action sequence (tuple of actions).
        best_ppi_profile: The corresponding PPI profile (np.array of shape (H,)).
        best_objective: The objective value (cost + penalty).
        feasible: Boolean indicating if any candidate achieved zero violation.
    """
    horizon = prediction_horizon_days
    total_timesteps = horizon * timesteps_per_day
    
    # Generate all candidate action sequences.
    candidate_sequences = list(itertools.product(candidate_actions, repeat=horizon))
    
    feasible_candidates = []
    objective_values = []
    
    # For each candidate, generate the candidate PPI trajectory and evaluate over all food forecasts.
    for candidate in candidate_sequences:
        candidate_ppi = generate_ppi_trajectory(base_ppi, candidate, horizon, timesteps_per_day)
        
        # Build the combined_inputs for the NN model: 
        # Assume combined_inputs has shape (1, total_timesteps, 2) where column0 is food, column1 is PPI.
        penalty_list = []
        for food_forecast in food_forecasts:
            
            food_combine_input = np.concatenate([food_hist.cpu().numpy().reshape(-1,1), food_forecast.cpu().numpy().reshape(-1, 1)], axis=0)
            ppi_combine_input = np.concatenate([ppi_hist.cpu().numpy().reshape(-1,1), candidate_ppi.reshape(-1, 1)], axis=0)
            combined_inputs = np.concatenate([food_combine_input, ppi_combine_input], axis=1)
            
            # Convert to tensor with batch dimension.
            combined_inputs_tensor = torch.from_numpy(combined_inputs).unsqueeze(0).float().to(device)
            
            # Use current_hist_inputs (assumed to be on device) for historical data.
            # Run MC dropout to simulate the future symptom trajectory.
            mean_pred, std_pred = predict_with_uncertainty_reg(model, current_hist_inputs, combined_inputs_tensor, num_samples=num_samples)
            mean_pred = mean_pred.squeeze(0).cpu().numpy()  # shape: (total_timesteps, 2)
            std_pred = std_pred.squeeze(0).cpu().numpy()      # shape: (total_timesteps, 2)
            
            # Compute violation: for each timestep and symptom, compute positive violation if any.
            violation = np.maximum(0, mean_pred + beta * std_pred - threshold)  # shape: (total_timesteps, 2)
            # total_violation = np.sum(violation)  # sum over all timesteps and both symptoms.
            total_violation = np.sum(violation[:,0] + 10*violation[:,1])
            penalty_list.append(total_violation)
        
        # Aggregate penalty over all food forecasts. We can take the maximum violation (worst-case)
        # or the average. Here, we take maximum to be conservative.
        aggregated_penalty = max(penalty_list)
        
        # Compute objective: For instance, total PPI usage + lambda_soft * aggregated_penalty.
        objective = np.sum(candidate_ppi) + lambda_soft * aggregated_penalty
        feasible_candidates.append(candidate)
        objective_values.append(objective)
    
    # Choose candidate with minimum objective.
    best_index = np.argmin(objective_values)
    best_candidate_actions = candidate_sequences[best_index]
    best_ppi_profile = generate_ppi_trajectory(base_ppi, best_candidate_actions, horizon, timesteps_per_day)
    best_objective = objective_values[best_index]
    
    # We consider a candidate fully feasible if aggregated_penalty is 0 in the best candidate.
    feasible = (max(penalty_list) == 0)  if objective_values[best_index] < np.inf else False
    
    return best_candidate_actions, best_ppi_profile, best_objective, feasible



def generate_random_meal(t_vals, num_profiles, device):
    base_meal_centers = [8.0/24.0, 12.0/24.0, 18.0/24.0]
    n_steps_plus_1 = t_vals.shape[0]
    meal_arrays = torch.zeros(num_profiles, n_steps_plus_1, device=device)
    for i in range(num_profiles):
        # We'll store random offsets/amplitudes for each day in a dictionary or so
        # but let's do an on-the-fly approach:
        # 1) We find the max day in t_vals
        max_day = int(torch.floor(t_vals[-1]).item()) + 1

        # We'll store meal info for each day => e.g. day_meals[day] = [(center, amp), (center, amp), (center, amp)]
        day_meals = {}

        # Generate random meal pattern for each integer day in [0..max_day]
        for day_i in range(max_day+1):
            meal_info = []
            for meal_center in base_meal_centers:
                # Random shift within +/- 0.5 hour => +/- 0.5/24 day
                center_shift = np.random.uniform(-0.5, 0.5)/24.0
                # random amplitude within e.g. [0.5..1.5]
                if meal_center == base_meal_centers[0]:
                    amplitude = np.random.uniform(0.6, 1.2)
                elif meal_center == base_meal_centers[1]:
                    amplitude = np.random.uniform(0.6, 1.5)
                elif meal_center == base_meal_centers[2]:
                    amplitude = np.random.uniform(0.5, 1.8)
                # final meal center
                meal_center_final = meal_center + center_shift
                meal_info.append((meal_center_final, amplitude))
            day_meals[day_i] = meal_info

        # Now fill meal_arrays[i, j] for each time step j
        for j in range(n_steps_plus_1):
            t_ = t_vals[j]
            # day_of_t = floor(t_)
            day_of_t = int(torch.floor(t_).item())
            # fraction of day
            t_day = (t_ - day_of_t)

            # If day_of_t is beyond max_day, we skip or clamp
            if day_of_t < 0 or day_of_t > max_day:
                # no meal for out-of-range days, or clamp
                continue

            # retrieve today's meal_info
            meal_info = day_meals[day_of_t]

            # define a small sigma for meal widths
            sigma = np.random.uniform(0.3, 0.6)/24.0  # 0.3..0.6 hour spread => 18..36 min
            # accumulate meal pulses
            for (meal_center_final, meal_amp) in meal_info:
                exponent = -0.5 * ((t_day - meal_center_final)/sigma)**2
                meal_arrays[i, j] += meal_amp * torch.exp(torch.tensor(exponent, device=device))
    return meal_arrays


# ==============================================
# MPC with Soft Constraints and Multiple Food Forecasts
# ==============================================
if __name__=="__main__":


    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    simulation_days = 200
    prediction_horizon_days = 5
    timesteps_per_day = 24
    day_step = 1/timesteps_per_day
    threshold = 3
    base_ppi_initial = 0
    num_profiles = 1
    MPC_start = 85
    PPI_start = MPC_start
    PPI_end = PPI_start + 60
    MPC_recompute_counter = 0
    hist_days = 7
    seq_len_hist = hist_days * timesteps_per_day
    MPC_implementation_interval = 1
    MPC_enabled = False

    data_path = "synthetic_data"
    # load parameters
    file_name = "parameters.pt"
    file_path = os.path.join(data_path, file_name)
    parameters = torch.load(file_path, weights_only=False)
    # constraints
    file_name = "constraints_traj.pt"
    file_path = os.path.join(data_path, file_name)
    constraints = torch.load(file_path)

    n_steps = int(simulation_days/day_step)
    t_vals = torch.linspace(0, simulation_days, n_steps+1, device=device)


    ppi_arrays = torch.zeros(num_profiles, n_steps+1, device=device)
    ppi_arrays_MPC = torch.zeros(num_profiles, n_steps+1000, device=device)
    ppi_arrays[0, PPI_start*timesteps_per_day:PPI_end*timesteps_per_day] = 1
    food_arrays = generate_random_meal(t_vals=t_vals, num_profiles=num_profiles, device=device)
    
    # load the plant model
    args = [param[0] for param in parameters.values()]
    gastric_model_full = Detailed_Model(*args)

    y_all = []
    reflux_all = []
    digestive_all = []

    BNN_model = SeqRegressor(symptom_dim=2, input_dim=2,
                     encoder_hidden_size=128,
                     decoder_hidden_size=128,
                     num_layers=1,
                     dropout=0.3).to(device) 
    
    # Load the best checkpoint
    checkpoint = torch.load('best_checkpoint.pth', map_location=device)
    BNN_model.load_state_dict(checkpoint['model_state_dict'])
    BNN_model.eval() 

    y0 = torch.tensor([0.15, 0.4, 0.1, 1.0, 1.0], device=device)
    base_ppi = 0

    for t_sim in range(simulation_days):
        if MPC_enabled: 
            if t_sim < MPC_start:
                def ode_func(t, Y):
                    i_float = (t - 0)/day_step
                    i_idx   = int(torch.round(i_float).item())
                    i_idx   = max(0, min(i_idx, n_steps))  
                    p_val   = ppi_arrays[0, i_idx]
                    meal_val= food_arrays[0, i_idx]

                    return gastric_model_full.forward_array(Y, p_val, meal_val)

                _, y_vals = rk4_solve(ode_func, y0, t_sim, t_sim+1, day_step)

                reflux_scores = noisy_acid_reflux_score(y_vals[:,-1].cpu().numpy(), a_high=constraints[0, 0].cpu().item(), k_r=20, noise_std=0.2)
                digestive_scores = noisy_digestive_issue_score(y_vals[:,-1].cpu().numpy(), a_low=constraints[0, 1].cpu().item(), k_d=20, noise_std=0.2)
                base_ppi = ppi_arrays[0, (t_sim+1)*timesteps_per_day].cpu()
            else:
                if MPC_recompute_counter % MPC_implementation_interval == 0:
                    # Generate food forecasts
                    t_vals_food_forecasts = torch.linspace(0, prediction_horizon_days, int(prediction_horizon_days/day_step), device=device)
                    food_forecasts = generate_random_meal(t_vals_food_forecasts, num_profiles=2, device='cpu')
                    # Get the current_hist_inputs
                    reflux_hist = torch.tensor(np.concatenate(reflux_all[-hist_days:], axis=0))
                    digestive_hist = torch.tensor(np.concatenate(digestive_all[-hist_days:], axis=0))
                    current_hist_inputs = torch.stack([reflux_hist, digestive_hist], dim=-1).unsqueeze(0).to(torch.float32)
                    # Get the food hist
                    food_hist = food_arrays[0, (t_sim-hist_days)*timesteps_per_day:t_sim*timesteps_per_day]
                    ppi_hist = ppi_arrays_MPC[0, (t_sim-hist_days)*timesteps_per_day:t_sim*timesteps_per_day]
                    # MPC compute the PPI dosage
                    best_candidate_actions, best_ppi_profile, best_objective, feasible = mpc_search(
                        BNN_model, base_ppi, food_forecasts, current_hist_inputs, food_hist, ppi_hist, prediction_horizon_days,
                        timesteps_per_day=24, threshold=3, candidate_actions=["same", "increase", "decrease"],
                        lambda_soft=1, beta=1.28, num_samples=5
                    )
                    # Modify the ppi_arrays_MPC
                    ppi_arrays_MPC[0, t_sim*timesteps_per_day:(t_sim+prediction_horizon_days)*timesteps_per_day] = torch.tensor(best_ppi_profile)
                # Run simulation with modified ppi_arrays_MPC
                def ode_func(t, Y):
                    i_float = (t - 0)/day_step
                    i_idx   = int(torch.round(i_float).item())
                    i_idx   = max(0, min(i_idx, n_steps))  
                    p_val   = ppi_arrays_MPC[0, i_idx]
                    meal_val= food_arrays[0, i_idx]

                    return gastric_model_full.forward_array(Y, p_val, meal_val)

                _, y_vals = rk4_solve(ode_func, y0, t_sim, t_sim+1, day_step)

                reflux_scores = noisy_acid_reflux_score(y_vals[:,-1].cpu().numpy(), a_high=constraints[0, 0].cpu().item(), k_r=20, noise_std=0.2)
                digestive_scores = noisy_digestive_issue_score(y_vals[:,-1].cpu().numpy(), a_low=constraints[0, 1].cpu().item(), k_d=20, noise_std=0.2)
                base_ppi = best_ppi_profile[-1]
                MPC_recompute_counter += 1    
        else:
            def ode_func(t, Y):
                i_float = (t - 0)/day_step
                i_idx   = int(torch.round(i_float).item())
                i_idx   = max(0, min(i_idx, n_steps))  
                p_val   = ppi_arrays[0, i_idx]
                meal_val= food_arrays[0, i_idx]

                return gastric_model_full.forward_array(Y, p_val, meal_val)

            _, y_vals = rk4_solve(ode_func, y0, t_sim, t_sim+1, day_step)

            reflux_scores = noisy_acid_reflux_score(y_vals[:,-1].cpu().numpy(), a_high=constraints[0, 0].cpu().item(), k_r=20, noise_std=0.2)
            digestive_scores = noisy_digestive_issue_score(y_vals[:,-1].cpu().numpy(), a_low=constraints[0, 1].cpu().item(), k_d=20, noise_std=0.2)
            base_ppi = ppi_arrays[0, (t_sim+1)*timesteps_per_day].cpu()

        if t_sim != simulation_days - 1:
            y_all.append(y_vals[:-1])
            reflux_all.append(reflux_scores[:-1])
            digestive_all.append(digestive_scores[:-1])
        else:
            y_all.append(y_vals)
            reflux_all.append(reflux_scores)
            digestive_all.append(digestive_scores)

        y0 = y_vals[-1,:]
            
    y_all = torch.cat(y_all, dim=0)
    reflux_all = np.concatenate(reflux_all, axis=0)
    digestive_all = np.concatenate(digestive_all, axis=0)

    plt.figure(figsize=(13, 9))
    plt.subplot(3,1,1)
    plt.plot(t_vals.cpu(), food_arrays.cpu()[0,:])
    plt.plot(t_vals.cpu().cpu(), ppi_arrays_MPC.cpu()[0,0:n_steps+1])
    plt.xlabel('Timestep')
    plt.ylabel('Food and PPI')

    plt.subplot(3,1,2)
    plt.plot(t_vals.cpu(), reflux_all, color='r', linestyle='-', label='Reflux')
    plt.plot(t_vals.cpu(), digestive_all, color='b', linestyle='--', label='Digestive')
    plt.xlabel('Timestep')
    plt.ylabel('Symptom')

    plt.subplot(3,1,3)
    plt.plot(t_vals.cpu(), y_all[:,-1].cpu(), color='r', linestyle='--', label='Acid')
    plt.xlabel('Timestep')
    plt.ylabel('acid level')
    plt.legend()
    plt.show()

    if MPC_enabled: 
        data_path = "mpc_results"
        os.makedirs(data_path, exist_ok=True)
        # store ppi_mpc
        file_name = "ppi_mpc.pt"
        file_path = os.path.join(data_path, file_name)
        torch.save(ppi_arrays_MPC.cpu()[0,0:n_steps+1], file_path)
        # store reflux symptoms
        file_name = "reflux_mpc.pt"
        file_path = os.path.join(data_path, file_name)
        torch.save(reflux_all, file_path)
        # store digestive symptoms
        file_name = "digestive_mpc.pt"
        file_path = os.path.join(data_path, file_name)
        torch.save(digestive_all, file_path)
        # store hidden acid
        file_name = "acid_mpc.pt"
        file_path = os.path.join(data_path, file_name)
        torch.save(y_all[:,-1].cpu(), file_path)
    else:
        data_path = "mpc_results"
        os.makedirs(data_path, exist_ok=True)
        # store ppi_mpc
        file_name = "ppi_wo_mpc.pt"
        file_path = os.path.join(data_path, file_name)
        torch.save(ppi_arrays.cpu(), file_path)
        # store reflux symptoms
        file_name = "reflux_wo_mpc.pt"
        file_path = os.path.join(data_path, file_name)
        torch.save(reflux_all, file_path)
        # store digestive symptoms
        file_name = "digestive_wo_mpc.pt"
        file_path = os.path.join(data_path, file_name)
        torch.save(digestive_all, file_path)
        # store hidden acid
        file_name = "acid_wo_mpc.pt"
        file_path = os.path.join(data_path, file_name)
        torch.save(y_all[:,-1].cpu(), file_path)

    print("MPC ends")







    