import torch
import numpy as np
from models import Detailed_Model
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float32)

def generate_patient_parameters(num_profiles, device="cpu"):
    """
    Generate patient profiles directly on the specified device.

    Args:
        num_profiles: Number of patient profiles to generate.
        device: Device to run computations ('cpu', 'cuda', 'mps').

    Returns:
        profiles: Dictionary of parameters for all patients.
    """
    # Generate base parameters as PyTorch tensors on the GPU
    # k_G            = torch.rand(num_profiles, device=device) * (0.07 - 0.03) + 0.03    # Max gastrin secretion rate: around 0.05
    # gamma_GS       = torch.rand(num_profiles, device=device) * (0.05 - 0.01) + 0.01    # S inhibition on G: around 0.01
    # delta_G        = torch.rand(num_profiles, device=device) * (0.1 - 0.05) + 0.05     # G clearance: around 0.08

    # k_H            = torch.rand(num_profiles, device=device) * (0.4 - 0.1) + 0.1       # Histamine production from G: around 0.2
    # gamma_HS       = torch.rand(num_profiles, device=device) * (0.03 - 0.005) + 0.005  # S inhibition on H: around 0.01
    # delta_H        = torch.rand(num_profiles, device=device) * (0.1 - 0.05) + 0.05     # H clearance: around 0.07

    # k_S            = torch.rand(num_profiles, device=device) * (0.02 - 0.005) + 0.005  # S production from acid: around 0.01
    # delta_S        = torch.rand(num_profiles, device=device) * (0.06 - 0.04) + 0.004   # S clearance: around 0.05
    # nS             = torch.rand(num_profiles, device=device) * (2 - 1) + 1 
    # KS             = torch.rand(num_profiles, device=device) * (0.7 - 0.3) + 0.3       # Half-sat for S

    # alpha          = torch.rand(num_profiles, device=device) * (2 - 0.5) + 0.5         # Max acid secretory scale: around 1
    # K_Ghill        = torch.rand(num_profiles, device=device) * (0.7 - 0.3) + 0.3       # Half-sat for G synergy, around 0.5
    # K_Hhill        = torch.rand(num_profiles, device=device) * (0.7 - 0.3) + 0.3       # Half-sat for H synergy, around 0.5
    # m              = torch.rand(num_profiles, device=device) * (3 - 1) + 1             # exponents for synergy (could be > 1 for stronger synergy)
    # n              = torch.rand(num_profiles, device=device) * (3 - 1) + 1             # exponents for synergy (could be > 1 for stronger synergy)
    # beta_A         = torch.rand(num_profiles, device=device) * (0.15 - 0.05) + 0.05     # acid clearance, around 0.1

    # # Pump (P) parameters for irreversible PPI binding
    # k_inhibit_detailed      = torch.rand(num_profiles, device=device) * (0.4 - 0.2) + 0.2       # rate of pump inactivation
    # k_regen_detailed        = torch.rand(num_profiles, device=device) * (0.45 - 0.25) + 0.25    # rate of pump regeneration
    # K_p                     = torch.rand(num_profiles, device=device) * (0.15 - 0.05) + 0.05    # saturable effect in p(t)

    # # Optional meal-buffering
    # k_buffer       = torch.rand(num_profiles, device=device) * (0.09 - 0.01) + 0.01    # how strongly the meal buffers acid

    # alpha_I        = torch.rand(num_profiles, device=device) * (4 - 0.5) + 0.5         # represent sensitivity of gastrin to food input
    # alpha_A        = torch.rand(num_profiles, device=device) * (4 - 0.7) + 0.7         # how acid level affect gastrin secretion
    # h              = torch.rand(num_profiles, device=device) * (3 - 1) + 1


    # Max gastrin secretion rate: around 0.05
    k_G = np.random.uniform(0.04, 0.06, size=num_profiles).astype(np.float32)

    # S inhibition on G: around 0.01
    gamma_GS = np.random.uniform(0.008, 0.012, size=num_profiles).astype(np.float32)

    # G clearance: around 0.08
    delta_G = np.random.uniform(0.07, 0.09, size=num_profiles).astype(np.float32)

    # Histamine production from G: around 0.2
    k_H = np.random.uniform(0.17, 0.23, size=num_profiles).astype(np.float32)

    # S inhibition on H: around 0.01
    gamma_HS = np.random.uniform(0.008, 0.012, size=num_profiles).astype(np.float32)

    # H clearance: around 0.07
    delta_H = np.random.uniform(0.06, 0.08, size=num_profiles).astype(np.float32)

    # S production from acid: around 0.01
    k_S = np.random.uniform(0.008, 0.012, size=num_profiles).astype(np.float32)

    # S clearance: around 0.05
    delta_S = np.random.uniform(0.04, 0.06, size=num_profiles).astype(np.float32)

    # S sensitivity
    nS = np.random.uniform(1, 2, size=num_profiles).astype(np.float32)

    # Half-sat for S
    KS = np.random.uniform(0.4, 0.6, size=num_profiles).astype(np.float32)

    # Max acid secretory scale: around 1
    alpha = np.random.uniform(0.9, 1.1, size=num_profiles).astype(np.float32)

    # Half-sat for G synergy, around 0.5
    K_Ghill = np.random.uniform(0.4, 0.6, size=num_profiles).astype(np.float32)

    # Half-sat for H synergy, around 0.5
    K_Hhill = np.random.uniform(0.4, 0.6, size=num_profiles).astype(np.float32)

    # Exponents for synergy (could be > 1 for stronger synergy)
    m = np.random.uniform(1, 3, size=num_profiles).astype(np.float32)

    # Exponents for synergy (could be > 1 for stronger synergy)
    n = np.random.uniform(1, 3, size=num_profiles).astype(np.float32)

    # Acid clearance, around 0.1
    beta_A = np.random.uniform(0.08, 0.12, size=num_profiles).astype(np.float32)

    # Pump parameters for irreversible PPI binding
    # k_inhibit_detailed = np.random.uniform(0.3, 0.4, size=num_profiles).astype(np.float32)
    k_inhibit_detailed = np.random.uniform(0.3, 0.4, size=num_profiles).astype(np.float32)

    # Rate of pump regeneration
    # k_regen_detailed = np.random.uniform(0.3, 0.4, size=num_profiles).astype(np.float32)
    k_regen_detailed = np.random.uniform(0.3/17, 0.5/17, size=num_profiles).astype(np.float32)

    # Saturable effect in p(t)
    K_p = np.random.uniform(0.08, 0.12, size=num_profiles).astype(np.float32)

    # Optional meal-buffering
    k_buffer = np.random.uniform(0.04, 0.06, size=num_profiles).astype(np.float32)

    # Sensitivity of gastrin to food input
    alpha_I = np.random.uniform(1, 2, size=num_profiles).astype(np.float32)

    # How acid level affects gastrin secretion
    alpha_A = np.random.uniform(2, 3, size=num_profiles).astype(np.float32)

    # Gastrin secretion sensitivity
    h = np.random.uniform(1, 3, size=num_profiles).astype(np.float32)

    # Combine parameters into a dictionary
    profiles = {
        "k_G": k_G,
        "gamma_GS": gamma_GS,
        "delta_G": delta_G,
        "k_H": k_H,
        "gamma_HS": gamma_HS,
        "delta_H": delta_H,
        "k_S": k_S,
        "delta_S": delta_S,
        "nS": nS,
        "KS": KS,
        "alpha": alpha,
        "K_Ghill": K_Ghill,
        "K_Hhill": K_Hhill,
        "m": m,
        "n": n,
        "beta_A": beta_A,
        "k_inhibit_detailed": k_inhibit_detailed,
        "k_regen_detailed": k_regen_detailed,
        "K_p": K_p,
        "k_buffer": k_buffer,
        "alpha_I": alpha_I,
        "alpha_A": alpha_A,
        "h": h
    }

    return profiles

###############################################################################
# Generate random PPI and meals
###############################################################################
def generate_random_ppi_and_meal(t_vals, num_profiles, ppi_intervals=3, device="cpu"):
    """
    Create random PPI arrays and meal arrays in advance (per patient),
    storing them for direct lookup during ODE solves.

    Arguments:
    ----------
    t_vals: torch.Tensor of shape [n_steps+1], the time vector used for integration.
    num_profiles: int, number of patient profiles.
    device: str, device for storing tensors.

    Returns:
    --------
    ppi_arrays: torch.Tensor, shape [num_profiles, n_steps+1], the p(t) schedules.
    meal_arrays: torch.Tensor, shape [num_profiles, n_steps+1], the meal(t) schedules.
    """
    n_steps_plus_1 = t_vals.shape[0]

    # We'll store ppi_arrays and meal_arrays
    ppi_arrays = torch.zeros(num_profiles, n_steps_plus_1, device=device)
    meal_arrays = torch.zeros(num_profiles, n_steps_plus_1, device=device)
    ppi_days = torch.zeros(num_profiles, 2*ppi_intervals, device=device)

    # Example: random PPI usage windows or intensities
    for i in range(num_profiles):
        # For demonstration, we randomly pick a start day for PPI:
        for ppi_interval_idx in range(ppi_intervals):
            ppi_start = np.random.uniform(100, 101) + ppi_interval_idx*250
            ppi_end   = ppi_start + np.random.uniform(50, 60)  # random PPI window length
            # ppi_end   = ppi_start + 56  # random PPI window length

            # Convert to torch for indexing
            ppi_start = torch.tensor(ppi_start, device=device)
            ppi_end   = torch.tensor(ppi_end, device=device)
            ppi_days[i, 0+2*ppi_interval_idx] = ppi_start
            ppi_days[i, 1+2*ppi_interval_idx] = ppi_end

        intervals = [(ppi_days[0, i], ppi_days[0, i+1]) 
            for i in range(0, 2*ppi_intervals, 2)]
        # Fill ppi_arrays[i] by scanning t_vals
        for j in range(n_steps_plus_1):
            t_ = t_vals[j]
            for start, end in intervals:
                if (t_ >= start) & (t_ < end):
                    ppi_arrays[i, j] = 1
                # ppi_arrays[i, j] = ((t_ >= start) & (t_ < end)).float()

        # Now define a random meal profile for each day,
        # For demonstration, we just do some random offset or amplitude each day
        # We'll do a simple approach: each day has a random amplitude
        # In a real scenario, you'd do something more sophisticated
        
        base_meal_centers = [8.0/24.0, 12.0/24.0, 18.0/24.0]

        # For each patient, for each day integer, define random offsets & amplitudes 
        # for the 3 daily meals, then apply to all t_ in that day.
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
    return ppi_arrays, meal_arrays, ppi_days

###############################################################################
# Symptom model
###############################################################################
def acid_reflux_score(a, a_high=1.2, k_r=20):
    """
    Compute the acid reflux score based on the acid level.
    
    Parameters:
        a (float or np.ndarray): Acid level(s)
        a_high (float): Upper threshold for acid level.
        k_r (float): Steepness parameter for reflux sigmoid.
    
    Returns:
        np.ndarray: Acid reflux score(s) between 1 and 10.
    """
    # Sigmoid function mapping acid level to a score between 0 and 1,
    # then scaled to a range of 1 to 10.
    score = 1 + 9 * (1 / (1 + np.exp(-k_r * (a - a_high))))
    return score

def digestive_issue_score(a, a_low=0.8, k_d=20):
    """
    Compute the digestive issue score based on the acid level.
    
    Parameters:
        a (float or np.ndarray): Acid level(s)
        a_low (float): Lower threshold for acid level.
        k_d (float): Steepness parameter for digestive sigmoid.
    
    Returns:
        np.ndarray: Digestive issue score(s) between 1 and 10.
    """
    # Sigmoid function mapping acid level to a score between 0 and 1,
    # inverted to increase score as acid level drops.
    score = 1 + 9 * (1 / (1 + np.exp(k_d * (a - a_low))))
    return score

def noisy_acid_reflux_score(a, a_high=1.2, k_r=20, noise_std=0.2):
    """
    Compute the acid reflux score with added Gaussian noise.
    
    Parameters:
        a (float or np.ndarray): Acid level(s)
        a_high (float): Upper threshold for acid level.
        k_r (float): Steepness parameter for reflux sigmoid.
        noise_std (float): Standard deviation of Gaussian noise.
    
    Returns:
        np.ndarray: Noisy acid reflux score(s), clipped to [1, 10].
    """
    base_score = acid_reflux_score(a, a_high, k_r)
    noisy_score = base_score + np.random.normal(0, noise_std, size=np.shape(a))
    noisy_score = np.floor(noisy_score)
    # Ensure the score remains within [1, 10]
    return np.clip(noisy_score, 1, 10)

def noisy_digestive_issue_score(a, a_low=0.8, k_d=20, noise_std=0.2):
    """
    Compute the digestive issue score with added Gaussian noise.
    
    Parameters:
        a (float or np.ndarray): Acid level(s)
        a_low (float): Lower threshold for acid level.
        k_d (float): Steepness parameter for digestive sigmoid.
        noise_std (float): Standard deviation of Gaussian noise.
    
    Returns:
        np.ndarray: Noisy digestive issue score(s), clipped to [1, 10].
    """
    base_score = digestive_issue_score(a, a_low, k_d)
    noisy_score = base_score + np.random.normal(0, noise_std, size=np.shape(a))
    noisy_score = np.floor(noisy_score)
    # Ensure the score remains within [1, 10]
    return np.clip(noisy_score, 1, 10)


###############################################################################
# Fixed-Step RK4 Integrator (Aligned to day boundaries)
###############################################################################
def rk4_solve(f, y0, t0, t_end, dt):
    """
    Fixed-step RK4. 
    Each step is dt days. 
    The number of steps = int((t_end - t0)/dt).
    """
    device = y0.device
    n_steps = int(round((t_end - t0)/dt))

    t_vals = torch.linspace(t0, t_end, steps=n_steps+1, device=device, dtype=y0.dtype)
    y_vals = torch.empty(n_steps+1, y0.shape[0], device=device, dtype=y0.dtype)

    y_vals[0] = y0

    for i in range(n_steps):
        t_current = t_vals[i]
        Y_current = y_vals[i]

        k1 = f(t_current, Y_current)
        k2 = f(t_current + 0.5*dt, Y_current + 0.5*dt*k1)
        k3 = f(t_current + 0.5*dt, Y_current + 0.5*dt*k2)
        k4 = f(t_current + dt,     Y_current + dt*k3)

        y_vals[i+1] = Y_current + (dt / 6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # return t_vals.cpu().numpy(), y_vals.cpu().numpy()
    return t_vals, y_vals

def generate_data(parameter_dict, t0, t_end, day_step): 
    n_steps = int((t_end - t0)/day_step)
    t_vals = torch.linspace(t0, t_end, n_steps+1, device=device)

    # 2) Generate random ppi_arrays and meal_arrays
    ppi_arrays, meal_arrays, ppi_days_array = generate_random_ppi_and_meal(t_vals, num_profiles, device=device)
    
    all_y_vals = []
    all_symptoms = []
    all_constraints = []

    for i in range(num_profiles):
        # Instantiate Detailed_Model with parameter_dict
        args = [param[i] for param in parameter_dict.values()]
        model = Detailed_Model(*args)

        # initial condition
        y0 = torch.tensor([0.15, 0.4, 0.1, 1.0, 1.0], device=device)

        # define the ODE func with array lookup
        def ode_func(t, Y):
            # we do direct indexing
            i_float = (t - t0)/day_step
            i_idx   = int(torch.round(i_float).item())
            i_idx   = max(0, min(i_idx, n_steps))  # clamp

            p_val   = ppi_arrays[i, i_idx]
            meal_val= meal_arrays[i, i_idx]

            # now pass p_val and meal_val into the model
            # we create small wrapper for model:
            return model.forward_array(Y, p_val, meal_val)

        t_vals, y_vals = rk4_solve(ode_func, y0, t0, t_end, day_step)

        acid_equ = y_vals[int(100/day_step), -1].cpu().numpy()
        
        a_high = acid_equ*.9
        a_low = acid_equ*.5


        reflux_scores = noisy_acid_reflux_score(y_vals[:,-1].cpu().numpy(), a_high=a_high, k_r=20, noise_std=0.2)
        digestive_scores = noisy_digestive_issue_score(y_vals[:,-1].cpu().numpy(), a_low=a_low, k_d=20, noise_std=0.2)
        
        for i in range(1, len(reflux_scores)):
            if i % int(1/day_step) != 0:
                reflux_scores[i] = reflux_scores[i-1]
                digestive_scores[i] = digestive_scores[i-1]


        scores_tensor = torch.tensor([np.array(reflux_scores), np.array(digestive_scores)])
        constraints_tensor = torch.tensor([(a_high), (a_low)])

        all_symptoms.append(scores_tensor)
        all_constraints.append(constraints_tensor)

        all_y_vals.append(y_vals)


    all_y_vals = torch.stack(all_y_vals, dim=0)
    all_symptoms = torch.stack(all_symptoms, dim=0)
    all_constraints = torch.stack(all_constraints, dim=0)

    return all_y_vals, ppi_arrays, meal_arrays, t_vals, ppi_days_array, all_symptoms, all_constraints

if __name__== "__main__":
    num_profiles = 1
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"The device is {device}")
    parameters = generate_patient_parameters(num_profiles, device=device)
    for i in parameters.keys():
        print(f"{i}: {parameters[i]}")


    t_end      = 500
    day_step   = 1.0/24.0 

    data, ppi_profile, meal_profile, t_hist, ppi_days_array, symptoms, constraints = generate_data(parameter_dict=parameters, t0=0, t_end=t_end, day_step=day_step)

    store_start_day = 75
    data_path = "synthetic_data"
    os.makedirs(data_path, exist_ok=True)
    # store parameters
    file_name = "parameters.pt"
    file_path = os.path.join(data_path, file_name)
    torch.save(parameters, file_path)
    # store acid traj
    file_name = "acid_traj.pt"
    file_path = os.path.join(data_path, file_name)
    torch.save(data[:,int(store_start_day/day_step):,-1], file_path)
    # ppi traj
    file_name = "ppi_traj.pt"
    file_path = os.path.join(data_path, file_name)
    torch.save(ppi_profile[:,int(store_start_day/day_step):], file_path)
    # food traj
    file_name = "food_traj.pt"
    file_path = os.path.join(data_path, file_name)
    torch.save(meal_profile[:,int(store_start_day/day_step):], file_path)
    # time traj
    file_name = "time_traj.pt"
    file_path = os.path.join(data_path, file_name)
    torch.save(t_hist[0:len(t_hist)-int(store_start_day/day_step)], file_path)
    # store symptom traj
    file_name = "symptom_traj.pt"
    file_path = os.path.join(data_path, file_name)
    torch.save(symptoms[:,:,int(store_start_day/day_step):], file_path)
    # store constraints traj
    file_name = "constraints_traj.pt"
    file_path = os.path.join(data_path, file_name)
    torch.save(constraints[:,:], file_path)





    for i in range(data.shape[0]):
        G_pat       = data[i,:,0].cpu().numpy()
        H_pat       = data[i,:,1].cpu().numpy()
        S_pat       = data[i,:,2].cpu().numpy()
        Pdetail_pat = data[i,:,3].cpu().numpy()
        Adetail_pat = data[i,:,4].cpu().numpy()
        PPI_starts  = ppi_days_array[i, 0].cpu().numpy().item()
        PPI_ends    = ppi_days_array[i, 1].cpu().numpy().item()
        ppi_pat     = ppi_profile[i,:].cpu().numpy()
        meal_apt    = meal_profile[i,:].cpu().numpy()
        reflux      = symptoms[i,0,:].cpu().numpy()
        digestive   = symptoms[i,1,:].cpu().numpy()


        plt.figure(figsize=(16,9))
        t_pat_detail = t_hist.cpu().numpy()
        
        # (A) Pump fraction
        plt.subplot(2,3,1)
        # plt.plot(t_norm_detail, Pdetail_norm, label='P wo PPI', color='g')
        plt.plot(t_pat_detail, Pdetail_pat, label='P with PPI', color='r')
        plt.axvline(PPI_starts, color='k', linestyle=':', label=f'PPI starts at {int(PPI_starts)} and ends at {int(PPI_ends)}')
        plt.axvline(PPI_ends, color='k', linestyle=':')
        # plt.xlim([PPI_starts-10, t_end]) 
        plt.title('Proton Pump Fraction')
        plt.xlabel('Time (days)')
        plt.ylabel('P')
        plt.grid(True)
        plt.legend()

        # (B) Acid
        plt.subplot(2,3,2)
        # plt.plot(t_norm_detail, Adetail_norm, label='Acid wo PPI', color='g')
        plt.plot(t_pat_detail, Adetail_pat, label='Acid with PPI', color='r')
        plt.axvline(PPI_starts, color='k', linestyle=':')
        plt.axvline(PPI_ends, color='k', linestyle=':')
        # plt.xlim([PPI_starts-10, t_end])  
        # plt.ylim([min(Adetail_pat[int((PPI_starts-10)/day_step):])*0.9, max(Adetail_pat[int((PPI_starts-10)/day_step):])*1.1])
        plt.title('Acid Load')
        plt.xlabel('Time (days)')
        plt.ylabel('A')
        plt.grid(True)
        plt.legend()

        # (C) Hormones
        plt.subplot(2,3,3)
        # plt.plot(t_norm_detail, G_norm, label='Gastrin wo PPI', color='b')
        # plt.plot(t_norm_detail, H_norm, label='Histamine wo PPI', color='orange')
        # plt.plot(t_norm_detail, S_norm, label='Somatostatin wo PPI', color='purple')
        plt.plot(t_pat_detail, G_pat, label='Gastrin with PPI', color='b', linestyle='--')
        plt.plot(t_pat_detail, H_pat, label='Histamine with PPI', color='orange', linestyle='--')
        plt.plot(t_pat_detail, S_pat, label='Somatostatin with PPI', color='purple', linestyle='--')
        plt.axvline(PPI_starts, color='k', linestyle=':')
        plt.axvline(PPI_ends, color='k', linestyle=':')
        # plt.xlim([PPI_starts-10, t_end])  
        plt.ylim([min(S_pat[int((PPI_starts-10)/day_step):])*0.9, max(H_pat[int((PPI_starts-10)/day_step):])*1.1])
        plt.title('G, H, S')
        plt.xlabel('Time (days)')
        plt.grid(True)
        plt.legend()

        # (D) PPI and meal profile
        plt.subplot(2,3,4)
        plt.plot(t_pat_detail, meal_apt, label='Meal', color='g')
        plt.plot(t_pat_detail, ppi_pat, label='PPI', color='r')
        # plt.xlim([PPI_starts-10, t_end])
        plt.title('PPI, Meals')
        plt.xlabel('Time (days)')
        plt.grid(True)
        plt.legend()

        # (E) symptoms
        plt.subplot(2,3,5)
        plt.plot(t_pat_detail, reflux, label='reflux', color='g')
        plt.plot(t_pat_detail, digestive, label='digestive', color='r')
        # plt.xlim([PPI_starts-10, t_end])
        plt.title('Symptom scores')
        plt.xlabel('Time (days)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
    print("!!!Code finishes here!!!")
