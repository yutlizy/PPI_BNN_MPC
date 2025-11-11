import torch.nn as nn
import torch

class Detailed_Model(nn.Module):
    def __init__(
            self, 
            k_G,
            gamma_GS,
            delta_G,
            k_H,
            gamma_HS,
            delta_H,
            k_S,
            delta_S,
            nS,
            KS,
            alpha,
            K_Ghill,
            K_Hhill,
            m,
            n,
            beta_A,
            k_inhibit_detailed,
            k_regen_detailed,
            K_p,
            k_buffer,
            alpha_I,
            alpha_A,
            h
        ):
        super().__init__()
        # Detailed model parameters
        self.k_G       = k_G   # Max gastrin secretion rate
        self.gamma_GS  = gamma_GS    # S inhibition on G
        self.delta_G   = delta_G    # G clearance

        self.k_H       = k_H     # Histamine production from G
        self.gamma_HS  = gamma_HS    # S inhibition on H
        self.delta_H   = delta_H    # H clearance

        self.k_S       = k_S    # S production from acid
        self.delta_S   = delta_S    # S clearance
        self.nS        = nS 
        self.KS        = KS        # Half-sat for S

        self.alpha     = alpha     # Max acid secretory scale
        self.K_Ghill   = K_Ghill     # Half-sat for G synergy
        self.K_Hhill   = K_Hhill    # Half-sat for H synergy
        self.m         =   m 
        self.n         =  n  # exponents for synergy (could be > 1 for stronger synergy)
        self.beta_A    = beta_A     # acid clearance

        # Pump (P) parameters for irreversible PPI binding
        self.k_inhibit_detailed = k_inhibit_detailed     # rate of pump inactivation
        self.k_regen_detailed   = k_regen_detailed    # rate of pump regeneration
        self.K_p       = K_p     # saturable effect in p(t)

        # Optional meal-buffering
        self.k_buffer  = k_buffer    # how strongly the meal buffers acid

        self.alpha_I   = alpha_I   # represent sensitivity of gastrin to food input
        self.alpha_A   = alpha_A   # how acid level affect gastrin secretion
        self.h         = h

    def forward(self, t, Y, p_fn, meal_fn):
        """
        Y = [G, H, S, P, A]
        p_fn(t) => p(t), PPI schedule
        meal_fn(t) => I(t), meal input
        """
        # Unpack states
        G, H, S, P, A = Y


        # Evaluate PPI & meal
        p_val = p_fn(t)
        I_val = meal_fn(t)

        # 1) Pump fraction P
        p_eff = p_val/(self.K_p + p_val)
        dP = -self.k_inhibit_detailed*p_eff*P + self.k_regen_detailed*(1 - P)

        #  Gastrin
        # G up if acid is low & meal is present, down if S is high
        # E.g. G(t) = k_G * Phi_G(A, I) - gamma_GS*S*G - delta_G*G
        # We'll do a simple Phi_G(A, I) = (1 + alpha_I*I)/(1 + alpha_A*A)
        Phi_G = (1.0 + self.alpha_I*I_val) / (1.0 + self.alpha_A*A)
        dG = (self.k_G)*Phi_G - self.gamma_GS*S*G - self.delta_G*G

        # Histamine
        # H up by G, down by S, cleared
        fG_H = G**self.h / (self.K_Ghill**self.h + G**self.h + 1e-9)  # Saturating effect of gastrin
        dH = self.k_H * fG_H - self.gamma_HS * S * H - self.delta_H * H

        # (3) Somatostatin
        # S up if acid is high, cleared
        # e.g. S(t) = k_S * (A^n / (K + A^n)) - delta_S*S
        # We'll do a simple saturable form or just linear in A
        # For simplicity: saturable with exponent=1
        S_prod = self.k_S * (A**self.nS / (self.KS + A**self.nS + 1e-9))
        dS = S_prod - self.delta_S*S

        # (5) Acid load A
        # A up = alpha * P * fG(G)*fH(H) - clearance - meal buffering
        # fG(G) = G^m/(K_Ghill^m + G^m), similarly fH(H)
        fG = G**self.m / (self.K_Ghill**self.m + G**self.m + 1e-9)
        fH = H**self.n / (self.K_Hhill**self.n + H**self.n + 1e-9)
        acid_prod = self.alpha * P * fG * fH

        # acid removal
        acid_clear = self.beta_A*A

        # meal buffering
        acid_buffer = self.k_buffer*I_val*A

        dA = acid_prod - acid_clear - acid_buffer

        return torch.stack([dG, dH, dS, dP, dA])
    

    def forward_array(self, Y, p_val, I_val):
        """
        Y = [G, H, S, P, A]
        p_fn(t) => p(t), PPI schedule
        meal_fn(t) => I(t), meal input
        """
        # Unpack states
        G, H, S, P, A = Y

        # 1) Pump fraction P
        p_eff = p_val/(self.K_p + p_val)
        dP = -self.k_inhibit_detailed*p_eff*P + self.k_regen_detailed*(1 - P)

        #  Gastrin
        # G up if acid is low & meal is present, down if S is high
        # E.g. G(t) = k_G * Phi_G(A, I) - gamma_GS*S*G - delta_G*G
        # We'll do a simple Phi_G(A, I) = (1 + alpha_I*I)/(1 + alpha_A*A)
        Phi_G = (1.0 + self.alpha_I*I_val) / (1.0 + self.alpha_A*A)
        dG = (self.k_G)*Phi_G - self.gamma_GS*S*G - self.delta_G*G

        # Histamine
        # H up by G, down by S, cleared
        fG_H = G**self.h / (self.K_Ghill**self.h + G**self.h + 1e-9)  # Saturating effect of gastrin
        dH = self.k_H * fG_H - self.gamma_HS * S * H - self.delta_H * H

        # (3) Somatostatin
        # S up if acid is high, cleared
        # e.g. S(t) = k_S * (A^n / (K + A^n)) - delta_S*S
        # We'll do a simple saturable form or just linear in A
        # For simplicity: saturable with exponent=1
        S_prod = self.k_S * (A**self.nS / (self.KS + A**self.nS + 1e-9))
        dS = S_prod - self.delta_S*S

        # (5) Acid load A
        # A up = alpha * P * fG(G)*fH(H) - clearance - meal buffering
        # fG(G) = G^m/(K_Ghill^m + G^m), similarly fH(H)
        fG = G**self.m / (self.K_Ghill**self.m + G**self.m + 1e-9)
        fH = H**self.n / (self.K_Hhill**self.n + H**self.n + 1e-9)
        acid_prod = self.alpha * P * fG * fH

        # acid removal
        acid_clear = self.beta_A*A

        # meal buffering
        acid_buffer = self.k_buffer*I_val*A

        dA = acid_prod - acid_clear - acid_buffer

        return torch.stack([dG, dH, dS, dP, dA])
