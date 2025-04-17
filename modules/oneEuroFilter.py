import math 

# Helper func to calculate alpha from cutoff frequency 
def smoothing_factor(t_e, cutoff):
    """calculate the smoothing factor alpha for a given time interval and cutoff frequency"""
    r = 2 * math.pi * cutoff * t_e 
    return r / (r + 1)

# Helper for exponential smoothing 
def exponential_smoothing(alpha, x, x_prev):
    """Performs exponential smoothing"""
    return alpha * x + (1 - alpha) * x_prev 

class OneEuroFilter:
    """
    A simple One-Euro Filter implementation for smoothing signals (e.g., coordinates).

    Relevant Parameters:
        min_cutoff: Minimum cutoff frequency. Lower values increase smoothing (reduce jitter) 
                    at low speeds. Higher values decrease smoothing.
        beta:       Determines how aggressively smoothing decreases as speed increases. 
                    Higher values reduce lag during fast movements. Lower values keep 
                    smoothing more consistent.
        d_cutoff:   Cutoff frequency for the derivative signal (usually okay at 1.0).
    """
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Init the one euro filter"""
        # Check params 
        if min_cutoff <= 0: raise ValueError("min cutoff must be > 0")
        if d_cutoff <= 0: raise ValueError("d_cutoff must be > 0")

        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        # Previous values 
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal for timestamp t and value x"""
        # Ensure t is greater than t_prev 
        if t <= self.t_prev:
            print(f"Warning: Timestamp {t} not greater than previous {self.t_prev}")
            return self.x_prev

        t_e = t - self.t_prev # time elapsed since last module 

        # --- filter the derivative --- 
        # smoothing factor for derivative (alpha_d)
        a_d = smoothing_factor(t_e, self.d_cutoff)
        # calculate the derivative (dx)
        dx = (x - self.x_prev) / t_e 
        # filter the derivative (edx)
        edx = exponential_smoothing(a_d, dx, self.dx_prev)

        # --- filter the value --- 
        # calc the cutoff freq based on the filtered derivative's speed 
        cutoff = self.min_cutoff + self.beta * abs(edx)
        # calculate the smoothing factor for the value (alpha)
        a = smoothing_factor(t_e, cutoff)
        # filter the value (ex)
        ex = exponential_smoothing(a, x, self.x_prev)

        # --- store current state for next iteration --- 
        self.x_prev = ex 
        self.dx_prev = edx 
        self.t_prev = t 

        return ex 

    def reset(self, t0, x0):
        """Resets the filter's state"""
        self.x_prev = float(x0) 
        self.dx_prev = 0.0 
        self.t_prev = float(t0)
