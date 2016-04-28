
import matplotlib.pyplot as plt
import numpy as np

# #
# # Simulated Annealer
# # From my homework 3 solutions.
# #


def simulate_annealing(model, energy_meter, stepper,
                       init_temp=2.0, thermostat=0.9,
                       annealing_interval=100, max_iterations=10000, stopping_energy=None, stopping_quiescence=None,
                       doPlot=True, showPlot=True, doPrint=True):
    """
    Uses simulated annealing to find a minimum-energy state for a model.

    @param model: The model to optimize state.
    @param energy_meter: A function taking the model state and returning an energy for the model.
    @param stepper: A function taking the model state and an integer intensity parameter and returning a new model state.
    @param init_temp: The initial temperature of the annealing system.
    @param thermostat: The ratio by which the system will cool.
    @param annealing_interval: The number of iterations between cooling events.
    @param max_iterations: Patience threshold for iterations.
    @param stopping_energy: Lower bound on energy, which signals it's time to stop.
    @param stopping_quiescence: A minimum energy fluctuation. Less than this signals it's time to stop.
    @return The optimal model state found.
    """
    # Sliding window to check for the energy estimates settling down.
    qui_win = 5000

    # Set initial values
    pos = 1
    energy_trace = [energy_meter(model),]
    reheats = []
    thawed = False
    best = model.copy()
    temperature = init_temp

    for itr in xrange(max_iterations):

        # Adaptive step size
        step_size = np.max((np.floor(np.sqrt(temperature)).astype(int),1))
        proposed_items = stepper(model, step_size)
        energy = energy_meter(proposed_items)
        energy_diff = energy - energy_trace[pos-1]

        # Test for acceptance based on step towards optimum, or a weighted coin flip.
        coin_weight = np.exp(-energy_diff/temperature)
        if (energy_diff < 0) or (np.random.rand() < coin_weight):
            energy_trace.append(energy)
            model = proposed_items
            thawed = True
            pos += 1
            if energy < np.min(energy_trace[:-1]):
                best = proposed_items.copy()
                if (stopping_energy is not None) and energy <= stopping_energy:
                    break

        # Maybe perform some cooling.
        if pos % annealing_interval == 0:
            if thawed:
                thawed = False
                temperature *= thermostat
                if doPrint:
                    print "cooling. t=%.06f" % temperature
                # If the temperature gets too low, reheat.
                if temperature < init_temp/50.:
                    temperature = init_temp
                    if pos not in reheats:
                        reheats.append(pos)
                        if doPrint:
                            print "reheating. pos=%d, itr=%d" % (pos,itr)

        if (stopping_quiescence is not None) and (len(energy_trace) > qui_win):
            if np.std(energy_trace[-qui_win:])/np.mean(energy_trace[-qui_win:]) < stopping_quiescence:
                break

    if doPlot:
        plt.plot(energy_trace, label="Configuration energy")
        plt.title("Simulated Annealing")
        plt.xlabel("Configurations")
        plt.ylabel("Energy")
        if stopping_energy is not None:
            plt.ylim(ymin=stopping_energy)
        [plt.axvline(r, color='r', label="Reheat") for r in reheats]
        plt.legend()
        if showPlot:
            plt.show()
    return best
