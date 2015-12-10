import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(12,10))
    colors = ['#1446A0', '#B0228C', '#EA3788', '#F18F01', '#76AE0A']
    problem_sizes = [int(p) for p in np.logspace(4, 8, num=5, base=2)]
    num_procs = np.logspace(0, 6, num=7, base=2)
    for idx, problem_size in enumerate(problem_sizes):
        benchmarks = np.loadtxt('all-times-{}'.format(problem_size))
        serial = np.loadtxt('time-serial_{}.out'.format(problem_size))
        serial_line = plt.axhline(serial, 0., 1., c=colors[idx], ls='--')
        plt.loglog(num_procs[:len(benchmarks)], benchmarks, c=colors[idx], label='{0} x {0} x {0} grid'.format(problem_size))

    plt.legend(loc='best')
    ax.set_xticks(num_procs)
    ax.set_xticklabels([int(n) for n in num_procs])
    plt.xlabel('Number of processors')
    plt.ylabel('Time in seconds')
    plt.title('Running time vs. number of processors')
    plt.show()
