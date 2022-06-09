import os

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "../phase_2/results/")


def save_results(label, results):
    filename = f'{label}_results.txt'
    results_path = os.path.join(RESULTS_PATH, filename)
    with open(results_path, 'w+') as f:
        for item in results:
            f.write("%s\n" % item)

