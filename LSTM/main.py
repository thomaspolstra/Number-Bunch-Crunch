import sys
from experiment import Experiment

if __name__ == '__main__':
    exp_name = 'default'

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]

    print('Running Experiment: ', exp_name)
    exp = Experiment(exp_name)
    exp.run()
    print('Beginning Testing')
    exp.test()
