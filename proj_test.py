from config import NUM_TRAINING_ITERATIONS, CONVERGENCE_THRESHOLD
from formulas import err
from models import Layer, cfile

f = None
curr_point = 0
target = []
attrs = []
total_runs = 0
data = None
num_incorrect = 0
prev_sample_err = 0
curr_sample_err = 0


def parse_data(fname):
    global curr_point, total_runs, target, attrs, num_incorrect
    global prev_sample_err, curr_sample_err, data, f

    curr_point = 0
    total_runs = 0
    target = []
    attrs = []
    num_incorrect = 0
    prev_sample_err = 0
    curr_sample_err = 0

    # select output filename based on phase
    if fname == 'training.txt':
        data_file = 'training_err.txt'
    elif fname == 'val.txt':
        data_file = 'val_err.txt'
    elif fname == 'testing.txt':
        data_file = 'testing_err.txt'
    else:
        data_file = 'err.txt'

    open(data_file, 'w+').close()     # clear log file
    data = cfile(data_file, 'w')      # open file for error logging

    f = open(fname, 'r').readlines()

    for row in f:
        row = [int(x.strip()) for x in row.split()]
        target.append(row[0:2])       # class in 2-bit vector format
        attrs.append(row[2:])         # binary feature vector

    print("DEBUG TARGET VALUES (FIRST 10):", target[:10])
    input("Press ENTER to continue...")


if __name__ == '__main__':

    print("Parsing the training dataset...")
    parse_data('training.txt')

    # construct network layout
    input_size = len(attrs[0])
    x = Layer(6, [0] * input_size, 1)   # hidden layer
    y = Layer(2, [0] * 6, 2)


    print("Beginning training the neural network:")
    while total_runs < NUM_TRAINING_ITERATIONS:

        x.input_vals = attrs[curr_point]
        x.eval()

        y.input_vals = x.layer_out
        y.eval()

        curr_err = err(y.layer_out, target[curr_point])
        y.backprop(target[curr_point])
        x.backprop(y)

        pred_index = y.layer_out.index(max(y.layer_out))
        true_index = target[curr_point].index(1)


        if pred_index != true_index:
            num_incorrect += 1

        if total_runs % 100 == 0:
            prev_sample_err = curr_sample_err
            curr_sample_err = curr_err
            if abs(prev_sample_err - curr_sample_err) < CONVERGENCE_THRESHOLD:
                print("Data has converged at iteration", total_runs)
                break

        print("Current iteration:", total_runs)
        print("Current error:", curr_err, "\n")
        data.w(curr_err)

        total_runs += 1
        curr_point += 1

        if curr_point >= len(f):
            curr_point = 0

    data.close()
    print("Neural network is done training! Hit enter for validation.")
    print("Error percentage on training set:", float(num_incorrect) / NUM_TRAINING_ITERATIONS)
    input()

    # -------------------------------- VALIDATION

    total_runs = 0
    num_incorrect = 0
    print("Parsing the validation dataset...")
    parse_data('val.txt')

    print("Beginning validation:")
    while curr_point < len(f):

        x.input_vals = attrs[curr_point]
        x.eval()
        y.input_vals = x.layer_out
        y.eval()

        # FIXED: use full output vector instead of first neuron only
        curr_err = err(y.layer_out, target[curr_point])

        pred_index = y.layer_out.index(max(y.layer_out))
        true_index = target[curr_point].index(1)


        if pred_index != true_index:
            num_incorrect += 1

        print("Current iteration:", total_runs)
        print("Current error:", curr_err, "\n")
        data.w(curr_err)

        total_runs += 1
        curr_point += 1

    data.close()
    print("Validation complete.")
    print("Error percentage on validation set:", float(num_incorrect) / NUM_TRAINING_ITERATIONS)
    print("Model Accuracy:", 1.0 - float(num_incorrect) / NUM_TRAINING_ITERATIONS)
    input()

    # -------------------------------- TESTING 

    total_runs = 0
    num_incorrect = 0
    curr_point = 0

    print("Begin testing the neural network:")
    parse_data('testing.txt')

    phase_name = "TESTING PHASE"
    while curr_point < len(f):

        x.input_vals = attrs[curr_point]
        x.eval()
        y.input_vals = x.layer_out
        y.eval()

        if total_runs < 3:  # print first few samples only
            print("---- DEBUG SAMPLE ----")
            print("Phase:", phase_name)
            print("Input length:", len(x.input_vals))
            print("Target vector:", target[curr_point])
            print("Network output:", y.layer_out)

        # FIXED: use full output vector here as well
        curr_err = err(y.layer_out, target[curr_point])

        pred_index = y.layer_out.index(max(y.layer_out))
        true_index = target[curr_point].index(1)


        if pred_index != true_index:
            num_incorrect += 1

        print("Current iteration:", total_runs)
        print("Current Error:", curr_err, "\n")
        data.w(curr_err)

        total_runs += 1
        curr_point += 1

    data.close()
    print("Testing done! Check generated output files.")
    print("Error percentage on testing set:", float(num_incorrect) / len(f))
    print("Model Accuracy:", 1.0 - float(num_incorrect) / len(f))
