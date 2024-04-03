def parse_training_data(x, y, out):
    x_train, y_train = [], []
    with open("training.data", 'r') as file:
        for line in file.readlines():
            data = line.split()
            pixels = []
            for i in range(y):
                row = []
                for j in range(x):
                    row.append(int(data[i*x + j]))
                pixels.append(row)
            answer = [0 for _ in range(out)]
            answer[int(data[-1])] = 1
            
            x_train.append(pixels)
            y_train.append(answer)
            
    return x_train, y_train


def parse_line(line, x, y):
    data = line.split()
    pixels = []
    for i in range(y):
        row = []
        for j in range(x):
            row.append(int(data[i*x + j]))
        pixels.append(row)
    
    return pixels