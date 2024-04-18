def read_from_txt(file_name: str):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    data = [float(line.strip()) for line in lines]

    return data


if __name__ == '__main__':
    file_name = '../datas/Neuron04a.txt'
    print(read_from_txt(file_name))