import os

with open("./mnist_data_jpg/mnist_test_jpg_10000.txt", "w") as f_out:
    for file in os.listdir(".\mnist_data_jpg\mnist_test_jpg_10000"):
        f_out.write(file + " " + file.split(".")[0][-1] + "\n")
        
with open("./mnist_data_jpg/mnist_train_jpg_60000.txt", "w") as f_out:
    for file in os.listdir(".\mnist_data_jpg\mnist_train_jpg_60000"):
        f_out.write(file + " " + file.split(".")[0][-1] + "\n")