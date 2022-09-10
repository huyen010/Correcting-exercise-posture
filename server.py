import time
from ast import While
import socket
import tqdm
import os
import predict
import truefalse
# device's IP address
#SERVER_HOST = "172.20.10.6"
import pymysql
connection = pymysql.connect(host="localhost", port=3306, user="root", passwd="", database="pose_database")
cursor = connection.cursor()

data_folder = "Data/TEST"
listfd = []
for folder_name in os.listdir(data_folder):
    listfd.append(folder_name)
SERVER_HOST = "172.20.10.4"
SERVER_PORT = 5002
# receive 4096 bytes each time
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"
# create the server socket
# TCP socket
filename = " "
while(True):
    s = socket.socket()
    # bind the socket to our local address
    s.bind((SERVER_HOST, SERVER_PORT))
     # enabling our server to accept connections
    # 5 here is the number of unaccepted connections that
    # the system will allow before refusing new connections
    s.listen(5)
    print(f"[*] Listening as {SERVER_HOST}:{SERVER_PORT}")
    # accept connection if there is any
    client_socket, address = s.accept()
    # if below code is executed, that means the sender is connected
    print(f"[+] {address} is connected.")
    # r eceive the file infos
    # receive using client socket, not server socket
    received = client_socket.recv(BUFFER_SIZE).decode()
    filename, filesize = received.split(SEPARATOR)
    # remove absolute path if there is
    filename = os.path.basename(filename)
    # convert to integer
    filesize = int(filesize)
    # start receiving the file from the socket
    # and writing to the file stream
    progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "wb") as f:
        while True:
            # read 1024 bytes from the socket (receive)
            bytes_read = client_socket.recv(BUFFER_SIZE)
            if not bytes_read:
                # nothing is received
                # file transmitting is done
                break
            # write to the file the bytes we just received
            f.write(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))
    # close the client socket
    client_socket.close()
    # close the server socket
    s.close()
    break
time.sleep(20)
HOST = '172.20.10.7'    # Cấu hình address server
PORT = 8000              # Cấu hình Port sử dụng
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Cấu hình socket
s.connect((HOST, PORT)) # tiến hành kết nối đến server
try:
    pred = predict.predict(filename)
    print(pred)
    if pred != 0:
        name = listfd[pred-1]
        print(name)
        check = truefalse.check(filename, name)
        print(check)
        name1 = name + check
        # insert = (
        #    "INSERT INTO course_pose_history (`name`, `Check`, `image`) VALUES (%s, %s, %s)"
        # )
        # val = (name, check, filename)
        # cursor.execute(insert, val)
        # connection.commit()
        s.sendall(bytes(name1, encoding='utf-8'))  # Gửi dữ liệu lên server
    else:
        s.sendall(b'K nhan dang')
except:
    print('name')
    s.sendall(b'K nhan dang')
data = s.recv(1024) # Đọc dữ liệu server trả về
print('Server Respone: ', repr(data))
s.close()
