import configparser
import pickle
import socket

CONFIG_PATH = "client.ini"


class Client:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        self.__host = config["DEFAULT"]["HOST"]
        self.__port = int(config["DEFAULT"]["PORT"])
        self.__socket = socket.socket()
        self.__no_connection_exit = config["DEFAULT"]["NO_CONNECTION_EXIT"] == "0"

    def __connect(self):
        try:
            if self.__socket is not None:
                self.__socket.close()
            self.__socket = socket.socket()
            self.__socket.connect((self.__host, self.__port))
            print("Connection on {}".format(self.__port))
        except ConnectionRefusedError:
            print(
                f"Please check connection setting and check if the servser application is running on {self.__host}:{self.__port}")
            if self.__no_connection_exit:
                exit(1)

    def send_data(self, class_code, pin_width):
        serialized_data = pickle.dumps((class_code, pin_width))
        while True:
            try:
                self.__socket.send(serialized_data)
                break
            except BrokenPipeError:
                self.__connect()