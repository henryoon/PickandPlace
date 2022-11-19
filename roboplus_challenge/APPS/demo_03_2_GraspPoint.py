from signal import signal, SIGPIPE, SIG_DFL
# Ignore SIG_PIPE and don't throw exceptions on it... (http://docs.python.org/library/signal.html)
signal(SIGPIPE, SIG_DFL)
import socket
import cv2
import time
import datetime
import numpy
import numpy as np
import copy
import base64

class GraspDetector:
    def __init__(self, ip, port):
        self.TCP_SERVER_IP = ip
        self.TCP_SERVER_PORT = port
        self.connectCount = 0
        self.rgb_data=[]
        self.connectServer()
        self.set_parameter([0.8,70,80,5000,200,10])
        time.sleep(1)

    def set_parameter(self,param):
        self.param=copy.deepcopy(param)

    def get_data(self,rgb_data, depth):
        # send param
        stringData = base64.b64encode(np.array(self.param))
        length = str(len(stringData))
        # image data
        self.sock.sendall(length.encode('utf-8').ljust(64))
        self.sock.send(stringData)

        # send image
        send_data = numpy.dstack((rgb_data, depth))
        stime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        stringData = base64.b64encode(send_data)
        length = str(len(stringData))

        # image data
        self.sock.sendall(length.encode('utf-8').ljust(64))
        self.sock.send(stringData)
        self.sock.send(stime.encode('utf-8').ljust(64))

        # image size
        stringData = base64.b64encode(np.array(rgb_data.shape))
        length = str(len(stringData))
        self.sock.sendall(length.encode('utf-8').ljust(64))
        self.sock.send(stringData)

        #recv
        #grips
        rec_data = self.sock.recv(64)
        length1 = rec_data.decode('utf-8')
        rec_data = self.recvall(self.sock, int(length1))
        grips = np.frombuffer(base64.b64decode(rec_data), np.float)
        grips=grips.reshape([int(grips.size / 11), 11])
        detected = {'grip': grips}

        # image
        rec_data = self.sock.recv(64)
        length1 = rec_data.decode('utf-8')
        rec_data = self.recvall(self.sock, int(length1))
        img_data = np.frombuffer(base64.b64decode(rec_data), np.uint8)
        # size
        rec_data = self.sock.recv(64)
        length1 = rec_data.decode('utf-8')
        rec_data = self.recvall(self.sock, int(length1))
        img_size = np.frombuffer(base64.b64decode(rec_data), np.int)

        result_image = img_data.reshape(np.append(img_size[0:2], [3]))
        detected.update({'im': result_image})

        # best_index
        rec_data = self.sock.recv(64)
        best_index = rec_data.decode('utf-8')
        detected.update({'best_ind': best_index})

        # best_n_inds
        rec_data = self.sock.recv(64)
        length1 = rec_data.decode('utf-8')
        rec_data = self.recvall(self.sock, int(length1))
        best_n_inds = np.frombuffer(base64.b64decode(rec_data), np.int)
        detected.update({'best_n_inds': best_n_inds})

        # best_grip
        rec_data = self.sock.recv(64)
        length1 = rec_data.decode('utf-8')
        rec_data = self.recvall(self.sock, int(length1))
        best_grip = np.frombuffer(base64.b64decode(rec_data), np.float)
        detected.update({'best': best_grip})
        return detected

    def connectServer(self):
        try:
            self.sock = socket.socket()
            self.sock.connect((self.TCP_SERVER_IP, self.TCP_SERVER_PORT))
            print(
                u'Client socket is connected with Server socket [ TCP_SERVER_IP: ' + self.TCP_SERVER_IP + ', TCP_SERVER_PORT: ' + str(
                    self.TCP_SERVER_PORT) + ' ]')
            self.connectCount = 0

        except Exception as e:
            print(e)
            self.connectCount += 1
            if self.connectCount == 10:
                print(u'Connect fail %d times. exit program' % (self.connectCount))
                sys.exit()
            print(u'%d times try to connect with server' % (self.connectCount))
            self.connectServer()

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

if __name__=='__main__':
    import sys

    for arg in sys.argv:
        print(arg)
    rgb_data = cv2.imread("/mnt/workspace/data/0100/2022_09_22_13_25_34_rgb.png")
    depth_data = cv2.imread("/mnt/workspace/data/0100/2022_09_22_13_25_34_depth.png", -1)
    TCP_IP = 'localhost'
    TCP_PORT = 5000
    GraspPointDetector=GraspDetector(TCP_IP, TCP_PORT)
    count=0
    while 1:
        count+=1
        cur_rgb_data=copy.deepcopy(rgb_data)
        cur_depth_data=copy.deepcopy(depth_data)
        ret=GraspPointDetector.get_data(cur_rgb_data, cur_depth_data)
        time.sleep(5)
