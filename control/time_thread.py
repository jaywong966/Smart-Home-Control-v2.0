import threading
import time

exitFlag = 0

class myThread (threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.status = True

    def run(self) -> None:
        print ("start thread：" + self.name)
        self.print_time(self.name, self.counter, 1)
        self.status = False
        print("end")


    def thread_status(self):
        return self.status

    def print_time(self, threadName, delay, counter):
        while counter:
            if exitFlag:
                threadName.exit()
            time.sleep(delay)
            counter -= 1

# 创建新线程
#thread1 = myThread(1, "Thread-1", 3)

# 开启新线程
#thread1.start()
#thread1.join()
#print ("退出主线程")