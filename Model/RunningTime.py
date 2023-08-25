import time

from Helper.Timer import FormatTimeSStohhmmss


class MyRunningTime:
    def __init__(self):
        self.runningTime = 0
        self.startTime = time.time()

    def CalculateExecutedTime(self):
        endTime = time.time()
        self.runningTime = endTime - self.startTime
        self.runningTime = FormatTimeSStohhmmss(self.runningTime)
        print(f'Total execution time: {self.runningTime}')
