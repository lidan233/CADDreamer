import os
import time
from datetime import date

class LWriter(object):
    def __init__(self, rootpath):
        if os.path.exists(rootpath):
            self.use = False
        self.timeDay = date.fromtimestamp(time.time())
        self.data = str(self.timeDay.year) + '_' + str(self.timeDay.month) + '_' + str(self.timeDay.day)
        if not os.path.exists(os.path.join(rootpath,self.data)):
            os.mkdir(os.path.join(rootpath,self.data))
        self.savepath = os.path.join(rootpath,self.data)

    def write(self,name):
        f = open(os.path.join(os.path.join(self.savepath, name)))
        return f

    def writePath(self,name):
        return os.path.join(os.path.join(self.savepath, name))

