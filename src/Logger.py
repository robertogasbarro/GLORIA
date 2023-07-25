import datetime


class Logger:
    def __init__(self):
        self.log = []

    def add(self, record):
        self.log.append((datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), record))

    def add_list(self, l):
        for item in l:
            self.log.append(item)

    def save(self, path):
        f = open(path, mode="w")
        s = ""
        for item in self.log:
            try:
                s += str(item[0])+"\n" + str(item[1]) + "\n\n"
            except:
                s += str(item) + "\n\n"
        f.write(s)
        f.close()
