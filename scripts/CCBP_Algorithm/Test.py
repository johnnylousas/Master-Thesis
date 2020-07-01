class Test:

    def __init__(self, test_id: str, status: str, launch_time: str, exe_time: str):
        self.test_id = test_id
        self.status = status
        self.launch_time = launch_time
        self.exe_time = exe_time

    def __str__(self):
        return "T%s: %s" % (self.test_id , self.status)

    def __repr__(self):
        return "T%s: %s" % (self.test_id, self.status)
