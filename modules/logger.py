from datetime import datetime

class Logger:
    def __init__(self, args, g_id, is_server=False):
        self.args = args
        self.g_id = g_id
        self.is_server = is_server
        
    def switch(self, c_id):
        self.c_id = c_id

    def print(self, message):
        now = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        msg = f'[{now}]'
        msg += f'[{self.args.model}]'
        msg += f'[g:{self.g_id}]'
        msg += f'[server]'if self.is_server else f'[c:{self.c_id}]'
        msg += f' {message}'
        print(msg)
