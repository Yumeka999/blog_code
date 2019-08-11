import logging
import logging.handlers

'''
author = "kalafinaian"
email= "kalafinaian@outlook.com"
create_time = 2019-08-11
'''

# 时间 - py文件:行数 - 日志级别(info,warning,error)  具体信息
S_LOG_FORMAT = "[%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s] %(message)s"

# 日志路径设置
S_LOG_URL = "./run_msg.log"
        
# 每天一个日志, 'midnight'表示半夜进行更新
logger_handler = logging.handlers.TimedRotatingFileHandler(S_LOG_URL, 'midnight', 1, 0, encoding="utf-8")

# 设置后缀为 年-月-日_时-分-秒.log
logger_handler.suffix = "%Y-%m-%d_%H-%M-%S.log"

# 给hanlder设置上述的日志格式
logger_handler.setFormatter(logging.Formatter(S_LOG_FORMAT))
             
# 得到一个Logger对象，单例模式             
run_logger = logging.getLogger()

# 日志级别设置为Info
run_logger.setLevel(logging.INFO)

# Logger对象加入上述设置好的hanlder
run_logger.addHandler(logger_handler)