import logging
import os
import sys
from pathlib import Path

import pexpect  # type: ignore[reportMissingModuleSource]

_LOG_NAME = "jiuwenclaw_chat_log"

def _get_module_logger() -> logging.Logger:
    """独立 logger：不碰 root，不向父级传播，避免与宿主程序的 logging.basicConfig 互相影响。"""
    log = logging.getLogger(_LOG_NAME)
    log.setLevel(logging.INFO)
    if not log.handlers:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jiuwenclaw_chat.log")
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        log.addHandler(fh)
        log.propagate = False
    return log

m_logger = _get_module_logger()

# def send_enter(child):
#     os.write(child.fileno(), b'\r\n')

class Tee:
    """同时输出到多个文件对象"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()  # 立即刷新，避免缓冲
    
    def flush(self):
        for f in self.files:
            f.flush()

def post_process_interaction_log(log_file_path: str | os.PathLike[str]) -> str:
    with open(log_file_path, 'r', encoding='utf-8') as file:
        log_content = file.read()
    remove_pattern = [
        "[38;2;255;255;255m",
        "[38;2;136;136;136m─",
        # ']8;;[?2026l[2A[2G[?25l[2G[?25l[2G[?25l[2G[?25l[3B',
        "[?2026l[4B[2G[?25l[?2026h[4A",
        "[?2026l[4B[2G[?25l[?2026h[2B",
        '[?25h[?2004l[>4;0m',
        "[?2026l[2A[2G[?25l",
        '[39m',
        '[0m',
        "]8;;",
        "[2K",
    ]
    for pattern in remove_pattern:
        log_content = log_content.replace(pattern, '')
    line_pattern_to_remove = ["esc to interrupt", "JIUWEN CLAW", " | mode:", " | Mode:"] # remove all lines that contain the pattern
    lines = log_content.split('\n')
    new_lines = []
    n_lines_to_remove = 0
    for line in lines:
        if any(pattern in line for pattern in line_pattern_to_remove):
            n_lines_to_remove += 1
            continue
        else:
            new_lines.append(line)
    log_content = '\n'.join(new_lines)
    print(f"Removed {n_lines_to_remove} lines")
    for _ in range(3):
        log_content = log_content.replace("\n\n", '\n')
    with open(log_file_path, 'w', encoding='utf-8') as file:
        file.write(log_content)
    return log_content

def interact_with_jiuwenclaw(
    input_string,
    *,
    interaction_log_file: str | os.PathLike[str] = "jiuwenclaw_interaction.log",
    cwd: str | os.PathLike[str] | None = None,
    timeout: float = 600,
    tee: bool = True,
    command: str = "jiuwenclaw-tui",
):
    # 1. 启动工具
    # child = pexpect.spawn('jiuwenclaw-tui', timeout = 600)
    # child.logfile = sys.stdout.buffer
    cwd_path = Path(cwd).expanduser().resolve() if cwd is not None else None
    child = pexpect.spawn(
        command,
        cwd=str(cwd_path) if cwd_path is not None else None,
        encoding='utf-8',
        timeout=timeout,
    )
    child.linesep = '\r\n'
    # child.logfile = sys.stdout # （可选）把工具的输出实时打印到屏幕，方便调试
    log_path = Path(interaction_log_file).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, 'w', encoding='utf-8')
    child.logfile = Tee(sys.stdout, log_file) if tee else log_file
    
    try:
        m_logger.info(f"\n==================================================\n请求内容：{input_string}\n==================================================")
        m_logger.info("等待工具启动完成")
        child.expect('https://gitcode.com/openJiuwen/agent-core', timeout=30)  # 替换成你工具实际的提示文字
        
        m_logger.info("等待2秒后按回车（确认workspace）")
        child.expect(pexpect.TIMEOUT, timeout=2)
        child.sendline('')  # 发送回车
        
        m_logger.info("等待2秒后发送请求")
        child.expect(pexpect.TIMEOUT, timeout=2)
        child.sendline(input_string)  # 发送你的字符串 + 回车
        child.sendline('')  # 多发送一个回车
        
        MAX_WAIT_TIME = timeout
        WAIT_INTERVAL = 5
        NO_RESPONSE_TIME = 20
        for _ in range(int(MAX_WAIT_TIME/WAIT_INTERVAL)):
            child.expect(pexpect.TIMEOUT, timeout=WAIT_INTERVAL) # 使用expect timeout代替time.sleep以防界面卡顿
            idx = child.expect(['esc to interrupt', pexpect.TIMEOUT], timeout=NO_RESPONSE_TIME)
            if idx == 0:
                m_logger.info(f"运行中，等待{WAIT_INTERVAL}秒")
                while(child.expect(['esc to interrupt', pexpect.TIMEOUT], timeout=2) == 0):
                    pass # 消费掉所有缓冲区中的信号
            elif idx == 1:
                m_logger.info("循环等待超时退出，一定时间内无刷新")
                break
        
        idx = child.expect(['mode:code.normal', '────────────────────────────────────────', pexpect.TIMEOUT], timeout=20)
        m_logger.info(f"捕捉到结束信号[{idx}]")
        
        m_logger.info("发送退出命令")
        child.sendline("/exit")  # 发送你的字符串 + 回车
        idx = child.expect([pexpect.TIMEOUT, pexpect.EOF], timeout=10)
        if idx == 0:
            m_logger.info("超时退出")
        elif idx == 1:
            m_logger.info("EOF退出")
        return child.before
    
    except pexpect.TIMEOUT:
        m_logger.error("错误：等待超时，工具可能没有按预期响应")
        return None
    except pexpect.EOF:
        m_logger.error("工具意外退出")
        return child.before
    finally:
        child.close()
        log_file.close()
        post_process_interaction_log(log_path)


# 测试运行
if __name__ == "__main__":
    post_process_interaction_log("jiuwenclaw_interaction.log")
    # result = interact_with_jiuwenclaw("修改你的工作目录workspace为/mnt/d/QCLAW_SPACE，你仅可以在最新的工作目录中创建修改文件，然后在线搜索今天NVDA的股价，保存到/mnt/d/QCLAW_SPACE/NVDA_price.txt")
    # m_logger.info("\n--- 最终输出 ---")
    # m_logger.info(result)