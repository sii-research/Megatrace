import time
import requests
import subprocess
import socket
import json

# 需要手动添加地址
SERVER_URL = " "
CLIENT_ID = 'jingneng'  # 使用主机名作为ID（或自定义）
PROCESS_NAME = 'node'   # 要监控的进程名
CHECK_INTERVAL = 5      # 检查间隔（秒）

def is_process_running():
    """检查目标进程是否在运行"""
    try:
        # 使用pgrep更可靠地检查进程
        result = subprocess.run(['pgrep', '-f', PROCESS_NAME], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
        return result.returncode == 0
    except Exception as e:
        print(f"进程检查异常: {e}")
        return False

def send_heartbeat():
    """发送心跳报文"""
    try:
        response = requests.post(
            SERVER_URL,
            json={
                "client_id": CLIENT_ID,
                "status": "alive",
                "process_running": is_process_running()
            },
            timeout=3
        )
        print(f"[{time.ctime()}] 心跳成功 | 进程状态: {is_process_running()} | 响应: {response.text}")
    except Exception as e:
        print(f"[{time.ctime()}] 心跳失败: {e}")

def main():
    """主循环"""
    while True:
        if is_process_running():
            send_heartbeat()
        else:
            print(f"[{time.ctime()}] 警告: 进程 {PROCESS_NAME} 未运行")
        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    print(f"启动监控 (PID: {os.getpid()})")
    print(f"监控进程: {PROCESS_NAME}")
    print(f"心跳服务器: {SERVER_URL}")
    main()
