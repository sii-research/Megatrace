import time
import threading
from flask import Flask, request
import requests
from datetime import datetime
from threading import Lock

app = Flask(__name__)

# 存储客户端信息 {client_id: {"last_time": float, "status": "online/offline"}}
clients = {}
clients_lock = Lock()

#手动飞书报警配置
FEISHU_WEBHOOK = " "
def send_feishu_alert(message):
    """发送飞书报警"""
    try:
        payload = {"msg_type": "text", "content": {"text": message}}
        requests.post(FEISHU_WEBHOOK, json=payload, timeout=5)
    except Exception as e:
        print(f"飞书报警发送失败: {e}")

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    """接收心跳"""
    data = request.json
    client_id = data.get("client_id")
    
    if not client_id:
        return "Client ID missing", 400

    current_time = time.time()
    alert_message = None

    with clients_lock:
        # 首次连接或断线恢复
        if client_id not in clients or clients[client_id]["status"] == "offline":
            alert_message = f"✅ 客户端 {client_id} 恢复连接！当前时间: {datetime.now()}"
        
        # 更新客户端状态
        clients[client_id] = {
            "last_time": current_time,
            "status": "online"
        }
        print(f"[服务端] 客户端 {client_id} 心跳 at {datetime.now()}")

    # 发送恢复通知（在锁外执行，避免阻塞）
    if alert_message:
        print(alert_message)
        send_feishu_alert(alert_message)

    return "OK"

def check_heartbeats():
    """检查客户端超时（20分钟）"""
    while True:
        time.sleep(120)  # 每30秒检查一次
        current_time = time.time()
        alert_messages = []

        with clients_lock:
            for client_id, info in list(clients.items()):
                last_time = info["last_time"]
                if current_time - last_time > 1200:  # 20分钟=1200秒
                    if info["status"] != "offline":
                        alert_msg = f"⚠️ 客户端 {client_id} 失联！最后活跃: {datetime.fromtimestamp(last_time)}"
                        alert_messages.append(alert_msg)
                        clients[client_id]["status"] = "offline"  # 标记为离线

        for msg in alert_messages:
            print(msg)
            send_feishu_alert(msg)

if __name__ == '__main__':
    threading.Thread(target=check_heartbeats, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
