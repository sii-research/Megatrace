#!/usr/bin/env python3
# coding: utf-8
"""
告警系统配置文件
所有配置参数、常量定义
"""

import os

# ======================================= AlertManager配置 =====================================================
# AlertManager API 地址
ALERTMANAGER_URL = ""
ALERTMANAGER_TIMEOUT = 10
ALERTMANAGER_MAX_RETRIES = 3
ALERTMANAGER_POLL_INTERVAL = 60

# ======================================= 飞书群组通知配置 =====================================================
# 飞书群组通知启用配置
ENABLE_FEISHU_GROUP_NOTIFICATION = False  # 是否启用飞书群组通知，禁用时会关闭所有群组通知
# 各等级群组通知启用配置
ENABLE_P0_GROUP_NOTIFICATION = True  # 是否启用P0群组通知
ENABLE_P1_GROUP_NOTIFICATION = True  # 是否启用P1群组通知
ENABLE_P2_GROUP_NOTIFICATION = True  # 是否启用P2群组通知
ENABLE_P3_GROUP_NOTIFICATION = False  # 是否启用P3群组通知（默认不启用）
# 飞书告警群自定义机器人webhook，需要配置自定义机器人在飞书群组中
P0_FEISHU_WEBHOOK_URL = ""
P1_FEISHU_WEBHOOK_URL = ""
P2_FEISHU_WEBHOOK_URL = ""
P3_FEISHU_WEBHOOK_URL = ""


# ======================================= 周报发送配置 =====================================================    
WEEKLY_REPORT_ENABLED = True  # 是否启用周报

WEEKLY_REPORT_DAY = 4  # 周报发送日期（0=周一，6=周日，5=周六）
WEEKLY_REPORT_HOUR = 18  # 周报发送小时（24小时制）
WEEKLY_REPORT_MINUTE = 0  # 周报发送分钟
#周报发送的群组webhook，需要配置自定义机器人在飞书群组中
WEEK_REPORT_FEISHU_WEBHOOK_URL = ""

# P3周报配置
ENABLE_P3_WEEKLY_REPORT = True  # 是否启用P3周报
P3_WEEKLY_REPORT_DAY = 4  # P3周报发送日期（0=周一，6=周日，5=周六）
P3_WEEKLY_REPORT_HOUR = 18  # P3周报发送小时（24小时制）
P3_WEEKLY_REPORT_MINUTE = 0  # P3周报发送分钟
P3_WEEKLY_REPORT_WEBHOOK_URL = ""


# ======================================= 飞书表格Webhook配置 =====================================================
# 飞书表格通知启用配置
ENABLE_FEISHU_TABLE_NOTIFICATION = True  # 是否启用飞书表格通知

# 飞书表格告警新增webhook，须在飞书表格自动化流程中设置，并配置webhook
FIRING_FEISHU_TABLE_WEBHOOK_URL = ""
# 飞书表格告警恢复webhook，须在飞书表格自动化流程中设置，并配置webhook
RESOLV_FEISHU_TABLE_WEBHOOK_URL = ""
# 飞书通知超时时间
FEISHU_TIMEOUT = 10



# ==========================================飞书电话通知配置 =====================================================
# 是否启用电话通知
ENABLE_PHONE_NOTIFICATION = True

# 飞书自建应用配置(基流电话告警机器人，一般情况勿动)
FEISHU_APP_ID = " "
FEISHU_APP_SECRET = ""
FEISHU_TENANT_ACCESS_TOKEN_URL = ""
FEISHU_MESSAGE_URL = " "
FEISHU_URGENT_PHONE_URL = " "
# 电话通知接收人配置 (open_id) - 只有P0需要电话通知
P0_PHONE_RECEIVE_IDS = [

        ]  





# ======================================= 重复推送群组通知配置 =====================================================
# 复发告警间隔控制（秒）
RECURRENCE_INTERVAL_SECONDS = 1200  # 20分钟，复发告警间隔小于此时间则忽略  #消除抖动，防止重复记录

# 重复推送配置
ENABLE_REPEAT_NOTIFICATION = True  # 是否允许重复推送
REPEAT_NOTIFICATION_INTERVAL_SECONDS = 86400  # 重复推送间隔（秒），24小时



# ======================================= 数据库配置 =====================================================
# 数据库文件路径
DATABASE_PATH = "./data/alerts.db"
# 数据库备份路径
DATABASE_BACKUP_PATH = "./data/backup/"
# 数据库备份保留天数
DATABASE_BACKUP_DAYS = 7

# ======================================= 日志配置 =====================================================
# 日志文件路径
LOG_FILE = "./logs/alert_system.log"
# 日志级别 (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = "INFO"
# 日志文件最大大小 (MB)
LOG_MAX_SIZE = 100
# 日志文件保留数量
LOG_BACKUP_COUNT = 5



# ======================================= 告警过滤配置 =====================================================
# 最小告警级别 (critical, warning, info)
MIN_SEVERITY = "warning"
# 排除的告警名称列表
EXCLUDE_ALERTNAMES = [
    # "TestAlert",
    # "DebugAlert"
]
# 包含的告警名称列表 (为空表示包含所有)
INCLUDE_ALERTNAMES = [
    # "HighCPULoad",
    # "DiskSpaceLow"
]
# 排除的标签
EXCLUDE_LABELS = {
    # "environment": "test",
    # "team": "dev"
}
# 包含的标签
INCLUDE_LABELS = {
    # "environment": "production",
    # "team": "ops"
}

# ======================================= 系统配置 =====================================================
# 是否启用守护进程模式
DAEMON_MODE = False
# PID文件路径
PID_FILE = "./alert_system.pid"
# 最大内存使用量 (MB)
MAX_MEMORY_MB = 1024
# 最大CPU使用率 (%)
MAX_CPU_PERCENT = 90

# ======================================= 环境变量覆盖 =====================================================
# 支持通过环境变量覆盖配置
def load_from_env():
    """从环境变量加载配置"""
    global ALERTMANAGER_URL, DATABASE_PATH, LOG_FILE, LOG_LEVEL
    
    # AlertManager配置
    if os.getenv("ALERTMANAGER_URL"):
        ALERTMANAGER_URL = os.getenv("ALERTMANAGER_URL")
    
    if os.getenv("ALERTMANAGER_TIMEOUT"):
        ALERTMANAGER_TIMEOUT = int(os.getenv("ALERTMANAGER_TIMEOUT"))
    
    # 数据库配置
    if os.getenv("DATABASE_PATH"):
        DATABASE_PATH = os.getenv("DATABASE_PATH")
    
    # 日志配置
    if os.getenv("LOG_FILE"):
        LOG_FILE = os.getenv("LOG_FILE")
    
    if os.getenv("LOG_LEVEL"):
        LOG_LEVEL = os.getenv("LOG_LEVEL")

# 加载环境变量配置
load_from_env()

# ======================================= 配置验证 =====================================================
def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 验证AlertManager URL
    if not ALERTMANAGER_URL.startswith(("http://", "https://")):
        errors.append("ALERTMANAGER_URL 必须是有效的HTTP/HTTPS URL")
    
    # 验证飞书Webhook URLs（只验证启用的通知器）
    if ENABLE_FEISHU_GROUP_NOTIFICATION:
        webhook_configs = [
            (ENABLE_P0_GROUP_NOTIFICATION, P0_FEISHU_WEBHOOK_URL, "P0"),
            (ENABLE_P1_GROUP_NOTIFICATION, P1_FEISHU_WEBHOOK_URL, "P1"),
            (ENABLE_P2_GROUP_NOTIFICATION, P2_FEISHU_WEBHOOK_URL, "P2"),
            (ENABLE_P3_GROUP_NOTIFICATION, P3_FEISHU_WEBHOOK_URL, "P3")
        ]
        
        for enabled, url, level in webhook_configs:
            if enabled and not url.startswith("https://"):
                errors.append(f"{level}_FEISHU_WEBHOOK_URL 必须是有效的HTTPS URL")
    
    # 验证飞书表格Webhook URLs
    if ENABLE_FEISHU_TABLE_NOTIFICATION:
        if not FIRING_FEISHU_TABLE_WEBHOOK_URL.startswith("https://"):
            errors.append("FIRING_FEISHU_TABLE_WEBHOOK_URL 必须是有效的HTTPS URL")
        if not RESOLV_FEISHU_TABLE_WEBHOOK_URL.startswith("https://"):
            errors.append("RESOLV_FEISHU_TABLE_WEBHOOK_URL 必须是有效的HTTPS URL")
    
    # 验证数据库路径
    if not DATABASE_PATH:
        errors.append("DATABASE_PATH 不能为空")
    
    # 验证日志级别
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    if LOG_LEVEL.upper() not in valid_log_levels:
        errors.append(f"LOG_LEVEL 必须是以下值之一: {', '.join(valid_log_levels)}")
    
    # 验证告警级别
    valid_severities = ["critical", "warning", "info"]
    if MIN_SEVERITY.lower() not in valid_severities:
        errors.append(f"MIN_SEVERITY 必须是以下值之一: {', '.join(valid_severities)}")
    
    # 验证周报配置
    if WEEKLY_REPORT_ENABLED:
        if not WEEK_REPORT_FEISHU_WEBHOOK_URL:
            errors.append("启用周报时，必须配置 WEEK_REPORT_FEISHU_WEBHOOK_URL")
        if not (0 <= WEEKLY_REPORT_DAY <= 6):
            errors.append("WEEKLY_REPORT_DAY 必须在 0-6 之间")
        if not (0 <= WEEKLY_REPORT_HOUR <= 23):
            errors.append("WEEKLY_REPORT_HOUR 必须在 0-23 之间")
        if not (0 <= WEEKLY_REPORT_MINUTE <= 59):
            errors.append("WEEKLY_REPORT_MINUTE 必须在 0-59 之间")
    
    # 验证P3周报配置
    if ENABLE_P3_WEEKLY_REPORT:
        if not P3_WEEKLY_REPORT_WEBHOOK_URL.startswith("https://"):
            errors.append("P3_WEEKLY_REPORT_WEBHOOK_URL 必须是有效的HTTPS URL")
        if not (0 <= P3_WEEKLY_REPORT_DAY <= 6):
            errors.append("P3_WEEKLY_REPORT_DAY 必须在 0-6 之间")
        if not (0 <= P3_WEEKLY_REPORT_HOUR <= 23):
            errors.append("P3_WEEKLY_REPORT_HOUR 必须在 0-23 之间")
        if not (0 <= P3_WEEKLY_REPORT_MINUTE <= 59):
            errors.append("P3_WEEKLY_REPORT_MINUTE 必须在 0-59 之间")
    
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

# 配置验证
if __name__ == "__main__":
    print("告警系统配置:")
    print(f"AlertManager URL: {ALERTMANAGER_URL}")
    print(f"数据库路径: {DATABASE_PATH}")
    print(f"日志文件: {LOG_FILE}")
    print(f"日志级别: {LOG_LEVEL}")
    print(f"最小告警级别: {MIN_SEVERITY}")
    print()
    
    print("飞书群组通知配置:")
    print(f"群组通知总开关: {'启用' if ENABLE_FEISHU_GROUP_NOTIFICATION else '禁用'}")
    if ENABLE_FEISHU_GROUP_NOTIFICATION:
        print(f"P0群组通知: {'启用' if ENABLE_P0_GROUP_NOTIFICATION else '禁用'}")
        print(f"P1群组通知: {'启用' if ENABLE_P1_GROUP_NOTIFICATION else '禁用'}")
        print(f"P2群组通知: {'启用' if ENABLE_P2_GROUP_NOTIFICATION else '禁用'}")
        print(f"P3群组通知: {'启用' if ENABLE_P3_GROUP_NOTIFICATION else '禁用'}")
    print()
    
    print("飞书表格通知配置:")
    print(f"表格通知: {'启用' if ENABLE_FEISHU_TABLE_NOTIFICATION else '禁用'}")
    print()
    
    print("周报配置:")
    print(f"通用周报: {'启用' if WEEKLY_REPORT_ENABLED else '禁用'}")
    if WEEKLY_REPORT_ENABLED:
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        print(f"通用周报发送时间: {weekdays[WEEKLY_REPORT_DAY]} {WEEKLY_REPORT_HOUR:02d}:{WEEKLY_REPORT_MINUTE:02d}")
    print(f"P3周报: {'启用' if ENABLE_P3_WEEKLY_REPORT else '禁用'}")
    if ENABLE_P3_WEEKLY_REPORT:
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        print(f"P3周报发送时间: {weekdays[P3_WEEKLY_REPORT_DAY]} {P3_WEEKLY_REPORT_HOUR:02d}:{P3_WEEKLY_REPORT_MINUTE:02d}")
        print(f"P3周报webhook: {P3_WEEKLY_REPORT_WEBHOOK_URL}")
    print()
    
    if validate_config():
        print("配置验证通过")
    else:
        print("配置验证失败")
