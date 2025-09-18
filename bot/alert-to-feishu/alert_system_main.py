#!/usr/bin/env python3
# coding: utf-8
"""
告警系统主程序
整合告警收集、数据库存储、通知分发三大模块
"""

import sys
import time
import signal
import logging
import argparse
from typing import List, Dict, Any, Set

# 导入模块
from config import (
    ALERTMANAGER_URL, ALERTMANAGER_TIMEOUT, ALERTMANAGER_MAX_RETRIES, ALERTMANAGER_POLL_INTERVAL,
    FEISHU_TIMEOUT,
    DATABASE_PATH, DATABASE_BACKUP_PATH, DATABASE_BACKUP_DAYS,
    LOG_FILE, LOG_LEVEL, LOG_MAX_SIZE, LOG_BACKUP_COUNT,
    MIN_SEVERITY, EXCLUDE_ALERTNAMES, INCLUDE_ALERTNAMES, EXCLUDE_LABELS, INCLUDE_LABELS,
    DAEMON_MODE, PID_FILE, MAX_MEMORY_MB, MAX_CPU_PERCENT,
    validate_config
)
from alert_collector import AlertCollector
from alert_database import AlertDatabase
from alert_notifier import NotificationManager, FeishuNotifier, FeishuTableNotifier, P3WeeklyReportNotifier, WeeklyReportNotifier, PhoneNotifier

# 日志配置
logger = logging.getLogger("alert_system")


class AlertSystem:
    """告警系统主类"""
    
    def __init__(self):
        """初始化告警系统"""
        # 验证配置
        if not validate_config():
            raise ValueError("配置验证失败")
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self._init_components()
        
        # 运行标志
        self.running = False
        
        # 周报发送时间记录
        self.last_weekly_report_time = None
        self.last_p3_weekly_report_time = None
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("告警系统初始化完成")
    
    def _setup_logging(self) -> None:
        """设置日志系统"""
        log_level = getattr(logging, LOG_LEVEL.upper())
        log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        
        # 确保日志目录存在
        import os
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        
        # 配置根日志记录器
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_FILE, encoding="utf-8")
            ]
        )
    
    def _init_components(self) -> None:
        """初始化系统组件"""
        # 初始化告警收集器
        self.collector = AlertCollector(
            ALERTMANAGER_URL,
            DATABASE_PATH,
            ALERTMANAGER_TIMEOUT,
            ALERTMANAGER_MAX_RETRIES
        )
        
        # 初始化数据库
        self.database = AlertDatabase(DATABASE_PATH)
        
        # 初始化通知管理器
        self.notification_manager = NotificationManager()
        
        # 添加不同等级的飞书通知器
        from config import (
            P0_FEISHU_WEBHOOK_URL, P1_FEISHU_WEBHOOK_URL, P2_FEISHU_WEBHOOK_URL, P3_FEISHU_WEBHOOK_URL,
            ENABLE_FEISHU_GROUP_NOTIFICATION, ENABLE_P0_GROUP_NOTIFICATION, ENABLE_P1_GROUP_NOTIFICATION,
            ENABLE_P2_GROUP_NOTIFICATION, ENABLE_P3_GROUP_NOTIFICATION
        )
        
        # 根据配置添加群组通知器
        if ENABLE_FEISHU_GROUP_NOTIFICATION:
            # P0紧急告警通知器
            if ENABLE_P0_GROUP_NOTIFICATION:
                feishu_p0_notifier = FeishuNotifier(P0_FEISHU_WEBHOOK_URL)
                self.notification_manager.add_notifier("feishu_p0", feishu_p0_notifier)
                logger.info("已启用P0群组通知")
            
            # P1重要告警通知器
            if ENABLE_P1_GROUP_NOTIFICATION:
                feishu_p1_notifier = FeishuNotifier(P1_FEISHU_WEBHOOK_URL)
                self.notification_manager.add_notifier("feishu_p1", feishu_p1_notifier)
                logger.info("已启用P1群组通知")
            
            # P2一般告警通知器
            if ENABLE_P2_GROUP_NOTIFICATION:
                feishu_p2_notifier = FeishuNotifier(P2_FEISHU_WEBHOOK_URL)
                self.notification_manager.add_notifier("feishu_p2", feishu_p2_notifier)
                logger.info("已启用P2群组通知")
            
            # P3提示告警通知器
            if ENABLE_P3_GROUP_NOTIFICATION:
                feishu_p3_notifier = FeishuNotifier(P3_FEISHU_WEBHOOK_URL)
                self.notification_manager.add_notifier("feishu_p3", feishu_p3_notifier)
                logger.info("已启用P3群组通知")
        else:
            logger.info("飞书群组通知已禁用")
        
        # 添加飞书表格通知器
        from config import (
            FIRING_FEISHU_TABLE_WEBHOOK_URL, RESOLV_FEISHU_TABLE_WEBHOOK_URL,
            ENABLE_FEISHU_TABLE_NOTIFICATION
        )
        
        if ENABLE_FEISHU_TABLE_NOTIFICATION:
            feishu_table_notifier = FeishuTableNotifier(FIRING_FEISHU_TABLE_WEBHOOK_URL, RESOLV_FEISHU_TABLE_WEBHOOK_URL)
            self.notification_manager.add_notifier("feishu_table", feishu_table_notifier)
            logger.info("已启用飞书表格通知")
        else:
            logger.info("飞书表格通知已禁用")
        
        # 添加P3周报通知器（使用独立的webhook）
        from config import P3_WEEKLY_REPORT_WEBHOOK_URL, ENABLE_P3_WEEKLY_REPORT
        if ENABLE_P3_WEEKLY_REPORT:
            p3_weekly_notifier = P3WeeklyReportNotifier(P3_WEEKLY_REPORT_WEBHOOK_URL)
            self.notification_manager.add_notifier("p3_weekly", p3_weekly_notifier)
            logger.info("P3周报通知器已启用")
        else:
            logger.info("P3周报通知器已禁用")
        
        # 添加告警周报通知器
        from config import WEEK_REPORT_FEISHU_WEBHOOK_URL
        weekly_report_notifier = WeeklyReportNotifier(WEEK_REPORT_FEISHU_WEBHOOK_URL)
        self.notification_manager.add_notifier("weekly_report", weekly_report_notifier)
        
        # 添加电话通知器（只用于P0告警）
        from config import FEISHU_APP_ID, FEISHU_APP_SECRET, P0_FEISHU_WEBHOOK_URL
        phone_notifier = PhoneNotifier(FEISHU_APP_ID, FEISHU_APP_SECRET, P0_FEISHU_WEBHOOK_URL)
        self.notification_manager.add_notifier("phone", phone_notifier)
        
        # 其他通知器可以在这里添加...
        
        logger.info("系统组件初始化完成")
    
    def _process_notification_results(self, results: Dict[str, bool], fingerprint: str, notification_type: str) -> int:
        """处理通知发送结果
        
        Args:
            results: 通知发送结果字典
            fingerprint: 告警指纹
            notification_type: 通知类型（新告警、复发告警、恢复告警等）
            
        Returns:
            成功发送的通知数量
        """
        success_count = 0
        for channel, success in results.items():
            if success:
                success_count += 1
                logger.info(f"[{channel}] 成功发送{notification_type}通知: {fingerprint}")
            else:
                logger.error(f"[{channel}] 发送{notification_type}通知失败: {fingerprint}")
        return success_count
    
    def _signal_handler(self, signum, frame) -> None:
        """信号处理函数"""
        logger.info(f"收到信号 {signum}，正在停止系统...")
        self.running = False
    
    def _filter_alerts(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤告警"""
        filtered_alerts = []
        
        for alert in alerts:
            # 按告警名称过滤
            alertname = alert.get("labels", {}).get("alertname", "")
            
            if EXCLUDE_ALERTNAMES and alertname in EXCLUDE_ALERTNAMES:
                continue
            
            if INCLUDE_ALERTNAMES and alertname not in INCLUDE_ALERTNAMES:
                continue
            
            # 按严重程度过滤
            severity = alert.get("labels", {}).get("severity", "").upper()
            min_severity = MIN_SEVERITY.upper()
            
            # 支持多种严重程度格式
            severity_levels = {
                "P0": 4, "CRITICAL": 4,
                "P1": 3, "HIGH": 3,
                "P2": 2, "WARNING": 2, "MEDIUM": 2,
                "P3": 1, "INFO": 1, "LOW": 1
            }
            
            # 如果严重程度不在已知列表中，默认为最低级别
            current_level = severity_levels.get(severity, 0)
            min_level = severity_levels.get(min_severity, 0)
            
            if current_level < min_level:
                logger.debug(f"告警 {alert.get('labels', {}).get('alertname', '')} 严重程度 {severity} 低于最小级别 {min_severity}，跳过")
                continue
            
            # 按标签过滤
            labels = alert.get("labels", {})
            
            # 排除标签
            for key, value in EXCLUDE_LABELS.items():
                if labels.get(key) == value:
                    continue
            
            # 包含标签
            if INCLUDE_LABELS:
                include_match = False
                for key, value in INCLUDE_LABELS.items():
                    if labels.get(key) == value:
                        include_match = True
                        break
                if not include_match:
                    continue
            
            filtered_alerts.append(alert)
        
        logger.info(f"过滤后剩余 {len(filtered_alerts)} 条告警")
        return filtered_alerts
    
    def process_alerts_once(self) -> bool:
        """处理一次告警"""
        try:
            # 1. 收集告警
            logger.info("开始收集告警...")
            alerts = self.collector.fetch_alerts()
            if alerts is None:
                logger.error("获取告警失败")
                return False
            
            if not alerts:
                logger.info("没有获取到告警")
                return True
            
            # 2. 过滤告警
            logger.info("开始过滤告警...")
            filtered_alerts = self._filter_alerts(alerts)
            
            if not filtered_alerts:
                logger.info("过滤后没有告警")
                return True
            
            # 3. 筛选需要通知的告警
            logger.info("开始筛选需要通知的告警...")
            alerts_to_notify, new_fingerprints, resolved_fingerprints, repeat_notification_fingerprints = self.collector.analyze_alerts(filtered_alerts)
            
            # 4. 存储到数据库
            logger.info("开始存储告警到数据库...")
            
            # 分别存储新告警和复发告警
            new_alerts = []
            recurrence_alerts = []
            
            for alert in filtered_alerts:
                fingerprint = alert.get("fingerprint", "")
                if fingerprint in new_fingerprints:
                    new_alerts.append(alert)
                # 注意：复发告警现在直接作为新告警处理，不需要单独存储
            
            # 存储新告警
            if new_alerts:
                store_result = self.database.store_alerts(new_alerts)
                logger.info(f"新告警存储结果: 成功 {store_result['success']} 条, 失败 {store_result['fail']} 条")
            
            # 复发告警现在直接作为新告警处理，不需要单独存储
            
            # 5. 发送新告警通知
            if alerts_to_notify:
                logger.info(f"开始发送 {len(alerts_to_notify)} 条告警通知...")
                for alert in alerts_to_notify:
                    fingerprint = alert.get("fingerprint", "")
                    is_new = fingerprint in new_fingerprints
                    
                    results = self.notification_manager.send_notification_by_severity(alert, is_new)
                    notification_type = '新' if is_new else '复发'
                    success_count = self._process_notification_results(results, fingerprint, f"{notification_type}告警")
                    
                    # 如果有任何通知器发送成功，更新推送状态和时间
                    if success_count > 0:
                        self.database.update_push_status(fingerprint, "reported")
                        # 更新群组推送时间
                        self.database.update_group_push_time(fingerprint)
                        # 更新表格推送时间
                        self.database.update_table_push_time(fingerprint)
                        logger.info(f"更新告警 {fingerprint} 推送状态和时间")
            else:
                logger.info("没有需要通知的告警")
            
            # 6. 发送恢复告警通知
            if resolved_fingerprints:
                logger.info(f"开始发送 {len(resolved_fingerprints)} 条恢复告警通知...")
                for fingerprint in resolved_fingerprints:
                    # 从数据库获取恢复的告警信息
                    with AlertDatabase(DATABASE_PATH) as db:
                        # 检查是否已经发送过恢复通知
                        # 获取当前活跃告警记录的startsAt，检查是否已经发送过该实例的恢复通知
                        active_alert = db.get_active_alert_by_fingerprint(fingerprint)
                        if active_alert:
                            starts_at = active_alert.get("startsAt", "")
                            if db.has_sent_recovery_notification(fingerprint, starts_at):
                                logger.info(f"告警 {fingerprint} (startsAt: {starts_at}) 已发送过恢复通知，跳过")
                                continue
                            
                        resolved_alert = db.get_alert_by_fingerprint(fingerprint)
                        if resolved_alert:
                            # 修改状态为resolved
                            resolved_alert["status"] = {"state": "resolved"}
                            results = self.notification_manager.send_notification_by_severity(resolved_alert, False)
                            success_count = self._process_notification_results(results, fingerprint, "恢复告警")
                            
                            # 如果有任何通知器发送成功，更新推送状态
                            if success_count > 0:
                                # 更新当前活跃告警记录的推送状态
                                if active_alert:
                                    starts_at = active_alert.get("startsAt", "")
                                    db.update_push_status_by_starts_at(fingerprint, starts_at, "recovery_reported")
                                    logger.info(f"更新恢复告警 {fingerprint} (startsAt: {starts_at}) 推送状态为 recovery_reported")
                        else:
                            logger.warning(f"未找到恢复告警的详细信息: {fingerprint}")
            
            # 7. 发送重复推送通知（只推送到群组，不推送到表格）
            if repeat_notification_fingerprints:
                logger.info(f"开始发送 {len(repeat_notification_fingerprints)} 条重复推送通知...")
                for fingerprint in repeat_notification_fingerprints:
                    # 从数据库获取告警信息
                    with AlertDatabase(DATABASE_PATH) as db:
                        alert = db.get_alert_by_fingerprint(fingerprint)
                        if alert:
                            # 获取当前重复推送次数
                            current_repeat_count = db.get_repeat_push_count(fingerprint)
                            new_repeat_count = current_repeat_count + 1
                            
                            # 只发送群组通知，不发送表格通知
                            results = self.notification_manager.send_group_notification_only(alert, new_repeat_count)
                            success_count = self._process_notification_results(results, fingerprint, f"重复推送(第{new_repeat_count}次)")
                            
                            # 如果群组通知发送成功，更新群组推送时间和重复推送次数
                            if success_count > 0:
                                db.update_group_push_time(fingerprint)
                                db.increment_repeat_push_count(fingerprint)
                                logger.info(f"更新告警 {fingerprint} 群组推送时间和重复推送次数 (第{new_repeat_count}次)")
                        else:
                            logger.warning(f"未找到告警的详细信息: {fingerprint}")
            
            return True
            
        except Exception as e:
            logger.exception(f"处理告警时发生错误: {e}")
            return False
    
    def run(self, daemon_mode: bool = False) -> None:
        """运行告警系统
        
        Args:
            daemon_mode: 是否以守护进程模式运行
        """
        self.running = True
        poll_interval = ALERTMANAGER_POLL_INTERVAL
        
        logger.info(f"告警系统启动，轮询间隔: {poll_interval} 秒")
        
        if daemon_mode:
            logger.info("以守护进程模式运行")
        
        try:
            while self.running:
                start_time = time.time()
                
                # 处理告警
                success = self.process_alerts_once()
                if not success:
                    logger.warning("本次处理失败，将在下次轮询时重试")
                
                # 检查是否需要发送通用周报
                if self.should_send_weekly_report():
                    logger.info("检测到通用周报发送时间，开始发送通用周报...")
                    weekly_success = self.send_weekly_report()
                    if weekly_success:
                        logger.info("通用周报发送成功")
                    else:
                        logger.error("通用周报发送失败")
                
                # 检查是否需要发送P3周报
                if self.should_send_p3_weekly_report():
                    logger.info("检测到P3周报发送时间，开始发送P3周报...")
                    p3_weekly_success = self.send_p3_weekly_report()
                    if p3_weekly_success:
                        logger.info("P3周报发送成功")
                    else:
                        logger.error("P3周报发送失败")
                
                # 计算下次轮询时间
                elapsed = time.time() - start_time
                sleep_time = max(0, poll_interval - elapsed)
                
                if self.running:
                    logger.info(f"将在 {sleep_time:.1f} 秒后再次轮询...")
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止...")
        except Exception as e:
            logger.exception(f"系统运行时发生错误: {e}")
        finally:
            self.stop()
    
    def send_p3_weekly_report(self) -> bool:
        """发送P3周报"""
        try:
            from config import ENABLE_P3_WEEKLY_REPORT
            if not ENABLE_P3_WEEKLY_REPORT:
                logger.info("P3周报功能已禁用，跳过发送")
                return False
            
            logger.info("开始生成P3周报...")
            
            # 获取P3告警周报数据
            summary_data = self.database.get_p3_alerts_weekly_summary()
            
            if not summary_data:
                logger.warning("没有P3告警数据，跳过周报发送")
                return False
            
            # 发送P3周报
            p3_weekly_notifier = self.notification_manager.notifiers.get("p3_weekly")
            if p3_weekly_notifier:
                success = p3_weekly_notifier.send_weekly_report(summary_data)
                if success:
                    logger.info("P3周报发送成功")
                else:
                    logger.error("P3周报发送失败")
                return success
            else:
                logger.error("P3周报通知器未找到")
                return False
                
        except Exception as e:
            logger.exception(f"发送P3周报时发生错误: {e}")
            return False
    
    def send_weekly_report(self) -> bool:
        """发送告警周报"""
        try:
            logger.info("开始生成告警周报...")
            
            # 获取告警周报数据
            summary_data = self.database.get_weekly_alert_summary()
            
            if not summary_data:
                logger.warning("没有告警数据，跳过周报发送")
                return False
            
            # 发送告警周报
            weekly_report_notifier = self.notification_manager.notifiers.get("weekly_report")
            if weekly_report_notifier:
                success = weekly_report_notifier.send_weekly_report(summary_data)
                if success:
                    logger.info("告警周报发送成功")
                else:
                    logger.error("告警周报发送失败")
                return success
            else:
                logger.error("告警周报通知器未找到")
                return False
                
        except Exception as e:
            logger.exception(f"发送告警周报时发生错误: {e}")
            return False
    
    def should_send_weekly_report(self) -> bool:
        """检查是否应该发送通用周报"""
        try:
            from config import WEEKLY_REPORT_ENABLED, WEEKLY_REPORT_DAY, WEEKLY_REPORT_HOUR, WEEKLY_REPORT_MINUTE
            from datetime import datetime
            
            if not WEEKLY_REPORT_ENABLED:
                return False
            
            now = datetime.now()
            
            # 检查是否是配置的周报发送时间
            if (now.weekday() == WEEKLY_REPORT_DAY and 
                now.hour == WEEKLY_REPORT_HOUR and 
                now.minute == WEEKLY_REPORT_MINUTE):
                
                # 检查是否已经发送过（避免重复发送）
                if (self.last_weekly_report_time is None or 
                    (now - self.last_weekly_report_time).total_seconds() > 3600):  # 1小时内不重复发送
                    
                    self.last_weekly_report_time = now
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"检查通用周报发送时间失败: {e}")
            return False
    
    def should_send_p3_weekly_report(self) -> bool:
        """检查是否应该发送P3周报"""
        try:
            from config import ENABLE_P3_WEEKLY_REPORT, P3_WEEKLY_REPORT_DAY, P3_WEEKLY_REPORT_HOUR, P3_WEEKLY_REPORT_MINUTE
            from datetime import datetime
            
            if not ENABLE_P3_WEEKLY_REPORT:
                return False
            
            now = datetime.now()
            
            # 检查是否是配置的P3周报发送时间
            if (now.weekday() == P3_WEEKLY_REPORT_DAY and 
                now.hour == P3_WEEKLY_REPORT_HOUR and 
                now.minute == P3_WEEKLY_REPORT_MINUTE):
                
                # 检查是否已经发送过（避免重复发送）
                if (self.last_p3_weekly_report_time is None or 
                    (now - self.last_p3_weekly_report_time).total_seconds() > 3600):  # 1小时内不重复发送
                    
                    self.last_p3_weekly_report_time = now
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"检查P3周报发送时间失败: {e}")
            return False
    
    def stop(self) -> None:
        """停止告警系统"""
        logger.info("正在停止告警系统...")
        self.running = False
        
        # 关闭数据库连接
        if hasattr(self, 'database'):
            self.database.close()
        
        logger.info("告警系统已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="告警系统")
    parser.add_argument("-d", "--daemon", action="store_true", help="以守护进程模式运行")
    parser.add_argument("--once", action="store_true", help="只运行一次，不进行轮询")
    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--p3-weekly", action="store_true", help="发送P3周报")
    parser.add_argument("--weekly-report", action="store_true", help="发送告警周报")
    
    args = parser.parse_args()
    
    try:
        # 创建告警系统实例
        alert_system = AlertSystem()
        
        if args.p3_weekly:
            # 发送P3周报
            logger.info("发送P3周报...")
            success = alert_system.send_p3_weekly_report()
            sys.exit(0 if success else 1)
        elif args.weekly_report:
            # 发送告警周报
            logger.info("发送告警周报...")
            success = alert_system.send_weekly_report()
            sys.exit(0 if success else 1)
        elif args.test:
            # 测试模式
            logger.info("运行测试模式...")
            success = alert_system.process_alerts_once()
            sys.exit(0 if success else 1)
        elif args.once:
            # 只运行一次
            logger.info("运行一次...")
            success = alert_system.process_alerts_once()
            sys.exit(0 if success else 1)
        else:
            # 正常运行
            alert_system.run(args.daemon)
    
    except Exception as e:
        logger.exception(f"程序启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
