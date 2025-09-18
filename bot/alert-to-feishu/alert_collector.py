#!/usr/bin/env python3
# coding: utf-8
"""
告警收集和筛选模块
负责从AlertManager获取告警数据，并进行筛选、识别新告警、复发告警和恢复告警
"""

import json
import time
import logging
import requests
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from alert_database import AlertDatabase

# 日志配置
logger = logging.getLogger("alert_collector")


class AlertCollector:
    """告警收集器类
    负责从AlertManager API获取告警数据，并进行筛选和分类
    """
    
    def __init__(self, alertmanager_url: str, db_path: str, timeout: int = 10, max_retries: int = 3):
        """初始化告警收集器
        
        Args:
            alertmanager_url: AlertManager的URL
            db_path: 数据库文件路径
            timeout: HTTP请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.alertmanager_url = alertmanager_url.rstrip('/')
        self.db_path = db_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_url = f"{self.alertmanager_url}/api/v2/alerts"
        
        logger.info(f"告警收集器初始化完成，目标URL: {self.alertmanager_url}")
    
    def fetch_alerts(self) -> Optional[List[Dict[str, Any]]]:
        """获取AlertManager告警数据
        
        Returns:
            告警数据列表或None（如果请求失败）
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"正在请求AlertManager API: {self.api_url} (尝试 {attempt+1}/{self.max_retries})")
                response = requests.get(self.api_url, timeout=self.timeout)
                
                if response.status_code == 200:
                    alerts = response.json()
                    logger.info(f"成功获取到 {len(alerts)} 条告警信息")
                    return alerts
                else:
                    logger.error(f"获取Alertmanager告警失败，HTTP状态码: {response.status_code}")
                    if attempt < self.max_retries - 1:
                        logger.info(f"将在 2 秒后重试...")
                        time.sleep(2)
                        continue
                    return None
            except requests.exceptions.RequestException as e:
                logger.error(f"请求Alertmanager失败: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"将在 2 秒后重试...")
                    time.sleep(2)
                    continue
                return None
            except Exception as e:
                logger.error(f"处理Alertmanager响应时发生错误: {e}")
                return None
        
        return None
    
    def filter_alerts(self, alerts: List[Dict[str, Any]], 
                     alertname: Optional[str] = None,
                     status: Optional[str] = None,
                     labels: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """过滤告警数据
        
        Args:
            alerts: 原始告警列表
            alertname: 按告警名称过滤
            status: 按状态过滤 (active, resolved, all)
            labels: 按标签过滤，格式为 {key: value}
            
        Returns:
            过滤后的告警列表
        """
        filtered_alerts = []
        
        for alert in alerts:
            # 按告警名称过滤
            if alertname and alert.get("labels", {}).get("alertname") != alertname:
                continue
            
            # 按状态过滤
            if status and status != "all":
                alert_status = alert.get("status", {}).get("state", "")
                if status == "active" and alert_status not in ["active", "firing"]:
                    continue
                if status == "resolved" and alert_status != "resolved":
                    continue
            
            # 按标签过滤
            if labels:
                match = True
                for key, value in labels.items():
                    alert_value = alert.get("labels", {}).get(key)
                    if alert_value is None or value not in str(alert_value):
                        match = False
                        break
                
                if not match:
                    continue
            
            filtered_alerts.append(alert)
        
        logger.info(f"过滤后剩余 {len(filtered_alerts)} 条告警")
        return filtered_alerts
    
    def _extract_alert_status(self, alert: Dict[str, Any]) -> str:
        """提取告警状态
        
        Args:
            alert: 告警数据
            
        Returns:
            告警状态字符串
        """
        status_obj = alert.get("status", {})
        if isinstance(status_obj, dict):
            return status_obj.get("state", "")
        else:
            return str(status_obj) if status_obj else ""
    
    def _parse_utc_time(self, utc_timestamp: str) -> datetime:
        """解析UTC时间戳
        
        Args:
            utc_timestamp: UTC时间戳
            
        Returns:
            datetime对象
        """
        try:
            if not utc_timestamp:
                return datetime.now()
            
            # 解析UTC时间
            if utc_timestamp.endswith('Z'):
                # 处理UTC时间格式 (2025-09-05T08:13:22.253Z)
                return datetime.fromisoformat(utc_timestamp.replace("Z", "+00:00"))
            elif '+' in utc_timestamp or utc_timestamp.count('-') > 2:
                # 处理带时区的时间格式
                return datetime.fromisoformat(utc_timestamp)
            else:
                # 假设是UTC时间，没有时区标识
                return datetime.fromisoformat(utc_timestamp + "+00:00")
        except Exception as e:
            logger.warning(f"UTC时间解析失败: {utc_timestamp}, 错误: {e}")
            return datetime.now()
    
    def validate_alert(self, alert: Dict[str, Any]) -> bool:
        """验证告警数据的完整性
        
        Args:
            alert: 告警数据
            
        Returns:
            是否有效
        """
        # 检查必需字段
        if not alert.get("fingerprint"):
            logger.warning("告警缺少fingerprint字段")
            return False
        
        if not alert.get("labels", {}).get("alertname"):
            logger.warning(f"告警 {alert.get('fingerprint')} 缺少alertname标签")
            return False
        
        return True
    
    def process_alerts(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理告警数据，验证和清理
        
        Args:
            alerts: 原始告警列表
            
        Returns:
            处理后的告警列表
        """
        processed_alerts = []
        
        for alert in alerts:
            if self.validate_alert(alert):
                # 添加处理时间戳
                alert["_processed_at"] = datetime.now().isoformat()
                processed_alerts.append(alert)
            else:
                logger.warning(f"跳过无效告警: {alert.get('fingerprint', 'unknown')}")
        
        logger.info(f"处理完成，有效告警 {len(processed_alerts)} 条")
        return processed_alerts
    
    def analyze_alerts(self, current_alerts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Set[str], Set[str], Set[str]]:
        """分析告警，识别新告警、恢复告警和重复推送告警
        
        Args:
            current_alerts: 当前获取的告警列表
            
        Returns:
            需要通知的告警列表、新告警的fingerprint集合、恢复告警的fingerprint集合、重复推送告警的fingerprint集合
        """
        alerts_to_notify = []
        new_alert_fingerprints = set()
        resolved_alert_fingerprints = set()
        
        # 获取当前告警的fingerprint集合
        current_fingerprints = {alert.get("fingerprint", "") for alert in current_alerts}
        
        # 获取数据库中的所有告警
        try:
            with AlertDatabase(self.db_path) as db:
                db_alerts = db.get_all_alerts()
                db_fingerprints = {alert.get("fingerprint", "") for alert in db_alerts}
                
                # 识别新告警、复发告警和重复推送告警
                new_fingerprints = set()
                recurrence_fingerprints = set()
                repeat_notification_fingerprints = set()
                
                # 获取当前轮询时间
                current_time = datetime.now()
                
                # 获取复发间隔配置
                from config import RECURRENCE_INTERVAL_SECONDS
                
                for fingerprint in current_fingerprints:
                    if fingerprint not in db_fingerprints:
                        # 数据库中不存在的指纹，是新告警
                        new_fingerprints.add(fingerprint)
                    else:
                        # 数据库中存在的指纹，检查startsAt是否不同
                        existing_alert = db.get_alert_by_fingerprint(fingerprint)
                        if existing_alert:
                            # 检查现有告警的状态
                            existing_state = self._extract_alert_status(existing_alert)
                            
                            # 如果现有告警状态是active，说明这是持续告警，不是复发
                            if existing_state in ["active", "firing"]:
                                logger.debug(f"忽略持续告警: {fingerprint}")
                                continue
                            
                            # 只有现有告警状态是resolved时，才可能是复发告警
                            if existing_state == "resolved":
                                existing_starts_at = existing_alert.get("startsAt", "")
                                current_alert = next((a for a in current_alerts if a.get("fingerprint") == fingerprint), None)
                                if current_alert:
                                    current_starts_at = current_alert.get("startsAt", "")
                                    if existing_starts_at != current_starts_at:
                                        # startsAt不同，检查是否为复发告警
                                        # 获取最新恢复时间
                                        latest_resolved_time = db.get_latest_resolved_time(fingerprint)
                                        
                                        if latest_resolved_time:
                                            try:
                                                # 解析最新恢复时间
                                                resolved_time = self._parse_utc_time(latest_resolved_time)
                                                # 计算时间差（秒）
                                                time_diff = (current_time - resolved_time).total_seconds()
                                                
                                                if time_diff >= RECURRENCE_INTERVAL_SECONDS:
                                                    # 间隔足够长，允许复发告警
                                                    recurrence_fingerprints.add(fingerprint)
                                                    logger.info(f"发现复发告警: {fingerprint}, 距离上次恢复: {time_diff:.1f}秒")
                                                else:
                                                    # 间隔太短，忽略复发告警
                                                    logger.info(f"忽略短期复发告警: {fingerprint}, 距离上次恢复: {time_diff:.1f}秒 < {RECURRENCE_INTERVAL_SECONDS}秒")
                                            except Exception as e:
                                                logger.error(f"解析恢复时间失败: {fingerprint}, {e}")
                                                # 解析失败时，允许复发告警
                                                recurrence_fingerprints.add(fingerprint)
                                        else:
                                            # 没有恢复记录，但现有告警状态是resolved，允许复发告警
                                            recurrence_fingerprints.add(fingerprint)
                                            logger.info(f"发现复发告警（现有告警已恢复）: {fingerprint}")
                            else:
                                # 现有告警状态不是resolved也不是active，可能是其他状态，忽略
                                logger.debug(f"忽略状态异常的告警: {fingerprint}, 状态: {existing_state}")
                
                # 识别恢复告警（数据库中状态不为resolved的告警，但当前不存在的告警）
                resolved_fingerprints = set()
                # 获取数据库中状态不为resolved的告警指纹
                db_active_fingerprints = db.get_active_fingerprints()
                # B组多出来的就是恢复告警
                for fingerprint in db_active_fingerprints - current_fingerprints:
                    resolved_fingerprints.add(fingerprint)
                
                logger.info(f"当前告警: {len(current_fingerprints)} 条")
                logger.info(f"数据库告警: {len(db_fingerprints)} 条")
                logger.info(f"新告警: {len(new_fingerprints)} 条")
                logger.info(f"复发告警: {len(recurrence_fingerprints)} 条")
                logger.info(f"恢复告警: {len(resolved_fingerprints)} 条")
                
                # 筛选需要通知的告警
                for alert in current_alerts:
                    fingerprint = alert.get("fingerprint", "")
                    
                    # 正确处理AlertManager的状态字段结构
                    status = self._extract_alert_status(alert)
                    
                    # 只处理active或firing状态的告警
                    if status not in ["active", "firing"]:
                        continue
                    
                    # 新告警需要通知
                    if fingerprint in new_fingerprints:
                        alerts_to_notify.append(alert)
                        new_alert_fingerprints.add(fingerprint)
                        logger.info(f"发现新告警: {fingerprint}")
                    # 复发告警需要通知（当成新告警处理）
                    elif fingerprint in recurrence_fingerprints:
                        alerts_to_notify.append(alert)
                        new_alert_fingerprints.add(fingerprint)  # 复发告警也加入新告警集合
                        logger.info(f"发现复发告警: {fingerprint}")
                    else:
                        logger.debug(f"忽略持续告警: {fingerprint}")
                
                # 记录恢复的告警并更新数据库状态
                for fingerprint in resolved_fingerprints:
                    resolved_alert_fingerprints.add(fingerprint)
                    logger.info(f"告警已恢复: {fingerprint}")
                    # 将数据库中的告警状态更新为resolved，不清空推送状态
                    db.update_alert_status(fingerprint, "resolved")
                
                # 检查active告警是否需要重复推送
                repeat_notification_fingerprints = set()
                for fingerprint in current_fingerprints:
                    if fingerprint in db_fingerprints:
                        # 检查是否为active状态
                        last_status = db.get_alert_last_status(fingerprint)
                        if last_status == "active":
                            # 检查是否需要重复推送
                            if db.should_repeat_group_notification(fingerprint):
                                repeat_notification_fingerprints.add(fingerprint)
                                logger.info(f"需要重复推送群组通知: {fingerprint}")
                
                logger.info(f"筛选出 {len(alerts_to_notify)} 条需要通知的告警，其中 {len(new_fingerprints)} 条为新告警，{len(recurrence_fingerprints)} 条为复发告警，{len(resolved_alert_fingerprints)} 条已恢复，{len(repeat_notification_fingerprints)} 条需要重复推送")
                return alerts_to_notify, new_alert_fingerprints, resolved_alert_fingerprints, repeat_notification_fingerprints
        except Exception as e:
            logger.error(f"分析告警失败: {e}")
            # 如果出错，返回所有active或firing状态的告警，并假设都是新告警
            alerts_to_notify = []
            for alert in current_alerts:
                status = self._extract_alert_status(alert)
                if status in ["active", "firing"]:
                    alerts_to_notify.append(alert)
            new_alert_fingerprints = {alert.get("fingerprint", "") for alert in alerts_to_notify}
            return alerts_to_notify, new_alert_fingerprints, set(), set()


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 测试告警收集器
    from config import ALERTMANAGER_URL, ALERTMANAGER_TIMEOUT, ALERTMANAGER_MAX_RETRIES, DATABASE_PATH
    
    collector = AlertCollector(ALERTMANAGER_URL, DATABASE_PATH, ALERTMANAGER_TIMEOUT, ALERTMANAGER_MAX_RETRIES)
    
    # 获取告警
    alerts = collector.fetch_alerts()
    if alerts:
        print(f"获取到 {len(alerts)} 条告警")
        
        # 过滤active状态的告警
        active_alerts = collector.filter_alerts(alerts, status="active")
        print(f"Active状态告警: {len(active_alerts)} 条")
        
        # 处理告警
        processed_alerts = collector.process_alerts(active_alerts)
        print(f"处理后告警: {len(processed_alerts)} 条")
        
        # 分析告警
        alerts_to_notify, new_fingerprints, resolved_fingerprints, repeat_fingerprints = collector.analyze_alerts(processed_alerts)
        print(f"需要通知的告警: {len(alerts_to_notify)} 条")
        print(f"新告警: {len(new_fingerprints)} 条")
        print(f"恢复告警: {len(resolved_fingerprints)} 条")
        print(f"重复推送告警: {len(repeat_fingerprints)} 条")
        
        # 显示前3条告警的基本信息
        for i, alert in enumerate(processed_alerts[:3]):
            print(f"\n告警 {i+1}:")
            print(f"  Fingerprint: {alert.get('fingerprint')}")
            print(f"  Alertname: {alert.get('labels', {}).get('alertname')}")
            print(f"  Status: {alert.get('status', {}).get('state')}")
            print(f"  Severity: {alert.get('labels', {}).get('severity')}")
    else:
        print("获取告警失败")
