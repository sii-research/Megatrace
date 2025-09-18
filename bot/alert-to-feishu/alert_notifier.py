#!/usr/bin/env python3
# coding: utf-8
"""
告警通知分发模块
负责将告警信息发送到各种通知渠道
"""

import json
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

# 日志配置
logger = logging.getLogger("alert_notifier")


class AlertNotifier:
    """告警通知器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
    
    def format_time(self, timestamp: str) -> str:
        """格式化时间戳，将UTC时间转换为CST时间"""
        try:
            if not timestamp:
                return "未知时间"
            
            # 解析UTC时间
            if timestamp.endswith('Z'):
                # 处理UTC时间格式 (2025-09-05T08:13:22.253Z)
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            elif '+' in timestamp or timestamp.count('-') > 2:
                # 处理带时区的时间格式
                dt = datetime.fromisoformat(timestamp)
            else:
                # 假设是UTC时间，没有时区标识
                dt = datetime.fromisoformat(timestamp + "+00:00")
            
            # 转换为CST时间 (UTC+8)
            cst_dt = dt.astimezone(datetime.now().astimezone().tzinfo)
            return cst_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logger.warning(f"时间格式转换失败: {timestamp}, 错误: {e}")
            return timestamp
    
    def _send_http_request(self, url: str, data: Dict[str, Any]) -> bool:
        """发送HTTP请求的通用方法"""
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, headers=headers, data=json.dumps(data))
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    return True
                else:
                    logger.error(f"请求失败: {result.get('msg')}")
            else:
                logger.error(f"请求失败: HTTP状态码 {response.status_code}")
            
            return False
        except Exception as e:
            logger.exception(f"发送HTTP请求异常: {e}")
            return False
    
    def send_notification(self, alert: Dict[str, Any], is_new: bool = True) -> bool:
        raise NotImplementedError("子类必须实现send_notification方法")


class FeishuNotifier(AlertNotifier):
    """飞书通知器"""
    
    def __init__(self, webhook_url: str):
        super().__init__("飞书")
        self.webhook_url = webhook_url
    
    
    def build_feishu_card(self, alert: Dict[str, Any], is_new: bool = True, repeat_count: int = 0) -> Dict[str, Any]:
        # 提取告警基本信息
        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        
        alertname = labels.get("alertname", "未知告警")
        alert_id = labels.get("id", "未知ID")
        alert_type = labels.get("alert_type", "未知类型")
        severity = labels.get("severity", "未知")
        node = labels.get("node", "未知节点")
        instance = labels.get("instance", "未知实例")
        job = labels.get("job", "未知任务")
        ip = labels.get("host_ip", "未知IP")
        fingerprint = alert.get("fingerprint", "未知指纹")
        
        # 提取更多详细字段
        cluster = labels.get("cluster", "未知集群")
        namespace = labels.get("exported_namespace", labels.get("namespace", "未知命名空间"))
        nodename = labels.get("nodename", "未知节点")
        pod_name = labels.get("pod_name", "未知Pod")
        pod_ip = labels.get("pod_ip", "未知Pod IP")
        pod_container_name = labels.get("pod_container_name", "未知容器")
        deployment = labels.get("deployment", "未知部署")
        
        # 正确处理AlertManager的状态字段结构
        status_obj = alert.get("status", {})
        if isinstance(status_obj, dict):
            status = status_obj.get("state", "未知")
        else:
            status = str(status_obj) if status_obj else "未知"
        
        summary = annotations.get("summary", "无描述")
        description = annotations.get("description", "无详细信息")
        
        # 设置卡片颜色
        if status == "resolved":
            color = "green"
        else:
            if severity.upper() == "P0":
                color = "red"
            elif severity.upper() == "P1":
                color = "orange"
            elif severity.upper() == "P2":
                color = "yellow"
            elif severity.upper() == "P3":
                color = "blue"
            else:
                color = "orange"
        
        # 提取开始时间
        start_time = self.format_time(alert.get("startsAt", ""))
        
        # 构建标题（支持重复推送次数）
        title_content = f"⚠【故障告警】 - {alertname}"
        if repeat_count > 0:
            title_content += f" -- 重复推送{repeat_count}次"
        
        # 构建卡片
        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": title_content if status in ["active", "firing"] else f"🟢【故障恢复】 - {alertname}"
                },
                "template": color
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"● **级别**: {severity}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"● **摘要**: {summary}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"● **描述**: {description}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"● **时间**: {start_time}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"● **节点**: {node}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"● **IP**: {ip}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"● **实例**: {instance}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"● **容器**: {pod_container_name}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"● **类型**: {alert_type}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"● **ID**: {alert_id}"
                    }
                },
                {
                    "tag": "hr"
                },

                # {
                #     "tag": "div",
                #     "text": {
                #         "tag": "lark_md",
                #         "content": f"● **指纹**: {fingerprint}"
                #     }
                # },
                # {
                #     "tag": "div",
                #     "text": {
                #         "tag": "lark_md",
                #         "content": f"● **命名空间**: {namespace}"
                #     }
                # },
                # {
                #     "tag": "div",
                #     "text": {
                #         "tag": "lark_md",
                #         "content": f"● **部署**: {deployment}"
                #     }
                # }
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": "ⓘ **请及时处理！**" if status in ["active", "firing"] else "🟢 **告警已处理**"
                    }
                }
            ]
        }
        
        return {"msg_type": "interactive", "card": card}
    
    def send_notification(self, alert: Dict[str, Any], is_new: bool = True) -> bool:
        try:
            card = self.build_feishu_card(alert, is_new)
            return self._send_http_request(self.webhook_url, card)
        except Exception as e:
            logger.exception(f"飞书通知发送异常: {e}")
            return False
    
    def send_notification_with_repeat_count(self, alert: Dict[str, Any], is_new: bool = True, repeat_count: int = 0) -> bool:
        """发送带重复推送次数的通知"""
        try:
            card = self.build_feishu_card(alert, is_new, repeat_count)
            return self._send_http_request(self.webhook_url, card)
        except Exception as e:
            logger.exception(f"飞书通知发送异常: {e}")
            return False


class FeishuTableNotifier(AlertNotifier):
    """飞书表格通知器"""
    
    def __init__(self, firing_webhook_url: str, resolv_webhook_url: str):
        self.firing_webhook_url = firing_webhook_url
        self.resolv_webhook_url = resolv_webhook_url
        self.enabled = True
    
    
    def build_table_data(self, alert: Dict[str, Any], is_new: bool = True) -> Dict[str, Any]:
        """构建飞书表格数据 - 发送所有字段的键值对格式"""
        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        
        # 获取状态
        status_obj = alert.get("status", {})
        if isinstance(status_obj, dict):
            status = status_obj.get("state", "未知")
        else:
            status = str(status_obj) if status_obj else "未知"
        
        # 构建类似图片中的键值对格式
        table_data = {
            "fingerprint": alert.get("fingerprint", "unknown"),
            "status": status,
            "starts_at": self.format_time(alert.get("startsAt", "")),
            "ends_at": self.format_time(alert.get("endsAt", "")),
            "id": labels.get("id", "unknown"),
            "alert_type": labels.get("alert_type", "unknown"),
            "alertname": labels.get("alertname", "unknown"),
            "instance": labels.get("instance", "unknown"),
            "severity": labels.get("severity", "unknown"),
            "job": labels.get("job", "unknown"),
            "node": labels.get("node", "unknown"),
            "host_ip": labels.get("host_ip", "unknown"),
            "cluster": labels.get("cluster", "unknown"),
            "namespace": labels.get("exported_namespace", labels.get("namespace", "unknown")),
            "nodename": labels.get("nodename", "unknown"),
            "pod_name": labels.get("pod_name", "unknown"),
            "pod_ip": labels.get("pod_ip", "unknown"),
            "pod_container_name": labels.get("pod_container_name", "unknown"),
            "deployment": labels.get("deployment", "unknown"),
            "summary": annotations.get("summary", "unknown"),
            "description": annotations.get("description", "unknown"),
            "recurrence_count": "0",  # 这个值需要从数据库获取，暂时设为0
            "labels": json.dumps(labels, ensure_ascii=False),
            "annotations": json.dumps(annotations, ensure_ascii=False)
        }
        
        return table_data
    
    def send_notification(self, alert: Dict[str, Any], is_new: bool = True) -> bool:
        """发送飞书表格通知"""
        try:
            # 根据告警状态选择webhook
            webhook_url = self.firing_webhook_url if is_new else self.resolv_webhook_url
            
            table_data = self.build_table_data(alert, is_new)
            return self._send_http_request(webhook_url, table_data)
        except Exception as e:
            logger.exception(f"飞书表格通知发送异常: {e}")
            return False


class PhoneNotifier(AlertNotifier):
    """电话通知器 - 只用于P0告警"""
    
    def __init__(self, app_id: str, app_secret: str, p0_group_webhook: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.p0_group_webhook = p0_group_webhook
        self.enabled = True
        self._access_token = None
        self._token_expires_at = 0
    
    def _get_access_token(self) -> str:
        """获取飞书访问令牌"""
        import time
        from datetime import datetime, timedelta
        
        # 如果令牌还有效，直接返回
        if self._access_token and datetime.now().timestamp() < self._token_expires_at:
            return self._access_token
        
        try:
            from config import FEISHU_TENANT_ACCESS_TOKEN_URL
            
            headers = {"Content-Type": "application/json"}
            data = {
                "app_id": self.app_id,
                "app_secret": self.app_secret
            }
            
            response = requests.post(FEISHU_TENANT_ACCESS_TOKEN_URL, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    self._access_token = result.get("tenant_access_token")
                    # 令牌有效期通常是2小时，这里设置为1.5小时以确保安全
                    self._token_expires_at = datetime.now().timestamp() + 5400
                    return self._access_token
                else:
                    logger.error(f"获取访问令牌失败: {result.get('msg')}")
            else:
                logger.error(f"获取访问令牌失败: HTTP状态码 {response.status_code}")
            
            return None
        except Exception as e:
            logger.exception(f"获取访问令牌异常: {e}")
            return None
    
    def _send_message_to_receiver_and_get_id(self, alert: Dict[str, Any]) -> str:
        """直接发送消息给接收人并获取message_id"""
        try:
            from config import FEISHU_MESSAGE_URL, P0_PHONE_RECEIVE_IDS
            
            access_token = self._get_access_token()
            if not access_token:
                return None
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}"
            }
            
            # 构建消息内容
            alertname = alert.get("labels", {}).get("alertname", "未知告警")
            severity = alert.get("labels", {}).get("severity", "未知级别")
            summary = alert.get("annotations", {}).get("summary", "")
            description = alert.get("annotations", {}).get("description", "")
            
            content = f"🚨 P0告警通知\n\n告警名称: {alertname}\n告警级别: {severity}\n告警摘要: {summary}\n告警描述: {description}"
            
            # 为每个接收人发送消息
            for receive_id in P0_PHONE_RECEIVE_IDS:
                data = {
                    "content": json.dumps({"text": content}),
                    "msg_type": "text",
                    "receive_id": receive_id
                }
                
                response = requests.post(
                    f"{FEISHU_MESSAGE_URL}?receive_id_type=open_id",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("code") == 0:
                        message_id = result.get("data", {}).get("message_id")
                        if message_id:
                            return message_id
                    else:
                        logger.error(f"发送消息给接收人失败: {result.get('msg')}")
                else:
                    logger.error(f"发送消息给接收人失败: HTTP状态码 {response.status_code}")
            
            return None
        except Exception as e:
            logger.exception(f"发送消息给接收人异常: {e}")
            return None
    
    def _send_urgent_phone(self, message_id: str) -> bool:
        """发送紧急电话通知"""
        try:
            from config import FEISHU_URGENT_PHONE_URL, P0_PHONE_RECEIVE_IDS
            
            access_token = self._get_access_token()
            if not access_token:
                return False
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}"
            }
            
            url = f"{FEISHU_URGENT_PHONE_URL.format(message_id=message_id)}?user_id_type=open_id"
            data = {
                "user_id_list": P0_PHONE_RECEIVE_IDS
            }
            
            # 添加调试日志
            logger.info(f"发送紧急电话请求URL: {url}")
            logger.info(f"发送紧急电话请求数据: {data}")
            
            response = requests.patch(url, headers=headers, json=data)
            
            # 添加响应调试
            logger.info(f"紧急电话响应状态码: {response.status_code}")
            logger.info(f"紧急电话响应内容: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    return True
                else:
                    logger.error(f"发送紧急电话失败: {result.get('msg')}")
            else:
                logger.error(f"发送紧急电话失败: HTTP状态码 {response.status_code}")
                logger.error(f"响应内容: {response.text}")
            
            return False
        except Exception as e:
            logger.exception(f"发送紧急电话异常: {e}")
            return False
    
    def send_notification(self, alert: Dict[str, Any], is_new: bool = True) -> bool:
        """发送电话通知 - 只用于P0告警"""
        try:
            # 只对新增的P0告警进行电话通知
            if not is_new:
                logger.info("恢复告警不需要电话通知")
                return True
            
            severity = alert.get("labels", {}).get("severity", "P3").upper()
            if severity != "P0":
                logger.info(f"非P0告警({severity})不需要电话通知")
                return True
            
            alertname = alert.get("labels", {}).get("alertname", "未知告警")
            
            # 发送消息给接收人并获取message_id
            message_id = self._send_message_to_receiver_and_get_id(alert)
            if not message_id:
                logger.error("获取message_id失败，无法发送电话通知")
                return False
            
            # 发送紧急电话通知
            success = self._send_urgent_phone(message_id)
            if success:
                logger.info(f"P0告警电话通知发送成功: {alertname}")
            else:
                logger.error(f"P0告警电话通知发送失败: {alertname}")
            
            return success
        except Exception as e:
            logger.exception(f"电话通知发送异常: {e}")
            return False


class P3WeeklyReportNotifier(AlertNotifier):
    """P3周报通知器"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.enabled = True
    
    def build_weekly_report_card(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """构建P3周报卡片"""
        total_count = summary_data.get("total_count", 0)
        active_count = summary_data.get("active_count", 0)
        resolved_count = summary_data.get("resolved_count", 0)
        recurrence_alert_count = summary_data.get("recurrence_alert_count", 0)
        total_recurrence = summary_data.get("total_recurrence", 0)
        period = summary_data.get("period", "")
        alert_details = summary_data.get("alert_details", [])
        
        # 构建告警详情文本
        alert_details_text = ""
        for alertname, count, total_recurrence in alert_details:
            alert_details_text += f"• {alertname}: {count}次"
            if total_recurrence > 0:
                alert_details_text += f" (复发{total_recurrence}次)"
            alert_details_text += "\n"
        
        if not alert_details_text:
            alert_details_text = "• 本周无P3告警"
        
        card = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": "📊 P3告警周报"
                    },
                    "template": "blue"
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**统计周期**: {period}"
                        }
                    },
                    {
                        "tag": "div",
                        "fields": [
                            {
                                "is_short": True,
                                "text": {
                                    "tag": "lark_md",
                                    "content": f"**总告警数**: {total_count}"
                                }
                            },
                            {
                                "is_short": True,
                                "text": {
                                    "tag": "lark_md",
                                    "content": f"**活跃告警**: {active_count}"
                                }
                            }
                        ]
                    },
                    {
                        "tag": "div",
                        "fields": [
                            {
                                "is_short": True,
                                "text": {
                                    "tag": "lark_md",
                                    "content": f"**已解决**: {resolved_count}"
                                }
                            },
                            {
                                "is_short": True,
                                "text": {
                                    "tag": "lark_md",
                                    "content": f"**复发告警**: {recurrence_alert_count}"
                                }
                            }
                        ]
                    },
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**总复发次数**: {total_recurrence}"
                        }
                    },
                    {
                        "tag": "hr"
                    },
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**告警详情**:\n{alert_details_text}"
                        }
                    }
                ]
            }
        }
        
        return card
    
    def send_weekly_report(self, summary_data: Dict[str, Any]) -> bool:
        """发送P3周报"""
        try:
            card = self.build_weekly_report_card(summary_data)
            return self._send_http_request(self.webhook_url, card)
        except Exception as e:
            logger.exception(f"P3周报发送异常: {e}")
            return False


class WeeklyReportNotifier(AlertNotifier):
    """告警周报通知器"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.enabled = True
    
    def build_weekly_report_card(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """构建告警周报卡片"""
        period = summary_data.get("period", "最近7天")
        total_stats = summary_data.get("total_stats", {})
        severity_stats = summary_data.get("severity_stats", {})
        top_alerts = summary_data.get("top_alerts", [])
        
        # 设置卡片颜色
        if total_stats.get("active_count", 0) > 0:
            color = "orange"
        else:
            color = "green"
        
        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"📊 告警周报 - {period}"
                },
                "template": color
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"📈 **总告警数**: {total_stats.get('total_count', 0)}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"🔴 **活跃告警**: {total_stats.get('active_count', 0)}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"✅ **已恢复**: {total_stats.get('resolved_count', 0)}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"🔄 **复发告警**: {total_stats.get('recurrence_count', 0)}"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"📢 **重复推送**: {total_stats.get('repeat_push_count', 0)} (共{total_stats.get('total_repeat_push', 0)}次)"
                    }
                }
            ]
        }
        
        # 添加各等级统计
        if severity_stats:
            severity_text = "**各等级统计:**\n"
            for severity in ['P0', 'P1', 'P2', 'P3']:
                if severity in severity_stats:
                    stats = severity_stats[severity]
                    severity_text += f"• {severity}: 总数{stats['total_count']} | 活跃{stats['active_count']} | 恢复{stats['resolved_count']} | 复发{stats['recurrence_count']} | 重复推送{stats['repeat_push_count']}\n"
            
            card["elements"].append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": severity_text
                }
            })
        
        # 添加TOP告警
        if top_alerts:
            top_text = "**TOP告警类型:**\n"
            for i, alert in enumerate(top_alerts[:5], 1):
                alertname = alert['alertname']
                count = alert['count']
                active_count = alert['active_count']
                top_text += f"{i}. {alertname}: {count}次 (活跃{active_count}次)\n"
            
            card["elements"].append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": top_text
                }
            })
        
        return {"msg_type": "interactive", "card": card}
    
    def send_weekly_report(self, summary_data: Dict[str, Any]) -> bool:
        """发送告警周报"""
        try:
            card = self.build_weekly_report_card(summary_data)
            return self._send_http_request(self.webhook_url, card)
        except Exception as e:
            logger.exception(f"告警周报发送异常: {e}")
            return False


class NotificationManager:
    """通知管理器"""
    
    def __init__(self):
        self.notifiers = {}
    
    def add_notifier(self, name: str, notifier: AlertNotifier) -> None:
        self.notifiers[name] = notifier
        logger.info(f"添加通知器: {name}")
    
    def send_notification(self, alert: Dict[str, Any], is_new: bool = True) -> Dict[str, bool]:
        results = {}
        for name, notifier in self.notifiers.items():
            if notifier.enabled:
                results[name] = notifier.send_notification(alert, is_new)
        return results
    
    def send_notification_by_severity(self, alert: Dict[str, Any], is_new: bool = True) -> Dict[str, bool]:
        """根据告警等级发送通知到对应的webhook"""
        results = {}
        
        # 获取告警等级
        severity = alert.get("labels", {}).get("severity", "P2").upper()
        
        # 根据等级选择对应的通知器（P3不发送到群组）
        if severity == "P0" and "feishu_p0" in self.notifiers:
            results["feishu_p0"] = self.notifiers["feishu_p0"].send_notification(alert, is_new)
        elif severity == "P1" and "feishu_p1" in self.notifiers:
            results["feishu_p1"] = self.notifiers["feishu_p1"].send_notification(alert, is_new)
        elif severity == "P2" and "feishu_p2" in self.notifiers:
            results["feishu_p2"] = self.notifiers["feishu_p2"].send_notification(alert, is_new)
        elif severity == "P3" and "feishu_p3" in self.notifiers:
            # P3等级根据配置决定是否发送到群组
            results["feishu_p3"] = self.notifiers["feishu_p3"].send_notification(alert, is_new)
        else:
            # 默认使用P2通知器
            if "feishu_p2" in self.notifiers:
                results["feishu_p2"] = self.notifiers["feishu_p2"].send_notification(alert, is_new)
        
        # 同时发送到飞书表格（所有等级都发送）
        if "feishu_table" in self.notifiers:
            results["feishu_table"] = self.notifiers["feishu_table"].send_notification(alert, is_new)
        
        # 电话通知（只对新增的P0告警）
        if is_new and severity == "P0" and "phone" in self.notifiers:
            results["phone"] = self.notifiers["phone"].send_notification(alert, is_new)
        
        return results
    
    def send_group_notification_only(self, alert: Dict[str, Any], repeat_count: int = 0) -> Dict[str, bool]:
        """只发送群组通知，不发送表格通知（用于重复推送）"""
        results = {}
        
        # 获取告警等级
        severity = alert.get("labels", {}).get("severity", "P2").upper()
        
        # 根据等级选择对应的通知器（P3不发送到群组）
        if severity == "P0" and "feishu_p0" in self.notifiers:
            results["feishu_p0"] = self.notifiers["feishu_p0"].send_notification_with_repeat_count(alert, True, repeat_count)
        elif severity == "P1" and "feishu_p1" in self.notifiers:
            results["feishu_p1"] = self.notifiers["feishu_p1"].send_notification_with_repeat_count(alert, True, repeat_count)
        elif severity == "P2" and "feishu_p2" in self.notifiers:
            results["feishu_p2"] = self.notifiers["feishu_p2"].send_notification_with_repeat_count(alert, True, repeat_count)
        elif severity == "P3" and "feishu_p3" in self.notifiers:
            # P3等级根据配置决定是否发送到群组
            results["feishu_p3"] = self.notifiers["feishu_p3"].send_notification_with_repeat_count(alert, True, repeat_count)
        else:
            # 默认使用P2通知器
            if "feishu_p2" in self.notifiers:
                results["feishu_p2"] = self.notifiers["feishu_p2"].send_notification_with_repeat_count(alert, True, repeat_count)
        
        # 不发送到飞书表格
        # 不发送电话通知
        
        return results
    
    
# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = NotificationManager()
    from config import FEISHU_WEBHOOK_URL
    feishu_notifier = FeishuNotifier(FEISHU_WEBHOOK_URL)
    manager.add_notifier("feishu", feishu_notifier)
    
    test_alert = {
        "fingerprint": "test_fingerprint",
        "status": {"state": "firing"},
        "labels": {
            "alertname": "测试告警",
            "severity": "critical"
        },
        "annotations": {
            "summary": "测试告警摘要",
            "description": "测试告警描述"
        },
        "startsAt": "2023-08-14T10:00:00Z"
    }
    
    results = manager.send_notification(test_alert, is_new=True)
    for channel, success in results.items():
        print(f"{channel}: {'成功' if success else '失败'}")
