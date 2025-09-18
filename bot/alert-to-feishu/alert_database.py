#!/usr/bin/env python3
# coding: utf-8
"""
告警数据库模块
负责告警数据的存储、查询和管理
"""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import os

logger = logging.getLogger("alert_database")


class AlertDatabase:
    """告警数据库类"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
        # 确保数据库目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self) -> None:
        """初始化数据库表结构"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # 创建告警主表
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL,
                alertname TEXT NOT NULL,
                status TEXT NOT NULL,
                severity TEXT,
                summary TEXT,
                description TEXT,
                starts_at TEXT,
                ends_at TEXT,
                push_status TEXT DEFAULT 'pending',
                group_push_time TEXT,
                table_push_time TEXT,
                repeat_push_count INTEGER DEFAULT 0,
                resolved_at TEXT,
                recurrence_count INTEGER DEFAULT 0,
                is_recurrence BOOLEAN DEFAULT 0,
                updated_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                raw_data TEXT NOT NULL
            )
            """)
            
            # 创建告警状态历史表
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_status_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                changed_at TEXT NOT NULL,
                raw_data TEXT NOT NULL,
                FOREIGN KEY (fingerprint) REFERENCES alerts (fingerprint)
            )
            """)
            
            self.conn.commit()
            logger.info("数据库表结构初始化完成")
            
        except sqlite3.Error as e:
            logger.error(f"初始化数据库失败: {e}")
            raise
    
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
    
    def _convert_utc_to_cst(self, utc_timestamp: str) -> str:
        """将UTC时间转换为CST时间
        
        Args:
            utc_timestamp: UTC时间戳
            
        Returns:
            CST时间戳
        """
        try:
            if not utc_timestamp:
                return ""
            
            # 解析UTC时间
            if utc_timestamp.endswith('Z'):
                # 处理UTC时间格式 (2025-09-05T08:13:22.253Z)
                dt = datetime.fromisoformat(utc_timestamp.replace("Z", "+00:00"))
            elif '+' in utc_timestamp or utc_timestamp.count('-') > 2:
                # 处理带时区的时间格式
                dt = datetime.fromisoformat(utc_timestamp)
            else:
                # 假设是UTC时间，没有时区标识
                dt = datetime.fromisoformat(utc_timestamp + "+00:00")
            
            # 转换为CST时间 (UTC+8)
            cst_dt = dt.astimezone(datetime.now().astimezone().tzinfo)
            return cst_dt.isoformat()
        except Exception as e:
            logger.warning(f"UTC时间转换失败: {utc_timestamp}, 错误: {e}")
            return utc_timestamp
    
    def store_alert(self, alert: Dict[str, Any], is_recurrence: bool = False) -> bool:
        """存储单个告警数据
        
        Args:
            alert: 告警数据
            is_recurrence: 是否为复发告警
        """
        try:
            fingerprint = alert.get("fingerprint", "")
            if not fingerprint:
                return False
            
            # 提取告警信息
            alertname = alert.get("labels", {}).get("alertname", "")
            status = self._extract_alert_status(alert)
            
            severity = alert.get("labels", {}).get("severity", "")
            summary = alert.get("annotations", {}).get("summary", "")
            description = alert.get("annotations", {}).get("description", "")
            # 将UTC时间转换为CST时间
            starts_at = self._convert_utc_to_cst(alert.get("startsAt", ""))
            ends_at = self._convert_utc_to_cst(alert.get("endsAt", ""))
            updated_at = datetime.now().isoformat()
            created_at = datetime.now().isoformat()
            raw_data = json.dumps(alert, ensure_ascii=False)
            
            # 总是插入新记录（新告警或复发告警）
            self.cursor.execute("""
            INSERT INTO alerts (
                fingerprint, alertname, status, severity, summary, 
                description, starts_at, ends_at, is_recurrence, updated_at, created_at, raw_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (fingerprint, alertname, status, severity, summary, 
                  description, starts_at, ends_at, is_recurrence, updated_at, created_at, raw_data))
            
            # 记录状态到历史表
            self.cursor.execute("""
            INSERT INTO alert_status_history (fingerprint, status, changed_at, raw_data)
            VALUES (?, ?, ?, ?)
            """, (fingerprint, status, updated_at, raw_data))
            
            logger.info(f"存储告警: {fingerprint}, 是否复发: {is_recurrence}")
            
            self.conn.commit()
            return True
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"存储告警失败: {e}")
            return False
    
    def store_alerts(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """批量存储告警数据"""
        success_count = 0
        fail_count = 0
        
        for alert in alerts:
            if self.store_alert(alert):
                success_count += 1
            else:
                fail_count += 1
        
        return {"success": success_count, "fail": fail_count}
    
    def get_alert_last_status(self, fingerprint: str) -> str:
        """获取告警的最新状态"""
        try:
            # 优先从status字段获取状态，如果为空则从raw_data获取
            self.cursor.execute("SELECT status, raw_data FROM alerts WHERE fingerprint = ? ORDER BY created_at DESC LIMIT 1", (fingerprint,))
            row = self.cursor.fetchone()
            if row:
                # 如果status字段有值，直接返回
                if row[0]:
                    return row[0]
                # 如果status字段为空，从raw_data获取
                elif row[1]:
                    try:
                        alert_data = json.loads(row[1])
                        return self._extract_alert_status(alert_data)
                    except json.JSONDecodeError:
                        return ""
            return ""
        except sqlite3.Error as e:
            logger.error(f"获取告警状态失败: {e}")
            return ""
    
    def get_push_status(self, fingerprint: str) -> str:
        """获取告警的推送状态"""
        try:
            self.cursor.execute("SELECT push_status FROM alerts WHERE fingerprint = ? ORDER BY created_at DESC LIMIT 1", (fingerprint,))
            row = self.cursor.fetchone()
            return row[0] if row and row[0] else ""
        except sqlite3.Error as e:
            logger.error(f"获取推送状态失败: {e}")
            return ""
    
    def get_active_alert_by_fingerprint(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """获取活跃告警记录（状态为active的最新记录）"""
        try:
            self.cursor.execute("""
            SELECT raw_data FROM alerts 
            WHERE fingerprint = ? AND status = 'active' 
            ORDER BY created_at DESC LIMIT 1
            """, (fingerprint,))
            row = self.cursor.fetchone()
            if row and row[0]:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    logger.error(f"解析活跃告警数据失败: {fingerprint}")
                    return None
            return None
        except sqlite3.Error as e:
            logger.error(f"获取活跃告警记录失败: {e}")
            return None
    
    def has_sent_recovery_notification(self, fingerprint: str, starts_at: str) -> bool:
        """检查是否已经发送过指定告警实例的恢复通知"""
        try:
            # 检查是否存在相同fingerprint和startsAt且推送状态为recovery_reported的记录
            self.cursor.execute("""
            SELECT COUNT(*) FROM alerts 
            WHERE fingerprint = ? AND starts_at = ? AND push_status = 'recovery_reported'
            """, (fingerprint, starts_at))
            count = self.cursor.fetchone()[0]
            return count > 0
        except sqlite3.Error as e:
            logger.error(f"检查恢复通知发送状态失败: {e}")
            return False
    
    def get_active_fingerprints(self) -> Set[str]:
        """获取数据库中状态不为resolved的告警指纹集合"""
        try:
            self.cursor.execute("SELECT fingerprint FROM alerts WHERE status != 'resolved'")
            rows = self.cursor.fetchall()
            return {row[0] for row in rows if row[0]}
        except sqlite3.Error as e:
            logger.error(f"获取活跃告警指纹失败: {e}")
            return set()
    
    def get_latest_resolved_time(self, fingerprint: str) -> Optional[str]:
        """获取告警的最新恢复时间
        
        Args:
            fingerprint: 告警指纹
            
        Returns:
            最新恢复时间字符串，如果没有恢复记录则返回None
        """
        try:
            self.cursor.execute("""
            SELECT resolved_at FROM alerts 
            WHERE fingerprint = ? AND resolved_at IS NOT NULL 
            ORDER BY resolved_at DESC LIMIT 1
            """, (fingerprint,))
            
            row = self.cursor.fetchone()
            return row[0] if row else None
            
        except sqlite3.Error as e:
            logger.error(f"获取最新恢复时间失败: {e}")
            return None
    
    def get_alert_by_fingerprint(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """根据fingerprint获取最新的告警信息
        
        Args:
            fingerprint: 告警指纹
            
        Returns:
            告警信息字典，如果不存在则返回None
        """
        try:
            self.cursor.execute("SELECT raw_data FROM alerts WHERE fingerprint = ? ORDER BY created_at DESC LIMIT 1", (fingerprint,))
            row = self.cursor.fetchone()
            if row and row[0]:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    logger.error(f"解析告警数据失败: {fingerprint}")
                    return None
            return None
        except sqlite3.Error as e:
            logger.error(f"获取告警信息失败: {e}")
            return None
    
    def update_alert_status(self, fingerprint: str, status: str) -> bool:
        """更新告警状态 - 更新所有相同fingerprint的记录"""
        try:
            # 获取最新的告警数据用于更新raw_data
            self.cursor.execute("SELECT raw_data FROM alerts WHERE fingerprint = ? ORDER BY created_at DESC LIMIT 1", (fingerprint,))
            row = self.cursor.fetchone()
            if not row or not row[0]:
                return False
                
            try:
                alert_data = json.loads(row[0])
                if "status" not in alert_data:
                    alert_data["status"] = {}
                alert_data["status"]["state"] = status
                
                updated_at = datetime.now().isoformat()
                raw_data = json.dumps(alert_data, ensure_ascii=False)
                
                # 如果是恢复状态，记录恢复时间
                resolved_at = None
                if status == "resolved":
                    resolved_at = updated_at
                
                # 关键修改：更新所有相同fingerprint的记录，而不仅仅是最新的一条
                self.cursor.execute("""
                UPDATE alerts SET status = ?, updated_at = ?, raw_data = ?, resolved_at = ?
                WHERE fingerprint = ?
                """, (status, updated_at, raw_data, resolved_at, fingerprint))
                
                self.conn.commit()
                logger.info(f"更新告警状态: {fingerprint} -> {status}")
                return True
                
            except json.JSONDecodeError:
                logger.error(f"解析告警数据失败: {fingerprint}")
                return False
                
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"更新告警状态失败: {e}")
            return False
    
    def update_push_status(self, fingerprint: str, push_status: str = "reported") -> bool:
        """更新告警推送状态（兼容旧版本）"""
        try:
            self.cursor.execute("""
            UPDATE alerts SET push_status = ?
            WHERE fingerprint = ?
            """, (push_status, fingerprint))
            
            self.conn.commit()
            return True
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"更新推送状态失败: {e}")
            return False
    
    def update_push_status_by_starts_at(self, fingerprint: str, starts_at: str, push_status: str) -> bool:
        """根据fingerprint和startsAt更新推送状态"""
        try:
            self.cursor.execute("""
            UPDATE alerts SET push_status = ?
            WHERE fingerprint = ? AND starts_at = ?
            """, (push_status, fingerprint, starts_at))
            
            self.conn.commit()
            return True
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"根据startsAt更新推送状态失败: {e}")
            return False
    
    def update_group_push_time(self, fingerprint: str) -> bool:
        """更新群组推送时间
        
        Args:
            fingerprint: 告警指纹
            
        Returns:
            更新是否成功
        """
        try:
            push_time = datetime.now().isoformat()
            self.cursor.execute("""
            UPDATE alerts SET group_push_time = ?
            WHERE fingerprint = ?
            """, (push_time, fingerprint))
            
            self.conn.commit()
            logger.info(f"更新群组推送时间: {fingerprint}")
            return True
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"更新群组推送时间失败: {e}")
            return False
    
    def update_table_push_time(self, fingerprint: str) -> bool:
        """更新表格推送时间
        
        Args:
            fingerprint: 告警指纹
            
        Returns:
            更新是否成功
        """
        try:
            push_time = datetime.now().isoformat()
            self.cursor.execute("""
            UPDATE alerts SET table_push_time = ?
            WHERE fingerprint = ?
            """, (push_time, fingerprint))
            
            self.conn.commit()
            logger.info(f"更新表格推送时间: {fingerprint}")
            return True
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"更新表格推送时间失败: {e}")
            return False
    
    def get_group_push_time(self, fingerprint: str) -> Optional[str]:
        """获取群组推送时间
        
        Args:
            fingerprint: 告警指纹
            
        Returns:
            群组推送时间，如果没有则返回None
        """
        try:
            self.cursor.execute("""
            SELECT group_push_time FROM alerts 
            WHERE fingerprint = ? AND status = 'active'
            ORDER BY created_at DESC LIMIT 1
            """, (fingerprint,))
            
            row = self.cursor.fetchone()
            return row[0] if row else None
            
        except sqlite3.Error as e:
            logger.error(f"获取群组推送时间失败: {e}")
            return None
    
    def should_repeat_group_notification(self, fingerprint: str) -> bool:
        """判断是否应该重复推送群组通知
        
        Args:
            fingerprint: 告警指纹
            
        Returns:
            是否应该重复推送
        """
        from config import ENABLE_REPEAT_NOTIFICATION, REPEAT_NOTIFICATION_INTERVAL_SECONDS
        
        if not ENABLE_REPEAT_NOTIFICATION:
            return False
        
        group_push_time = self.get_group_push_time(fingerprint)
        if not group_push_time:
            return True  # 没有推送过，应该推送
        
        try:
            from datetime import datetime
            last_push_time = datetime.fromisoformat(group_push_time.replace('Z', '+00:00'))
            current_time = datetime.now()
            time_diff = (current_time - last_push_time).total_seconds()
            
            return time_diff >= REPEAT_NOTIFICATION_INTERVAL_SECONDS
            
        except Exception as e:
            logger.error(f"计算重复推送时间差失败: {e}")
            return True  # 解析失败时允许推送
    
    
    def increment_repeat_push_count(self, fingerprint: str) -> bool:
        """增加重复推送次数
        
        Args:
            fingerprint: 告警指纹
            
        Returns:
            更新是否成功
        """
        try:
            self.cursor.execute("""
            UPDATE alerts SET repeat_push_count = repeat_push_count + 1
            WHERE fingerprint = ?
            """, (fingerprint,))
            
            self.conn.commit()
            logger.info(f"增加重复推送次数: {fingerprint}")
            return True
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"增加重复推送次数失败: {e}")
            return False
    
    def get_repeat_push_count(self, fingerprint: str) -> int:
        """获取重复推送次数
        
        Args:
            fingerprint: 告警指纹
            
        Returns:
            重复推送次数
        """
        try:
            self.cursor.execute("""
            SELECT repeat_push_count FROM alerts 
            WHERE fingerprint = ? AND status = 'active'
            ORDER BY created_at DESC LIMIT 1
            """, (fingerprint,))
            
            row = self.cursor.fetchone()
            return row[0] if row else 0
            
        except sqlite3.Error as e:
            logger.error(f"获取重复推送次数失败: {e}")
            return 0
    
    def increment_recurrence_count(self, fingerprint: str) -> bool:
        """增加告警的复发次数"""
        try:
            self.cursor.execute("""
            UPDATE alerts SET recurrence_count = recurrence_count + 1
            WHERE fingerprint = ?
            """, (fingerprint,))
            
            self.conn.commit()
            return True
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"增加复发次数失败: {e}")
            return False
    
    def get_recurrence_count(self, fingerprint: str) -> int:
        """获取告警的复发次数"""
        try:
            self.cursor.execute("""
            SELECT recurrence_count FROM alerts WHERE fingerprint = ?
            """, (fingerprint,))
            
            result = self.cursor.fetchone()
            return result[0] if result else 0
            
        except sqlite3.Error as e:
            logger.error(f"获取复发次数失败: {e}")
            return 0
    
    def get_alert_status_history(self, fingerprint: str) -> List[Dict[str, Any]]:
        """获取告警的状态变化历史"""
        try:
            self.cursor.execute("""
            SELECT status, changed_at, raw_data 
            FROM alert_status_history 
            WHERE fingerprint = ? 
            ORDER BY changed_at
            """, (fingerprint,))
            
            history = []
            for row in self.cursor.fetchall():
                status, changed_at, raw_data = row
                history.append({
                    "status": status,
                    "changed_at": changed_at,
                    "raw_data": raw_data
                })
            
            return history
            
        except sqlite3.Error as e:
            logger.error(f"获取告警状态历史失败: {e}")
            return []
    
    def calculate_recurrence_count(self, fingerprint: str) -> int:
        """计算告警的实际复发次数（基于状态历史）"""
        try:
            history = self.get_alert_status_history(fingerprint)
            if len(history) < 2:
                return 0
            
            # 计算从resolved到active的次数
            recurrence_count = 0
            for i in range(1, len(history)):
                prev_status = history[i-1]["status"]
                curr_status = history[i]["status"]
                
                # 如果从resolved变为active，说明复发了
                if prev_status == "resolved" and curr_status in ["active", "firing"]:
                    recurrence_count += 1
            
            return recurrence_count
            
        except Exception as e:
            logger.error(f"计算复发次数失败: {e}")
            return 0
    
    def get_p3_alerts_weekly_summary(self) -> Dict[str, Any]:
        """获取P3告警的周报总结"""
        try:
            from datetime import datetime, timedelta
            
            # 获取一周前的时间
            week_ago = datetime.now() - timedelta(days=7)
            week_ago_str = week_ago.strftime("%Y-%m-%d %H:%M:%S")
            
            # 查询P3告警统计
            self.cursor.execute("""
            SELECT 
                COUNT(*) as total_count,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as active_count,
                COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved_count,
                COUNT(CASE WHEN recurrence_count > 0 THEN 1 END) as recurrence_count,
                SUM(recurrence_count) as total_recurrence
            FROM alerts 
            WHERE severity = 'P3' AND created_at >= ?
            """, (week_ago_str,))
            
            stats = self.cursor.fetchone()
            
            # 查询P3告警详情
            self.cursor.execute("""
            SELECT alertname, COUNT(*) as count, SUM(recurrence_count) as total_recurrence
            FROM alerts 
            WHERE severity = 'P3' AND created_at >= ?
            GROUP BY alertname
            ORDER BY count DESC
            LIMIT 10
            """, (week_ago_str,))
            
            alert_details = self.cursor.fetchall()
            
            return {
                "total_count": stats[0] if stats else 0,
                "active_count": stats[1] if stats else 0,
                "resolved_count": stats[2] if stats else 0,
                "recurrence_alert_count": stats[3] if stats else 0,
                "total_recurrence": stats[4] if stats else 0,
                "alert_details": alert_details,
                "period": f"{week_ago.strftime('%Y-%m-%d')} 至 {datetime.now().strftime('%Y-%m-%d')}"
            }
            
        except sqlite3.Error as e:
            logger.error(f"获取P3周报失败: {e}")
            return {}
    
    def get_weekly_alert_summary(self) -> Dict[str, Any]:
        """获取所有告警的周报统计"""
        try:
            from datetime import datetime, timedelta
            
            # 获取一周前的时间
            week_ago = datetime.now() - timedelta(days=7)
            week_ago_str = week_ago.strftime("%Y-%m-%d %H:%M:%S")
            
            # 获取各等级告警统计
            self.cursor.execute("""
            SELECT 
                severity,
                COUNT(*) as total_count,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as active_count,
                COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved_count,
                COUNT(CASE WHEN is_recurrence = 1 THEN 1 END) as recurrence_count,
                COUNT(CASE WHEN repeat_push_count > 0 THEN 1 END) as repeat_push_count,
                SUM(repeat_push_count) as total_repeat_push
            FROM alerts 
            WHERE created_at >= ?
            GROUP BY severity
            ORDER BY 
                CASE severity 
                    WHEN 'P0' THEN 1 
                    WHEN 'P1' THEN 2 
                    WHEN 'P2' THEN 3 
                    WHEN 'P3' THEN 4 
                    ELSE 5 
                END
            """, (week_ago_str,))
            
            severity_stats = {}
            total_stats = {
                "total_count": 0,
                "active_count": 0,
                "resolved_count": 0,
                "recurrence_count": 0,
                "repeat_push_count": 0,
                "total_repeat_push": 0
            }
            
            for row in self.cursor.fetchall():
                severity, total, active, resolved, recurrence, repeat_push, total_repeat_push = row
                severity_stats[severity] = {
                    "total_count": total,
                    "active_count": active,
                    "resolved_count": resolved,
                    "recurrence_count": recurrence,
                    "repeat_push_count": repeat_push,
                    "total_repeat_push": total_repeat_push or 0
                }
                
                # 累计总数
                total_stats["total_count"] += total
                total_stats["active_count"] += active
                total_stats["resolved_count"] += resolved
                total_stats["recurrence_count"] += recurrence
                total_stats["repeat_push_count"] += repeat_push
                total_stats["total_repeat_push"] += (total_repeat_push or 0)
            
            # 获取告警类型统计（TOP 10）
            self.cursor.execute("""
            SELECT 
                alertname,
                COUNT(*) as count,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as active_count
            FROM alerts 
            WHERE created_at >= ?
            GROUP BY alertname
            ORDER BY count DESC
            LIMIT 10
            """, (week_ago_str,))
            
            top_alerts = []
            for row in self.cursor.fetchall():
                alertname, count, active_count = row
                top_alerts.append({
                    "alertname": alertname, 
                    "count": count,
                    "active_count": active_count
                })
            
            return {
                "period": f"{week_ago.strftime('%Y-%m-%d')} 至 {datetime.now().strftime('%Y-%m-%d')}",
                "total_stats": total_stats,
                "severity_stats": severity_stats,
                "top_alerts": top_alerts
            }
            
        except sqlite3.Error as e:
            logger.error(f"获取告警周报统计失败: {e}")
            return {}
    
    def get_all_alerts(self) -> List[Dict[str, Any]]:
        """获取所有告警数据"""
        try:
            self.cursor.execute("SELECT raw_data FROM alerts ORDER BY updated_at DESC")
            rows = self.cursor.fetchall()
            
            alerts = []
            for row in rows:
                if row[0]:
                    try:
                        alert_data = json.loads(row[0])
                        alerts.append(alert_data)
                    except json.JSONDecodeError:
                        continue
            
            return alerts
        except sqlite3.Error as e:
            logger.error(f"获取所有告警失败: {e}")
            return []
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    db = AlertDatabase("test_alerts.db")
    
    test_alert = {
        "fingerprint": "test_fingerprint",
        "status": {"state": "firing"},
        "labels": {"alertname": "TestAlert", "severity": "critical"},
        "annotations": {"summary": "测试告警", "description": "测试描述"}
    }
    
    success = db.store_alert(test_alert)
    print(f"存储告警: {'成功' if success else '失败'}")
    
    status = db.get_alert_last_status("test_fingerprint")
    print(f"告警状态: {status}")
    
    db.close()
    os.remove("test_alerts.db")
    print("数据库测试成功")
