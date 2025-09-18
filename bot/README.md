# Venus Monitoring

一个基于 Prometheus 和 Grafana 的 Kubernetes 监控解决方案，提供全面的集群监控、告警和可视化功能。

## 功能特性

- **全面监控**: 支持节点、Pod、服务、GPU、硬件等全方位监控
- **智能告警**: 多级别告警规则，支持 P1/P2 优先级分类
- **可视化仪表板**: 基于 Grafana 的丰富监控仪表板
- **自动化部署**: 基于 Helm 的一键部署和升级
- **可扩展架构**: 支持自定义 Exporter 和告警规则

## 架构组件

### 核心监控
- **Prometheus Operator**: 管理 Prometheus 实例和配置
- **Prometheus**: 时序数据库和查询引擎
- **Alertmanager**: 告警管理和通知
- **Grafana**: 监控数据可视化

### 数据采集器
- **Node Exporter**: 节点系统指标采集
- **Kube State Metrics**: Kubernetes资源状态指标
- **IPMI Exporter**: 硬件BMC指标采集（内置）
- **NVIDIA GPU Exporter**: GPU设备指标采集（子Chart）
- **Harbor Exporter**: Harbor仓库指标采集（子Chart）
- **PostgreSQL Exporter**: 数据库指标采集（子Chart）

### 告警规则
- **GPU告警**: 温度、功耗、ECC错误、掉卡等
- **节点告警**: CPU、内存、磁盘、网络、RDMA等
- **硬件告警**: IPMI传感器状态、温度、风扇、电源、电压等
- **K8s告警**: Pod状态、节点状态、资源使用等

## 快速开始

### 前置要求

- Kubernetes 1.19+
- Helm 3.0+
- kubectl 配置正确

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd venus-monitoring
   ```

2. **一键安装**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **访问服务**
   ```bash
   # 访问 Grafana
   kubectl port-forward -n monitoring svc/venus-monitoring-grafana 3000:80
   
   # 访问 Prometheus
   kubectl port-forward -n monitoring svc/venus-monitoring-prometheus-server 9090:80
   ```

### 默认访问信息

- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090

## 配置说明

### 基础配置

```yaml
# values.yaml
global:
  namespace: monitoring

# 启用/禁用组件
nodeExporter:
  enabled: true

ipmiExporter:
  enabled: true
  ipmi:
    username: "admin"
    password: "admin"
    targets:
      - "10.42.1.201"
      - "10.42.1.202"
```

### 自定义配置

```bash
# 使用自定义 values 文件
./install.sh -f values-production.yaml

# 指定命名空间
./install.sh -n my-monitoring

# 升级现有安装
./install.sh -u
```

## 使用指南

### 监控范围

#### 1. 节点监控
- CPU 使用率、负载、温度
- 内存使用率、交换空间
- 磁盘 I/O、使用率、错误
- 网络流量、错误率、连接数
- RDMA 性能指标

#### 2. Kubernetes 监控
- Pod 状态、重启次数、资源使用
- 节点状态、容量、调度
- 服务发现、端点状态
- 存储卷状态、持久化

#### 3. GPU 监控
- GPU 温度、功耗、利用率
- 显存使用率、ECC 错误
- 计算单元状态、掉卡检测
- 多 GPU 集群管理

#### 4. 硬件监控
- IPMI 传感器状态
- 温度、风扇、电源监控
- 电压、电流、功耗
- 内存 ECC 错误、磁盘状态

### 告警等级

#### P1 级别（严重）
- GPU 掉卡、温度过高
- 节点不可用、磁盘满
- 硬件故障、电源异常
- 内存 ECC 错误

#### P2 级别（警告）
- 资源使用率过高
- 网络错误率过高
- 服务重启频繁
- 硬件状态异常

## 故障排除

### 常见问题

1. **Pod 启动失败**
   ```bash
   kubectl describe pod <pod-name> -n monitoring
   kubectl logs <pod-name> -n monitoring
   ```

2. **指标采集失败**
   ```bash
   # 检查 ServiceMonitor
   kubectl get servicemonitor -n monitoring
   
   # 检查 Prometheus 配置
   kubectl get prometheus -n monitoring -o yaml
   ```

3. **告警不触发**
   ```bash
   # 检查 PrometheusRule
   kubectl get prometheusrule -n monitoring
   
   # 检查 Alertmanager 配置
   kubectl get alertmanager -n monitoring -o yaml
   ```

### 日志查看

```bash
# 查看所有 Pod 状态
kubectl get pods -n monitoring

# 查看特定组件日志
kubectl logs -n monitoring -l app.kubernetes.io/component=prometheus
kubectl logs -n monitoring -l app.kubernetes.io/component=grafana
```

## 升级和维护

### 升级 Chart

```bash
# 更新依赖
helm dependency update

# 升级安装
./install.sh -u
```

### 备份配置

```bash
# 备份 values
helm get values venus-monitoring -n monitoring > backup-values.yaml

# 备份 Prometheus 数据
kubectl exec -n monitoring <prometheus-pod> -- tar czf /tmp/backup.tar.gz /prometheus
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License