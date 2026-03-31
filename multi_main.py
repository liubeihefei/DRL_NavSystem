#!/usr/bin/env python3
"""
多进程导航系统启动文件

将每个节点运行在独立进程中，充分利用多核 CPU。
每个节点完全独立，拥有自己的执行器和进程空间。

进程分配：
- ekf_fusion_node:    EKF 定位融合（单线程）
- lidar_costmap_node: 激光雷达处理（单线程）
- map_planner_node:   地图+规划（3个线程组：数据/雷达/规划）
- controller_node:    控制（单线程）

使用方式：
  python3 multi_main.py

按 Ctrl+C 可优雅关闭所有节点。
"""

import os
import sys
import signal
import time
import logging
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
from datetime import datetime
from typing import List, Tuple, Optional


# ========== 工具函数 ==========

def get_project_root() -> str:
    """获取项目根目录"""
    return os.path.dirname(os.path.abspath(__file__))


def ensure_log_dir(log_dir: str) -> None:
    """确保日志目录存在"""
    os.makedirs(log_dir, exist_ok=True)


# ========== 日志处理 ==========

class QueueHandler(logging.Handler):
    """日志队列处理器，将子进程的日志发送到主进程"""

    def __init__(self, queue: Queue, node_name: str):
        super().__init__()
        self._queue = queue
        self._node_name = node_name

    def emit(self, record):
        try:
            msg = self.format(record)
            self._queue.put((self._node_name, record.levelname, msg, record.asctime))
        except Exception:
            pass


def log_worker(log_queue: Queue, log_file_path: str, stop_event: Event) -> None:
    """日志收集进程 - 将所有节点日志写入文件和控制台"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s [%(process)d][%(name)s] %(levelname)s: %(message)s')

    # 文件 handler
    ensure_log_dir(os.path.dirname(log_file_path))
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    root_logger.info("=" * 60)
    root_logger.info("Log worker started")
    root_logger.info(f"Log file: {log_file_path}")
    root_logger.info("=" * 60)

    while not stop_event.is_set():
        try:
            # 非阻塞获取日志
            while True:
                try:
                    node_name, level, message, timestamp = log_queue.get_nowait()
                    log_entry = f"[{node_name}] {message}"

                    record = logging.LogRecord(
                        name=node_name,
                        level=getattr(logging, level, logging.INFO),
                        pathname='',
                        lineno=0,
                        msg=log_entry,
                        args=(),
                        exc_info=None
                    )
                    record.created = time.mktime(time.strptime(timestamp)) if timestamp else time.time()
                    record.asctime = timestamp or time.strftime('%Y-%m-%d %H:%M:%S')

                    for handler in root_logger.handlers:
                        handler.emit(record)
                except Exception:
                    break
            time.sleep(0.05)
        except Exception as e:
            root_logger.error(f"Log worker error: {e}")

    root_logger.info("Log worker stopped")


# ========== 节点包装函数 ==========

def node_wrapper(
    node_module: str,
    run_func_name: str,
    node_name: str,
    log_dir: str,
    log_timestamp: str,
    log_queue: Queue,
    init_delay: float = 0.0
) -> None:
    """
    节点包装函数，运行在子进程中

    Args:
        node_module: 模块名 (如 'ekf_fusion_node')
        run_func_name: 入口函数名 (如 'run_ekf_fusion_node')
        node_name: 节点名
        log_dir: 日志目录
        log_timestamp: 时间戳
        log_queue: 日志队列
        init_delay: 初始化延迟（秒），用于控制启动顺序
    """
    # 设置进程名
    mp.current_process().name = node_name

    # 设置 Python 日志
    logger = logging.getLogger(node_name)
    logger.setLevel(logging.INFO)

    # 添加队列处理器
    queue_handler = QueueHandler(log_queue, node_name)
    queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(queue_handler)

    logger.info(f"{node_name} starting in process {os.getpid()}")

    # 等待延迟（用于控制启动顺序）
    if init_delay > 0:
        logger.info(f"Waiting {init_delay}s before initialization...")
        time.sleep(init_delay)

    try:
        # 动态导入并调用节点入口函数
        module = __import__(node_module, fromlist=[run_func_name])

        run_func = getattr(module, run_func_name, None)
        if run_func is None:
            logger.error(f"Function '{run_func_name}' not found in module '{node_module}'")
            return

        # 调用节点的入口函数
        run_func(log_dir=log_dir, log_timestamp=log_timestamp)

    except Exception as e:
        logger.error(f"{node_name} crashed: {e}", exc_info=True)
        raise
    finally:
        logger.info(f"{node_name} exited")


# ========== 进程管理器 ==========

class ProcessManager:
    """进程管理器 - 启动、监控、关闭所有节点进程"""

    def __init__(self):
        self.processes: List[Tuple[str, Process]] = []
        self.log_queue: Queue = Queue()
        self.stop_event: Event = Event()

        # 生成统一时间戳
        self.start_timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 创建统一日志目录
        project_root = get_project_root()
        self.log_dir: str = os.path.join(project_root, 'logs', f'navigation_{self.start_timestamp}')
        ensure_log_dir(self.log_dir)

        self.log_file: str = os.path.join(self.log_dir, f'multiprocess_main_{self.start_timestamp}.log')

    def _start_log_worker(self) -> None:
        """启动日志收集进程"""
        self.log_process = Process(
            target=log_worker,
            args=(self.log_queue, self.log_file, self.stop_event),
            name='log_worker',
            daemon=True
        )
        self.log_process.start()
        time.sleep(0.2)  # 等待日志进程启动

    def start_node(
        self,
        node_module: str,
        run_func_name: str,
        node_name: str,
        init_delay: float = 0.0
    ) -> Process:
        """启动单个节点进程"""
        p = Process(
            target=node_wrapper,
            args=(
                node_module,
                run_func_name,
                node_name,
                self.log_dir,
                self.start_timestamp,
                self.log_queue,
                init_delay
            ),
            name=node_name
        )
        p.start()
        self.processes.append((node_name, p))
        print(f"  + {node_name:<20} PID: {p.pid}  (delay: {init_delay:.1f}s)")
        return p

    def _print_banner(self, is_test_mode: bool) -> None:
        """打印启动横幅"""
        print(f"\n{'=' * 70}")
        print(f"       MULTI-PROCESS NAVIGATION SYSTEM")
        print(f"{'=' * 70}")
        print(f"  Session:     {self.start_timestamp}")
        print(f"  Log dir:     {self.log_dir}")
        print(f"  Test mode:   {'YES (controller disabled)' if is_test_mode else 'NO'}")
        print(f"{'=' * 70}")
        print(f"  Starting nodes:")
        print()

    def _print_ready(self) -> None:
        """打印就绪信息"""
        print()
        print(f"{'=' * 70}")
        print(f"  All nodes started!")
        print(f"  Press Ctrl+C to gracefully shutdown all nodes")
        print(f"{'=' * 70}\n")

    def _print_status(self) -> None:
        """打印所有进程状态"""
        print(f"\n  {'Node':<25} {'PID':<8} {'Status'}")
        print(f"  {'-'*25} {'-'*8} {'-'*10}")
        for name, p in self.processes:
            status = "running" if p.is_alive() else f"exited({p.exitcode})"
            print(f"  {name:<25} {p.pid:<8} {status}")

    def start_all(self) -> None:
        """启动所有节点"""
        from config_loader import get_config
        config = get_config()
        is_test_mode = config.get('common.test_mode', False)

        # 打印横幅
        self._print_banner(is_test_mode)

        # 启动日志收集进程
        self._start_log_worker()

        # 定义节点启动信息
        # 每个节点: (module_name, run_func_name, display_name, init_delay)
        nodes = [
            ('ekf_fusion_node',    'run_ekf_fusion_node',    'ekf_fusion_node',    0.0),
            ('lidar_costmap_node',  'run_lidar_costmap_node', 'lidar_costmap_node', 0.5),
            ('map_planner_node',    'run_map_planner_node',   'map_planner_node',   1.0),
        ]

        if not is_test_mode:
            nodes.append(
                ('controller_node', 'run_controller_node', 'controller_node', 2.0)
            )

        # 依次启动节点
        for module, func, name, delay in nodes:
            self.start_node(module, func, name, init_delay=delay)

        # 打印就绪信息
        self._print_ready()

    def shutdown(self, signum=None, frame=None) -> None:
        """关闭所有节点"""
        print("\n\nShutdown signal received...")
        print("Stopping all nodes...\n")

        # 首先发送停止事件
        self.stop_event.set()

        # 优雅关闭每个进程
        for name, p in self.processes:
            if p.is_alive():
                print(f"  Stopping {name:<20} (PID: {p.pid})...")
                p.terminate()

        # 等待进程结束
        for name, p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                print(f"  Force killing {name:<20} (PID: {p.pid})...")
                p.kill()
                p.join(timeout=2)

        # 关闭日志进程
        if hasattr(self, 'log_process') and self.log_process.is_alive():
            self.log_process.terminate()
            self.log_process.join(timeout=2)

        # 打印最终状态
        print("\n  Final status:")
        self._print_status()
        print("\n  All nodes stopped.")

    def wait(self) -> None:
        """等待所有节点运行"""
        try:
            while True:
                time.sleep(2)

                # 检查进程状态
                all_dead = True
                for name, p in self.processes:
                    if p.is_alive():
                        all_dead = False
                    elif p.exitcode is not None and p.exitcode != 0:
                        print(f"\n  WARNING: {name} exited with code {p.exitcode}")

                if all_dead:
                    print("\n  All processes have exited.")
                    break

        except KeyboardInterrupt:
            self.shutdown()

    def get_cpu_usage(self) -> None:
        """获取各进程 CPU 使用情况（调试用）"""
        try:
            import psutil
            print("\n  CPU Usage per Process:")
            for name, p in self.processes:
                try:
                    proc = psutil.Process(p.pid)
                    cpu = proc.cpu_percent(interval=0.1)
                    print(f"    {name:<20}: {cpu:.1f}%")
                except:
                    pass
        except ImportError:
            pass


def main() -> None:
    """主入口"""
    print("Initializing Multi-Process Navigation System...")

    # 设置 multiprocessing 启动方式为 'spawn'
    # 这样可以确保跨平台兼容性，避免 fork 相关的问题
    try:
        mp.set_start_method('spawn', force=False)
    except RuntimeError:
        pass  # 已经设置过了

    manager = ProcessManager()

    # 设置信号处理
    signal.signal(signal.SIGINT, manager.shutdown)
    signal.signal(signal.SIGTERM, manager.shutdown)

    try:
        manager.start_all()
        manager.wait()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        manager.shutdown()
        raise


if __name__ == '__main__':
    main()
