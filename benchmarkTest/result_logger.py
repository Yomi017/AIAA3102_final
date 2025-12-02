"""
ALFworld 测试结果记录和统计模块
"""

import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path


class ResultLogger:
    """结果记录器"""
    
    def __init__(self, log_dir: str = "benchmark_results", prefix: str = "agent"):
        """
        初始化结果记录器
        
        Args:
            log_dir: 日志目录路径
            prefix: 测试类型前缀 ("agent" 或 "baseline")
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.prefix = prefix
        
        # 生成时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建本次测试的目录（带前缀）
        self.session_dir = self.log_dir / f"{prefix}_session_{self.timestamp}"
        self.session_dir.mkdir(exist_ok=True)
        
        # 文件路径
        self.summary_file = self.session_dir / "summary.json"
        self.detailed_csv = self.session_dir / "detailed_results.csv"
        self.statistics_file = self.session_dir / "statistics.txt"
        
        # 初始化数据
        self.results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
        print(f"📊 结果将保存到: {self.session_dir}")
    
    def add_result(self, result: Dict[str, Any]):
        """添加单个游戏结果"""
        self.results.append(result)
    
    def save_detailed_csv(self):
        """保存详细的CSV结果"""
        if not self.results:
            return
        
        with open(self.detailed_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'game_num', 
                'success', 
                'steps', 
                'successful_steps',
                'action_success_rate',
                'task_preview'
            ])
            writer.writeheader()
            
            for result in self.results:
                # 计算动作成功率
                steps = result.get('steps', 0)
                successful_steps = result.get('successful_steps', 0)
                action_rate = f"{successful_steps/steps*100:.1f}%" if steps > 0 else "N/A"
                
                # 任务预览（前100字符）
                task = result.get('task', '')
                task_preview = task.split('Your task is to:')[-1].strip()[:100] if 'Your task is to:' in task else task[:100]
                
                writer.writerow({
                    'game_num': result.get('game_num', 'N/A'),
                    'success': '✅' if result.get('success', False) else '❌',
                    'steps': steps,
                    'successful_steps': successful_steps,
                    'action_success_rate': action_rate,
                    'task_preview': task_preview
                })
        
        print(f"✅ 详细结果已保存: {self.detailed_csv}")
    
    def save_summary_json(self):
        """保存JSON格式的汇总数据"""
        summary = {
            'timestamp': self.timestamp,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'total_games': len(self.results),
            'results': self.results,
            'statistics': self.calculate_statistics()
        }
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 汇总数据已保存: {self.summary_file}")
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """计算统计数据"""
        if not self.results:
            return {}
        
        total_games = len(self.results)
        successful_games = sum(1 for r in self.results if r.get('success', False))
        
        total_steps = sum(r.get('steps', 0) for r in self.results)
        total_successful_steps = sum(r.get('successful_steps', 0) for r in self.results)
        
        # 任务完成率
        task_success_rate = successful_games / total_games * 100 if total_games > 0 else 0
        
        # 动作成功率
        action_success_rate = total_successful_steps / total_steps * 100 if total_steps > 0 else 0
        
        # 平均步数
        avg_steps = total_steps / total_games if total_games > 0 else 0
        avg_successful_steps = total_successful_steps / total_games if total_games > 0 else 0
        
        return {
            'task_completion': {
                'total_games': total_games,
                'successful_games': successful_games,
                'failed_games': total_games - successful_games,
                'success_rate_percent': round(task_success_rate, 2)
            },
            'action_execution': {
                'total_steps': total_steps,
                'successful_steps': total_successful_steps,
                'failed_steps': total_steps - total_successful_steps,
                'success_rate_percent': round(action_success_rate, 2)
            },
            'averages': {
                'avg_steps_per_game': round(avg_steps, 2),
                'avg_successful_steps_per_game': round(avg_successful_steps, 2)
            }
        }
    
    def save_statistics_txt(self):
        """保存可读的统计文本"""
        stats = self.calculate_statistics()
        
        if not stats or 'task_completion' not in stats:
            print("⚠️  没有足够的数据生成统计报告")
            return
        
        with open(self.statistics_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ALFworld 测试统计报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"测试时间: {self.timestamp}\n")
            f.write(f"测试时长: {(datetime.now() - self.start_time).total_seconds():.1f} 秒\n")
            f.write(f"会话目录: {self.session_dir}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("【任务完成情况】\n")
            f.write("-" * 70 + "\n")
            tc = stats['task_completion']
            f.write(f"总游戏数:     {tc['total_games']}\n")
            f.write(f"✅ 成功:      {tc['successful_games']} 个\n")
            f.write(f"❌ 失败:      {tc['failed_games']} 个\n")
            f.write(f"📈 成功率:    {tc['success_rate_percent']:.2f}%\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("【动作执行情况】\n")
            f.write("-" * 70 + "\n")
            ae = stats['action_execution']
            f.write(f"总执行步数:   {ae['total_steps']}\n")
            f.write(f"✅ 成功步数:  {ae['successful_steps']}\n")
            f.write(f"❌ 失败步数:  {ae['failed_steps']}\n")
            f.write(f"📈 成功率:    {ae['success_rate_percent']:.2f}%\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("【平均数据】\n")
            f.write("-" * 70 + "\n")
            avg = stats['averages']
            f.write(f"平均每局步数:         {avg['avg_steps_per_game']:.2f}\n")
            f.write(f"平均每局成功步数:     {avg['avg_successful_steps_per_game']:.2f}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("【详细结果】\n")
            f.write("-" * 70 + "\n")
            for i, result in enumerate(self.results, 1):
                success_icon = "✅" if result.get('success', False) else "❌"
                steps = result.get('steps', 0)
                successful_steps = result.get('successful_steps', 0)
                action_rate = f"{successful_steps/steps*100:.1f}" if steps > 0 else "0"
                
                f.write(f"\n游戏 {i}: {success_icon}\n")
                f.write(f"  执行步数: {steps}\n")
                f.write(f"  成功步数: {successful_steps}\n")
                f.write(f"  动作成功率: {action_rate}%\n")
                
                # 提取任务描述
                task = result.get('task', '')
                if 'Your task is to:' in task:
                    task_goal = task.split('Your task is to:')[-1].strip()
                    f.write(f"  任务: {task_goal[:80]}...\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("报告结束\n")
            f.write("=" * 70 + "\n")
        
        print(f"✅ 统计报告已保存: {self.statistics_file}")
    
    def print_summary(self):
        """在控制台打印汇总信息"""
        stats = self.calculate_statistics()
        
        print(f"\n\n{'='*70}")
        print("📊 测试统计汇总")
        print(f"{'='*70}")
        
        print(f"\n【任务完成情况】")
        tc = stats['task_completion']
        print(f"  总游戏数: {tc['total_games']}")
        print(f"  ✅ 成功: {tc['successful_games']}")
        print(f"  ❌ 失败: {tc['failed_games']}")
        print(f"  📈 任务完成率: {tc['success_rate_percent']:.2f}%")
        
        print(f"\n【动作执行情况】")
        ae = stats['action_execution']
        print(f"  总执行步数: {ae['total_steps']}")
        print(f"  ✅ 成功步数: {ae['successful_steps']}")
        print(f"  ❌ 失败步数: {ae['failed_steps']}")
        print(f"  📈 动作成功率: {ae['success_rate_percent']:.2f}%")
        
        print(f"\n【平均数据】")
        avg = stats['averages']
        print(f"  平均每局步数: {avg['avg_steps_per_game']:.2f}")
        print(f"  平均每局成功步数: {avg['avg_successful_steps_per_game']:.2f}")
        
        print(f"\n{'='*70}")
        print(f"📂 详细结果保存在: {self.session_dir}")
        print(f"{'='*70}\n")
    
    def finalize(self):
        """完成记录，保存所有文件"""
        self.save_detailed_csv()
        self.save_summary_json()
        self.save_statistics_txt()
        self.print_summary()
