import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class QuantStrategy:
    def __init__(self, data_path):
        # 读取数据并标准化列名
        self.data = pd.read_excel(data_path)
        self._preprocess_data()

    def _preprocess_data(self):
        """数据预处理"""
        # 重命名列（中英混合列名标准化）
        column_mapping = {
            '股票代码_Stkcd': 'stock_code',
            '最新股票名称_Lstknm': 'stock_name',
            '日期_Date': 'date',
            '收盘价_ClPr': 'close',
            '月收益率_Monret': 'monthly_return',
            '总股数_Fullshr': 'total_shares',
            '等权平均市场月收益率_Mreteq': 'market_return',
            '月无风险收益率_Monrfret': 'risk_free',
            '信息发布日期_Infopubdt': 'info_date'
        }
        self.data.rename(columns=column_mapping, inplace=True)

        # 转换日期格式
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['info_date'] = pd.to_datetime(self.data['info_date'])

        # 假设ST股票标记为名称包含"ST"（需根据实际数据调整）
        self.data['is_ST'] = self.data['stock_name'].str.contains('ST')

        # 假设行业信息需要外部数据，此处仅示例
        self.data['industry'] = '未知'  # 需要补充行业数据

        # 计算超额收益率
        self.data['excess_return'] = self.data['monthly_return'] - self.data['risk_free']

        # 计算市值
        self.data['market_value'] = self.data['close'] * self.data['total_shares']

    def calculate_iv(self):
        """计算特质波动率"""
        iv_list = []

        # 获取三因子数据（示例，需替换实际因子数据）
        # 假设已合并三因子数据到主数据，列名为MKT/SMB/HML
        # 此处使用市场收益率作为示例
        self.data['MKT'] = self.data['market_return'] - self.data['risk_free']
        self.data['SMB'] = 0  # 需补充规模因子
        self.data['HML'] = 0  # 需补充价值因子

        for stock, group in self.data.groupby('stock_code'):
            try:
                X = group[['MKT', 'SMB', 'HML']]
                X = sm.add_constant(X)
                y = group['excess_return']

                model = sm.OLS(y, X).fit()
                residuals = model.resid
                iv = np.sqrt(np.var(residuals))

                iv_list.append({
                    'stock_code': stock,
                    'iv': iv,
                    'last_date': group['date'].max()
                })
            except:
                continue

        self.iv_data = pd.DataFrame(iv_list)
        self.data = pd.merge(self.data, self.iv_data, on='stock_code')

    def build_portfolios(self):
        """构建投资组合"""
        # 生成调仓日期序列（每月最后交易日）
        self.data['year_month'] = self.data['date'].dt.to_period('M')
        rebalance_dates = self.data.groupby('year_month')['date'].max().values

        all_portfolios = []
        for i in range(len(rebalance_dates) - 1):
            current_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            # 获取当前期数据
            current_data = self.data[self.data['date'] == current_date]

            # 过滤条件
            filtered = current_data[
                (~current_data['is_ST']) &
                (current_data['industry'] != '金融') &
                (current_data['monthly_return'].between(-0.0999, 0.0999))  # 近似排除涨跌停
                ]

            # 按IV分组
            filtered['group'] = pd.qcut(filtered['iv'], 30, labels=False)

            # 添加持有期信息
            filtered['hold_start'] = current_date
            filtered['hold_end'] = next_date
            all_portfolios.append(filtered)

        self.portfolios = pd.concat(all_portfolios)

    def calculate_returns(self):
        """计算组合收益"""
        # 合并持有期收益数据
        merged = pd.merge(
            self.portfolios[['stock_code', 'hold_start', 'group']],
            self.data[['stock_code', 'date', 'monthly_return']],
            left_on=['stock_code', 'hold_start'],
            right_on=['stock_code', 'date'],
            how='left'
        )

        # 计算等权重组合收益
        portfolio_returns = merged.groupby(['hold_start', 'group']).apply(
            lambda x: np.mean(x['monthly_return'])
        ).reset_index(name='portfolio_return')

        # 计算累积收益
        portfolio_returns['cum_return'] = (
            portfolio_returns.groupby('group')['portfolio_return']
            .transform(lambda x: (1 + x).cumprod())
        )
        self.portfolio_returns = portfolio_returns


    def calculate_portfolio_metrics(self, top_group):
        """计算最优投资组合的指标"""
        # 获取最优组合的收益数据
        top_group_returns = self.portfolio_returns[self.portfolio_returns['group'] == top_group]

        # 计算指标
        arithmetic_mean = np.mean(top_group_returns['portfolio_return'])  # 几何平均
        geometric_mean = np.exp(np.log(top_group_returns['portfolio_return']).mean())  # 算数平均
        std_dev = np.std(top_group_returns['portfolio_return'])  # 标准差
        max_return = np.max(top_group_returns['portfolio_return'])  # 最大值
        min_return = np.min(top_group_returns['portfolio_return'])  # 最小值
        win_rate = np.mean(top_group_returns['portfolio_return'] > 0)  # 收益胜率
        annualized_return = geometric_mean ** (12 / len(top_group_returns)) - 1  # 年化收益率
        max_drawdown = (1 - top_group_returns['cum_return'] / top_group_returns['cum_return'].cummax()).max()  # 最大回撤

        return {
            '几何平均': arithmetic_mean,
            '算数平均': geometric_mean,
            '标准差': std_dev,
            '最大值': max_return,
            '最小值': min_return,
            '收益胜率': win_rate,
            '年化收益率': annualized_return,
            '最大回撤': max_drawdown
        }

    def visualize_returns(self):
        """可视化累积收益"""
        plt.figure(figsize=(12, 7))

        for group in self.portfolio_returns['group'].unique():
            group_data = self.portfolio_returns[self.portfolio_returns['group'] == group]
            plt.plot(group_data['hold_start'],
                     group_data['cum_return'],
                     label=f'Group {group}',
                     alpha=0.7)

        plt.title('各个投资组合累积收益率', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('累积净值（初始=1）', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        plt.savefig('各个投资组合累积收益率.png', dpi=300)
        plt.show()


class EnhancedQuantStrategy(QuantStrategy):
    def build_portfolios(self):
        """构建投资组合"""
        # 生成调仓日期序列（每月最后交易日）
        self.data['year_month'] = self.data['date'].dt.to_period('M')
        rebalance_dates = self.data.groupby('year_month')['date'].max().values

        all_portfolios = []
        for i in range(len(rebalance_dates) - 1):
            current_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            # 获取当前期数据
            current_data = self.data[self.data['date'] == current_date]

            # 过滤条件
            filtered = current_data[
                (~current_data['is_ST']) &
                (current_data['industry'] != '金融') &
                (current_data['monthly_return'].between(-0.0999, 0.0999))  # 近似排除涨跌停
                ]

            # 按特质波动率分组
            filtered['iv_group'] = pd.qcut(filtered['iv'], 6, labels=False)
            # 按市值分组
            filtered['market_value_group'] = pd.qcut(filtered['market_value'], 6, labels=False)
            # 创建最终的组合分组，结合特质波动率组和市值组
            filtered['group'] = filtered['iv_group'].astype(str) + '_' + filtered['market_value_group'].astype(str)

            # 添加持有期信息
            filtered['hold_start'] = current_date
            filtered['hold_end'] = next_date
            all_portfolios.append(filtered)

        self.portfolios = pd.concat(all_portfolios)

# 使用示例
if __name__ == "__main__":
    strategy = EnhancedQuantStrategy('按月整理的近10年A股数据.xlsx')
    strategy.calculate_iv()
    strategy.build_portfolios()
    strategy.calculate_returns()
    strategy.visualize_returns()

    # 选择累积收益最高的投资组合
    top_group = strategy.portfolio_returns.groupby('group')['cum_return'].last().idxmax()
    metrics = strategy.calculate_portfolio_metrics(top_group)
    print(metrics)

    # 提取该组合的每月收益率数据
    best_monthly_returns = strategy.portfolios[strategy.portfolios['group'] == top_group]
    best_monthly_returns = best_monthly_returns.sort_values('hold_start')

    plt.figure(figsize=(12, 7))
    # 绘制每月收益率的条形图（绿色表示正收益，红色表示负收益）
    plt.bar(best_monthly_returns['hold_start'], best_monthly_returns['monthly_return'],
            color=np.where(best_monthly_returns['monthly_return'] > 0, 'green', 'red'), width=25)
    plt.xlabel('日期')
    plt.ylabel('月收益率/%')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.title(f"累积收益最高的组合（{top_group}）的每月收益")
    plt.savefig('最高投资组合的各月收益.png', dpi=300)
    plt.show()
