import sys
import numpy as np
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings('ignore')

class FinalCompetitionSolver:
    def __init__(self, dimensions, array_size, max_queries):
        self.dim = dimensions
        self.size = array_size
        self.max_queries = max_queries
        self.queries_made = 0
        self.total_points = array_size ** dimensions
        
        # 使用字典来收集观测值
        # key: tuple(point) -> list of values
        # (虽然我们只采样一次，但保留这个结构很方便)
        self.point_values = {} 
        
        print(f"# Final Solver: 3D, {self.total_points} points, {self.max_queries} queries", file=sys.stderr)
        print(f"# Strategy: Uniform Random Sampling + KDTree + Tiling (for wrap) + IDW (k=5)", file=sys.stderr)

    def format_query(self, index):
        return ','.join(map(str, index))

    def query(self, index):
        if self.queries_made >= self.max_queries:
            return None
            
        print(self.format_query(index))
        sys.stdout.flush()
        
        response = input().strip()
        value = float(response)
        
        self.queries_made += 1
        
        # 存储观测值
        key = tuple(index)
        if key not in self.point_values:
            self.point_values[key] = []
        self.point_values[key].append(value)
        
        return value

    def run(self):
        """
        执行完整的采样和预测流程
        """
        # 1. 采样: 使用所有预算在整个空间进行均匀随机采样
        print("# Phase 1: Uniform Random Sampling", file=sys.stderr)
        self.sample_uniformly(self.max_queries)
        
        print(f"# Sampling complete. Queries used: {self.queries_made}/{self.max_queries}", file=sys.stderr)
        
        # 2. 预测: 使用带环绕处理的KDTree和IDW进行预测
        print("# Phase 2: Generating predictions...", file=sys.stderr)
        prediction = self.predict_idw_with_wrap(k_neighbors=5, power=2)
        
        print("# Prediction complete.", file=sys.stderr)
        print(prediction)
        sys.stdout.flush()

    def sample_uniformly(self, budget):
        """
        在整个 1M 空间中采样 'budget' 个唯一的随机点。
        这解决了采样偏差和“stochastic”陷阱。
        """
        print(f"# Generating {budget} unique random points for sampling...", file=sys.stderr)
        
        # 1. 生成 'budget' 个唯一的 1D 索引 (0 到 999,999)
        # replace=False 确保了所有点都是唯一的
        unique_indices = np.random.choice(self.total_points, budget, replace=False)
        
        points_to_query = []
        for idx in unique_indices:
            # 2. 将 1D 索引转换回 3D 坐标 (i, j, k)
            i = idx // (self.size * self.size)
            j = (idx % (self.size * self.size)) // self.size
            k = idx % self.size
            points_to_query.append([i, j, k])
            
        print(f"# Querying {len(points_to_query)} points...", file=sys.stderr)
        
        # 3. 执行所有查询
        for i, pt in enumerate(points_to_query):
            if self.queries_made >= self.max_queries:
                break
            self.query(pt)
            if (i + 1) % 20000 == 0:
                print(f"# Sampled {i+1}/{budget} points", file=sys.stderr)

    def predict_idw_with_wrap(self, k_neighbors=5, power=2):
        """
        使用KDTree和反距离加权(IDW)进行快速预测。
        通过“平铺”(Tiling)数据来处理周期性边界。
        """
        
        # --- 步骤 1: 准备训练数据 ---
        unique_points = []
        averaged_values = []
        
        for key, vals in self.point_values.items():
            unique_points.append(list(key))
            # 因为我们对每个点只采样一次，所以 mean(vals) 就是 vals[0]
            averaged_values.append(np.mean(vals)) 
            
        X_train = np.array(unique_points)
        y_train = np.array(averaged_values)
        
        print(f"# Using {len(X_train)} unique observation points for prediction", file=sys.stderr)

        # --- 步骤 2: 平铺数据以处理周期性环绕 (关键!) ---
        # 假设环绕轴是 0 (i 轴)
        wrap_axis = 0 
        
        X_train_tiled = [X_train]
        
        # 1. 创建 -size 副本 (i-100)
        X_train_minus = X_train.copy()
        X_train_minus[:, wrap_axis] -= self.size
        X_train_tiled.append(X_train_minus)
        
        # 2. 创建 +size 副本 (i+100)
        X_train_plus = X_train.copy()
        X_train_plus[:, wrap_axis] += self.size
        X_train_tiled.append(X_train_plus)
        
        # 3. 堆叠成一个 (900k, 3) 的数组
        X_final = np.vstack(X_train_tiled)
        # y_train 也需要平铺 3 次
        y_final = np.tile(y_train, 3) 
        
        print(f"# Building KD-tree on {len(X_final)} tiled points...", file=sys.stderr)
        
        # --- 步骤 3: 构建 KD-tree ---
        # 这是快速搜索的核心
        tree = KDTree(X_final)
        
        print("# KD-tree built. Starting batch prediction...", file=sys.stderr)
        
        # --- 步骤 4: 批量预测 ---
        predictions = np.zeros(self.total_points, dtype=np.float32)
        idx = 0
        
        # 为了内存效率，我们一次只生成和处理一个 i-slice (100x100 = 10k 点)
        for i in range(self.size):
            batch_indices = []
            for j in range(self.size):
                for k in range(self.size):
                    batch_indices.append([i, j, k])
            
            batch_indices = np.array(batch_indices)
            
            # 查询 KD-tree 找到 K 个最近的邻居
            distances, indices = tree.query(batch_indices, k=k_neighbors)
            
            # --- 步骤 5: 应用反距离加权 (IDW) ---
            for bi in range(len(batch_indices)):
                dist = distances[bi]
                ind = indices[bi]
                
                # 检查除零错误 (如果查询点恰好是采样点)
                if dist[0] < 1e-10:
                    pred_value = y_final[ind[0]]
                else:
                    # weights = 1 / (distance^power)
                    weights = 1.0 / (dist ** power)
                    weights_sum = np.sum(weights)
                    values = y_final[ind]
                    
                    # 预测值 = sum(w * v) / sum(w)
                    pred_value = np.sum(weights * values) / weights_sum
                
                predictions[idx] = pred_value
                idx += 1
                
            if (i + 1) % 10 == 0:
                print(f"# Processed layer {i+1}/{self.size}", file=sys.stderr)
                
        return self.format_prediction_fast(predictions)

    def format_prediction_fast(self, predictions):
        """
        使用 numpy 高效格式化输出字符串
        """
        print("# Formatting output string...", file=sys.stderr)
        pred_3d = predictions.reshape(self.size, self.size, self.size)
        
        result_parts = []
        for i in range(self.size): # 遍历每个 2D 'layer'
            layer_parts = []
            for j in range(self.size): # 遍历 'layer' 中的每 'row'
                row = pred_3d[i, j, :]
                # .join() 是格式化字符串最快的方式
                row_str = ' '.join([f"{v:.6f}" for v in row])
                layer_parts.append(row_str)
            result_parts.append(' '.join(layer_parts))
            
        return ' '.join(result_parts)

def main():
    if len(sys.argv) != 4:
        sys.stderr.write("Usage: python program.py <dimensions> <array_size> <max_queries>\n")
        sys.exit(1)
    
    try:
        dimensions = int(sys.argv[1])
        array_size = int(sys.argv[2])
        max_queries = int(sys.argv[3])
        
        if dimensions != 3 or array_size != 100 or max_queries != 300000:
             sys.stderr.write(f"Warning: This solver is optimized for 3 100 300000.\n")
             sys.stderr.write(f"Running with: {dimensions} {array_size} {max_queries}\n")

        # 始终使用这个为最终竞赛优化的 Solver
        solver = FinalCompetitionSolver(dimensions, array_size, max_queries)
        solver.run()
        
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()