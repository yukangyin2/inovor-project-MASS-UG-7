import sys
import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import warnings
warnings.filterwarnings('ignore')

class StateSpaceExplorer:
    def __init__(self, dimensions, array_size, max_queries):
        self.dim = dimensions
        self.size = array_size
        self.max_queries = max_queries
        self.queries_made = 0
        self.observed_points = []
        self.observed_values = []
        self.total_points = array_size ** dimensions
        self.query_ratio = max_queries / self.total_points
        self.init_gaussian_process()
        
    def init_gaussian_process(self):
        if self.query_ratio < 0.2:
            length_scale = max(1.0, self.size * 0.3)
            kernel = Matern(length_scale=length_scale, nu=2.5) + WhiteKernel(noise_level=1e-5)
        else:
            length_scale = max(1.0, self.size * 0.1)
            kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=1e-5)
            
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            alpha=1e-6,
            normalize_y=True
        )
    
    def get_all_indices(self):
        if self.dim == 1:
            return [[i] for i in range(self.size)]
        elif self.dim == 2:
            indices = []
            for i in range(self.size):
                for j in range(self.size):
                    indices.append([i, j])
            return indices
        else:  # 3D
            indices = []
            for i in range(self.size):
                for j in range(self.size):
                    for k in range(self.size):
                        indices.append([i, j, k])
            return indices
    
    def format_query(self, index):
        if self.dim == 1:
            return str(index[0])
        else:
            return ','.join(map(str, index))
    
    def parse_response(self, response):
        return float(response.strip())
    
    def query(self, index):
        if self.queries_made >= self.max_queries:
            return None
            
        print(self.format_query(index))
        sys.stdout.flush()
        response = input()
        value = self.parse_response(response)
        self.observed_points.append(index)
        self.observed_values.append(value)
        self.queries_made += 1
        
        return value
    
    def select_next_query(self, remaining_queries):
        if not self.observed_points:
            if self.dim == 1:
                return [self.size // 2]
            elif self.dim == 2:
                return [self.size // 2, self.size // 2]
            else:
                return [self.size // 2, self.size // 2, self.size // 2]
        
        all_indices = self.get_all_indices()
        unqueried = []
        for idx in all_indices:
            if idx not in self.observed_points:
                unqueried.append(idx)
        
        if not unqueried:
            return None
        
        if remaining_queries > len(unqueried) * 0.3:
            return self.uncertainty_sampling(unqueried)
        else:
            return self.hybrid_sampling(unqueried, remaining_queries)
    
    def uncertainty_sampling(self, candidates):
        if len(self.observed_points) < 3:
            if self.dim == 1:
                for idx in [0, self.size - 1]:
                    if [idx] not in self.observed_points:
                        return [idx]
            return candidates[len(candidates) // 2]
        
        X = np.array(self.observed_points)
        y = np.array(self.observed_values)
        self.gp.fit(X, y)
        X_candidates = np.array(candidates)
        _, std = self.gp.predict(X_candidates, return_std=True)
        best_idx = np.argmax(std)
        return candidates[best_idx]
    
    def hybrid_sampling(self, candidates, remaining):
        if len(self.observed_points) < 2:
            if self.dim == 1:
                return [0] if [0] not in self.observed_points else [self.size - 1]
            return candidates[0]

        scores = []
        for candidate in candidates:
            min_dist = float('inf')
            for obs_point in self.observed_points:
                dist = np.linalg.norm(np.array(candidate) - np.array(obs_point))
                min_dist = min(min_dist, dist)
            scores.append(min_dist)

        best_idx = np.argmax(scores)
        return candidates[best_idx]
    
    def predict_all(self):
        if len(self.observed_points) < 2:
            if self.observed_values:
                const_value = np.mean(self.observed_values)
            else:
                const_value = 0.5
            return self.format_prediction([const_value] * self.total_points)
        
        X_train = np.array(self.observed_points)
        y_train = np.array(self.observed_values)
        all_indices = self.get_all_indices()
        X_all = np.array(all_indices)
        if self.query_ratio > 0.5 or len(self.observed_points) > self.total_points * 0.4:
            try:
                self.gp.fit(X_train, y_train)
                predictions = self.gp.predict(X_all)
            except:
                predictions = self.rbf_interpolation(X_train, y_train, X_all)
        else:
            predictions = self.smooth_interpolation(X_train, y_train, X_all)
        
        predictions = np.clip(predictions, 
                            min(y_train) - 0.1 * np.std(y_train),
                            max(y_train) + 0.1 * np.std(y_train))
        
        return self.format_prediction(predictions)
    
    def rbf_interpolation(self, X_train, y_train, X_all):
        try:
            rbf = RBFInterpolator(X_train, y_train, 
                                smoothing=0.001, 
                                kernel='multiquadric')
            return rbf(X_all)
        except:
            return self.simple_interpolation(X_train, y_train, X_all)
    
    def smooth_interpolation(self, X_train, y_train, X_all):
        predictions = []
        
        for point in X_all:
            distances = []
            for i, obs_point in enumerate(X_train):
                dist = np.linalg.norm(point - obs_point)
                distances.append((dist, y_train[i]))

            if min(d[0] for d in distances) < 1e-6:
                predictions.append([d[1] for d in distances if d[0] < 1e-6][0])
            else:
                sigma = self.size * 0.5 
                weights = []
                values = []
                for dist, val in distances:
                    weight = np.exp(-dist**2 / (2 * sigma**2))
                    weights.append(weight)
                    values.append(val)

                weights = np.array(weights)
                weights = weights / weights.sum()
                pred = np.sum(weights * np.array(values))
                predictions.append(pred)
        
        return np.array(predictions)
    
    def simple_interpolation(self, X_train, y_train, X_all):
        predictions = []
        for point in X_all:
            distances = [np.linalg.norm(point - obs) for obs in X_train]
            nearest_idx = np.argmin(distances)
            predictions.append(y_train[nearest_idx])
        return np.array(predictions)
    
    def format_prediction(self, predictions):
        if self.dim == 1:
            return ' '.join([f"{v:.6f}" for v in predictions])
        elif self.dim == 2:
            result = []
            for i in range(self.size):
                row = predictions[i*self.size:(i+1)*self.size]
                result.append(' '.join([f"{v:.6f}" for v in row]))
            return ' '.join(result)
        else: 
            result = []
            for i in range(self.size):
                layer_start = i * self.size * self.size
                layer_end = (i + 1) * self.size * self.size
                layer = predictions[layer_start:layer_end]
                layer_rows = []
                for j in range(self.size):
                    row = layer[j*self.size:(j+1)*self.size]
                    layer_rows.append(' '.join([f"{v:.6f}" for v in row]))
                result.append(' '.join(layer_rows))
            return ' '.join(result)
    
    def run(self):
        while self.queries_made < self.max_queries:
            remaining = self.max_queries - self.queries_made
            next_query = self.select_next_query(remaining)
            
            if next_query is None:
                break
                
            self.query(next_query)
        
        prediction = self.predict_all()
        print(prediction)
        sys.stdout.flush()

def main():
    if len(sys.argv) != 4:
        sys.stderr.write("Usage: python program.py <dimensions> <array_size> <max_queries>\n")
        sys.exit(1)
    
    try:
        dimensions = int(sys.argv[1])
        array_size = int(sys.argv[2])
        max_queries = int(sys.argv[3])

        if dimensions not in [1, 2, 3]:
            raise ValueError("Dimensions must be 1, 2, or 3")
        if array_size <= 0:
            raise ValueError("Array size must be positive")
        if max_queries <= 0:
            raise ValueError("Max queries must be positive")

        explorer = StateSpaceExplorer(dimensions, array_size, max_queries)
        explorer.run()
        
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()