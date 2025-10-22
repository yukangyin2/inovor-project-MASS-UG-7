import sys
import numpy as np
from scipy.interpolate import RBFInterpolator, interp1d
from scipy.ndimage import gaussian_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import warnings
warnings.filterwarnings('ignore')


class CompetitionExplorer:
    def __init__(self, dimensions, array_size, max_queries):
        self.dim = dimensions
        self.size = array_size
        self.max_queries = max_queries
        self.queries_made = 0
        self.observed_points = []
        self.observed_values = []
        self.total_points = array_size ** dimensions
        self.query_ratio = max_queries / self.total_points
        

        self.competition_type = self.identify_competition()
        
    def identify_competition(self):

        if self.dim == 3:
            return "3d_medium"
        elif self.dim == 2 and self.size == 100:
            if self.max_queries == 1000:
                return "cylindrical"  
            elif self.max_queries == 20000:
                return "stochastic"   
        return "general"
    
    def format_query(self, index):

        if self.dim == 1:
            return str(index[0])
        return ','.join(map(str, index))
    
    def query(self, index):

        if self.queries_made >= self.max_queries:
            return None
            
        print(self.format_query(index))
        sys.stdout.flush()
        
        response = input().strip()
        value = float(response)
        
        self.observed_points.append(index)
        self.observed_values.append(value)
        self.queries_made += 1
        
        return value
    
    def get_all_indices(self):
        if self.dim == 1:
            return [[i] for i in range(self.size)]
        elif self.dim == 2:
            return [[i, j] for i in range(self.size) for j in range(self.size)]
        else:  # 3D
            return [[i, j, k] for i in range(self.size) 
                   for j in range(self.size) for k in range(self.size)]
    

    def run_3d_medium(self):
    
        print(f"# 3D Medium: size={self.size}, queries={self.max_queries}", file=sys.stderr)
        
    
        query_points = self.generate_3d_fast_sampling()
        
        print(f"# Generated {len(query_points)} sampling points", file=sys.stderr)
        
       
        for i, point in enumerate(query_points):
            if self.queries_made >= self.max_queries:
                break
            self.query(point)
            if (i + 1) % 100 == 0:
                print(f"# Progress: {i+1}/{len(query_points)} queries", file=sys.stderr)
        
        print(f"# Sampling complete, starting prediction...", file=sys.stderr)
        
      
        return self.predict_3d_fast()
    
    def generate_3d_fast_sampling(self):
        
        points = []
        
   
        for i in [0, self.size-1]:
            for j in [0, self.size-1]:
                for k in [0, self.size-1]:
                    points.append([i, j, k])
        
   
        mid = self.size // 2
        points.append([mid, mid, mid])

        remaining = self.max_queries - len(points)
        if remaining > 0:
         
            points_per_dim = max(3, int(np.cbrt(remaining)) + 1)
            points_per_dim = min(points_per_dim, self.size)  
            
            step = max(1, (self.size - 1) // (points_per_dim - 1))
            
            for i in range(0, self.size, step):
                for j in range(0, self.size, step):
                    for k in range(0, self.size, step):
                        pt = [min(i, self.size-1), 
                              min(j, self.size-1), 
                              min(k, self.size-1)]
                        if pt not in points and len(points) < self.max_queries:
                            points.append(pt)
        
        return points[:self.max_queries]

    
    def predict_3d_fast(self):

        all_indices = self.get_all_indices()
        total_points = len(all_indices)
        
        print(f"# Predicting {total_points} points...", file=sys.stderr)
        
        if len(self.observed_points) < 3:
            mean_val = np.mean(self.observed_values) if self.observed_values else 0.5
            predictions = [mean_val] * total_points
        else:
            X_train = np.array(self.observed_points)
            y_train = np.array(self.observed_values)
            X_all = np.array(all_indices)
            
       
            sample_ratio = len(self.observed_points) / total_points
            
            if sample_ratio > 0.3:
               
                print(f"# Using RBF interpolation (sample ratio: {sample_ratio:.2%})", file=sys.stderr)
                predictions = self.fast_rbf_interpolation(X_train, y_train, X_all)
            else:
               
                print(f"# Using weighted interpolation (sample ratio: {sample_ratio:.2%})", file=sys.stderr)
                predictions = self.fast_weighted_interpolation(X_train, y_train, X_all)
            
          
            predictions = np.clip(predictions, 
                                np.min(y_train) - 0.2*np.std(y_train),
                                np.max(y_train) + 0.2*np.std(y_train))
        
        print(f"# Prediction complete!", file=sys.stderr)
        return self.format_prediction(predictions)
    
    def fast_rbf_interpolation(self, X_train, y_train, X_all):
       
        try:
     
            max_train_points = 500
            if len(X_train) > max_train_points:
                indices = np.random.choice(len(X_train), max_train_points, replace=False)
                X_train = X_train[indices]
                y_train = y_train[indices]
                print(f"# Subsampled to {max_train_points} training points", file=sys.stderr)
            

            rbf = RBFInterpolator(X_train, y_train, 
                                kernel='multiquadric',
                                smoothing=0.1)
            
            batch_size = 5000
            predictions = []
            for i in range(0, len(X_all), batch_size):
                batch = X_all[i:i+batch_size]
                pred_batch = rbf(batch)
                predictions.extend(pred_batch)
                if i % 10000 == 0:
                    print(f"# Predicted {i}/{len(X_all)} points", file=sys.stderr)
            
            return np.array(predictions)
        except Exception as e:
            print(f"# RBF failed ({e}), using weighted interpolation", file=sys.stderr)
            return self.fast_weighted_interpolation(X_train, y_train, X_all)
    
    def fast_weighted_interpolation(self, X_train, y_train, X_all):

        predictions = []

        power = 2  
        max_neighbors = min(20, len(X_train))  
        
        for i, point in enumerate(X_all):
       
            distances = np.linalg.norm(X_train - point, axis=1)
          
            if np.min(distances) < 1e-10:
                nearest_idx = np.argmin(distances)
                predictions.append(y_train[nearest_idx])
            else:
               
                nearest_indices = np.argpartition(distances, min(max_neighbors, len(distances)-1))[:max_neighbors]
                nearest_distances = distances[nearest_indices]
                nearest_values = y_train[nearest_indices]
                
           
                weights = 1.0 / (nearest_distances ** power)
                weights = weights / np.sum(weights)
                
                prediction = np.sum(weights * nearest_values)
                predictions.append(prediction)
            
      
            if (i + 1) % 5000 == 0:
                print(f"# Predicted {i+1}/{len(X_all)} points", file=sys.stderr)
        
        return np.array(predictions)

    def run_cylindrical(self):

        query_points = self.generate_radial_sampling_points()
        
        for point in query_points:
            if self.queries_made >= self.max_queries:
                break
            self.query(point)
        

        return self.predict_cylindrical()
    
    def generate_radial_sampling_points(self):

        points = []
        center = self.size / 2.0
        
    
        center_int = self.size // 2
        points.append([center_int, center_int])
        
     
        max_radius = np.sqrt(2) * self.size / 2
        num_radii = int(np.sqrt(self.max_queries / 8))  
        
        for r_idx in range(1, num_radii):
            radius = (r_idx / num_radii) * max_radius
     
            num_angles = max(8, int(2 * np.pi * radius / 3))
            num_angles = min(num_angles, 36) 
            
            for angle_idx in range(num_angles):
                angle = 2 * np.pi * angle_idx / num_angles
                x = center + radius * np.cos(angle)
                y = center + radius * np.sin(angle)
                
               
                xi = int(np.round(x))
                yi = int(np.round(y))
                
                if 0 <= xi < self.size and 0 <= yi < self.size:
                    pt = [xi, yi]
                    if pt not in points and len(points) < self.max_queries:
                        points.append(pt)
        
       
        boundary_points = []
        for i in [0, self.size-1]:
            for j in range(0, self.size, 5):
                boundary_points.append([i, j])
                boundary_points.append([j, i])
        
        for pt in boundary_points:
            if pt not in points and len(points) < self.max_queries:
                points.append(pt)
        
    
        if len(points) < self.max_queries:
            step = max(3, self.size // int(np.sqrt(self.max_queries - len(points))))
            for i in range(0, self.size, step):
                for j in range(0, self.size, step):
                    pt = [i, j]
                    if pt not in points and len(points) < self.max_queries:
                        points.append(pt)
        
        return points[:self.max_queries]
    
    def predict_cylindrical(self):
       
        all_indices = self.get_all_indices()
        
        if len(self.observed_points) < 3:
            mean_val = np.mean(self.observed_values) if self.observed_values else 0.5
            predictions = [mean_val] * len(all_indices)
        else:
            X_train = np.array(self.observed_points)
            y_train = np.array(self.observed_values)
            X_all = np.array(all_indices)
            
            try:
                
                rbf = RBFInterpolator(X_train, y_train, 
                                    kernel='thin_plate_spline',
                                    smoothing=0.05)
                predictions = rbf(X_all)
                
                
                predictions = np.clip(predictions,
                                    np.min(y_train) - 0.1*np.std(y_train),
                                    np.max(y_train) + 0.1*np.std(y_train))
                
                
                predictions_2d = predictions.reshape(self.size, self.size)
                predictions_2d = gaussian_filter(predictions_2d, sigma=0.8)
                predictions = predictions_2d.flatten()
                
            except Exception as e:
            
                try:
                    kernel = RBF(length_scale=self.size*0.15) + WhiteKernel(noise_level=0.01)
                    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                                 alpha=1e-6, normalize_y=True)
                    gp.fit(X_train, y_train)
                    predictions = gp.predict(X_all)
                except:
                    predictions = self.nearest_neighbor_prediction(X_train, y_train, X_all)
        
        return self.format_prediction(predictions)
    

    def run_stochastic(self):

        
        all_indices = self.get_all_indices()
        num_points = len(all_indices)
        
        
        samples_per_point = self.max_queries // num_points
        remaining = self.max_queries % num_points
        
        print(f"# Stochastic strategy: {samples_per_point} samples per point, {remaining} extra", file=sys.stderr)
        
     
        point_values = {tuple(idx): [] for idx in all_indices}
        
  
        for sample_round in range(samples_per_point):
            print(f"# Round {sample_round + 1}/{samples_per_point}", file=sys.stderr)
            for idx in all_indices:
                if self.queries_made >= self.max_queries:
                    break
                val = self.query(idx)
                point_values[tuple(idx)].append(val)
            
            if self.queries_made >= self.max_queries:
                break
        
    
        if self.queries_made < self.max_queries:
            print(f"# Extra sampling: {remaining} queries", file=sys.stderr)
            for idx in all_indices[:remaining]:
                if self.queries_made >= self.max_queries:
                    break
                val = self.query(idx)
                point_values[tuple(idx)].append(val)
        
      
        predictions = []
        for idx in all_indices:
            key = tuple(idx)
            if point_values[key]:
                
                avg_value = np.mean(point_values[key])
                predictions.append(avg_value)
            else:
               
                predictions.append(0.5)
        
        print(f"# Total queries used: {self.queries_made}/{self.max_queries}", file=sys.stderr)
        print(f"# Sample statistics: min={np.min(predictions):.3f}, max={np.max(predictions):.3f}, mean={np.mean(predictions):.3f}", file=sys.stderr)
        
        return self.format_prediction(predictions)
    

    def nearest_neighbor_prediction(self, X_train, y_train, X_all):
 
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
        else:  # 3D
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

        if self.competition_type == "3d_medium":
            prediction = self.run_3d_medium()
        elif self.competition_type == "cylindrical":
            prediction = self.run_cylindrical()
        elif self.competition_type == "stochastic":
            prediction = self.run_stochastic()
        else:
            
            prediction = self.run_general()
        
        print(prediction)
        sys.stdout.flush()
    
    def run_general(self):

        query_points = self.generate_uniform_grid()
        for point in query_points:
            if self.queries_made >= self.max_queries:
                break
            self.query(point)

        all_indices = self.get_all_indices()
        if len(self.observed_points) < 3:
            mean_val = np.mean(self.observed_values) if self.observed_values else 0.5
            predictions = [mean_val] * len(all_indices)
        else:
            X_train = np.array(self.observed_points)
            y_train = np.array(self.observed_values)
            X_all = np.array(all_indices)
            
            try:
                rbf = RBFInterpolator(X_train, y_train, kernel='thin_plate_spline', smoothing=0.1)
                predictions = rbf(X_all)
            except:
                predictions = self.nearest_neighbor_prediction(X_train, y_train, X_all)
        
        return self.format_prediction(predictions)
    
    def generate_uniform_grid(self):
        points = []
        if self.dim == 1:
            step = max(1, self.size // self.max_queries)
            for i in range(0, self.size, step):
                if len(points) < self.max_queries:
                    points.append([i])
        elif self.dim == 2:
            points_per_dim = int(np.sqrt(self.max_queries))
            step = max(1, self.size // points_per_dim)
            for i in range(0, self.size, step):
                for j in range(0, self.size, step):
                    if len(points) < self.max_queries:
                        points.append([i, j])
        else:  # 3D
            points_per_dim = int(np.cbrt(self.max_queries))
            step = max(1, self.size // points_per_dim)
            for i in range(0, self.size, step):
                for j in range(0, self.size, step):
                    for k in range(0, self.size, step):
                        if len(points) < self.max_queries:
                            points.append([i, j, k])
        return points


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
        
        explorer = CompetitionExplorer(dimensions, array_size, max_queries)
        explorer.run()
        
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()