import sys

def main():
    dimensions = int(sys.argv[1])
    size = int(sys.argv[2])
    max_queries = int(sys.argv[3])

    if dimensions != 2:
        sys.exit(1)

    point_sums = {}
    point_counts = {}
    all_points = []
    for i in range(size):
        for j in range(size):
            all_points.append((i, j))
            point_sums[(i, j)] = 0.0
            point_counts[(i, j)] = 0
    
    total_points = len(all_points)
    samples_per_point = max_queries // total_points
    
    queries_made = 0

    for round_num in range(samples_per_point):
        for i, j in all_points:
            if queries_made >= max_queries:
                break
            print("{},{}".format(i, j))
            sys.stdout.flush()
            
            response = sys.stdin.readline().strip()
            try:
                value = float(response)
            except:
                value = 0.5
            
            point_sums[(i, j)] += value
            point_counts[(i, j)] += 1
            queries_made += 1
        
        if queries_made >= max_queries:
            break
    
    results = []
    for i, j in all_points:
        if point_counts[(i, j)] > 0:
            avg = point_sums[(i, j)] / point_counts[(i, j)]
        else:
            avg = 0.5
        results.append(avg)
    output_strings = []
    for val in results:
        output_strings.append("{:.6f}".format(val))
    
    print(" ".join(output_strings))
    sys.stdout.flush()

if __name__ == "__main__":
    main()