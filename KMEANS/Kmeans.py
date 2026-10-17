import numpy as np
import matplotlib.pyplot as plt

class kmeans:
    def __init__(self, n_try, n_iter, n_cluster):
        self.n_try = n_try
        self.n_iter = n_iter
        self.n_cluster = n_cluster
        self.centers = None

    def discal(self, aoc, dp):
        dis = float('inf')
        cn = None
        for p in aoc:
            dist = np.linalg.norm(dp - p)
            if dist < dis:
                dis = dist
                cn = p
        return cn

    def fit_data(self, dataset):
        _dim = dataset.shape[1]
        best_centers = None
        min_inertia = float('inf')

        for i in range(self.n_try):
            
            n_center = np.random.uniform(low=dataset.min(), high=dataset.max(), size=(self.n_cluster, _dim))

            for j in range(self.n_iter):
                clusters = {tuple(center): [] for center in n_center}

                for k in dataset:
                    center = self.discal(n_center, k)
                    clusters[tuple(center)].append(k)
                
                new_centers = []
                for center, points in clusters.items():
                    if len(points) > 0:
                        new_centers.append(np.mean(points, axis=0))
                    else:
                        new_centers.append(center)
                
                n_center = np.array(new_centers)

            
            current_inertia = 0
            for k in dataset:
                
                dis = float('inf')
                for p in n_center:
                    dist = np.sum((k - p) ** 2)
                    if dist < dis:
                        dis = dist
                current_inertia += dis

            if current_inertia < min_inertia:
                min_inertia = current_inertia
                best_centers = n_center

        
        self.centers = best_centers
        print(f"Training completed successfully! Best Inertia: {min_inertia:.4f}")
        return self.centers
    
    def plot(self, dataset):
        """Plots the dataset points colored by their cluster assignment, along with final centers."""
        if self.centers is None:
            print("Error: You must run fit_data(dataset) before plotting!")
            return

        # Group dataset points by their closest final center
        clusters = {tuple(center): [] for center in self.centers}
        for k in dataset:
            center = self.discal(self.centers, k)
            clusters[tuple(center)].append(k)

        # Set up the plot
        plt.figure(figsize=(8, 6))
        colors = ['red', 'blue', 'yellow', 'green', 'orange', 'purple', 'cyan', 'magenta']

        # Plot each cluster's points
        for i, (center, points) in enumerate(clusters.items()):
            if len(points) > 0:
                points_arr = np.array(points)
                color = colors[i % len(colors)]
                plt.scatter(points_arr[:, 0], points_arr[:, 1], color=color, label=f'Cluster {i+1}', alpha=0.6)

            # Plot the cluster center as a large X
            plt.scatter(center[0], center[1], color='black', marker='X', s=250, edgecolors='white', linewidth=1.5)

        plt.title('K-Means Cluster Assignments (Best of All Tries)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()