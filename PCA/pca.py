import numpy as np

class PCA:    
    def __init__(self):
        # Initializing PCA class
        self.centered_dataset = None
        self.mean = None
        self.sorted_evals = None
        self.principal_components = None
        self.scores = None
        self.topk_pcs = None


    def fit_data(self,dataset):
        # Calculating mean of the dataset
        self.mean = np.mean(dataset, axis = 0)
        
        # Centering dataset by subtracting mean from dataset
        self.centered_dataset = dataset - self.mean

        # Calculating covariance matrix of the centered dataset
        # The same can be achieved by using covariance_matrix = np.cov(centered_dataset,rowvar=False) 
        covariance_matrix = (1/dataset.shape[0])*(self.centered_dataset.T @ self.centered_dataset)

        # Calculating eigenvalues and eigen vectors of the covariance matrix
        evals,evecs = np.linalg.eigh(covariance_matrix)

        # np.argsort() returns the index values of the sorted array so we can sort eigen values and eigen vectors (also principal components) in descending order
        # Which also helps us to select top k eigen vectors which contains more information of the dataset
        sorted_index = np.argsort(evals)[::-1]
        self.sorted_evals = evals[sorted_index]
        self.principal_components = evecs[:, sorted_index]
        print("Dataset fitted successfully")

    def compress(self,ratio= 95):
        if(ratio <= 100):
            # default set to 95 so Algorithm will choose top k principal components which explains 95% of the data
            current_sum = 0
            sum_of_evals = np.sum(self.sorted_evals)
            top_k = 0
    
            # for loop iterates over array of eigen values to find no of eigen vectors needed to retrive 95% of information
            for i in range(len(self.sorted_evals)):
                current_sum += self.sorted_evals[i]
                top_k = top_k + 1
                if(current_sum / sum_of_evals >= ratio/100):
                    break
            
            # Selecting top k principal components
            self.topk_pcs = self.principal_components[:, :top_k]
    
            # Projecting centered dataset onto lower dimension using the formula (X^T . W)
            self.scores = (self.centered_dataset @ self.topk_pcs)

            print(f"The data projected into the low dimensional PCA space of {top_k} dimension")
    
            # Returing Original dataset by adding back the mean
            return self.scores
        else:
            print("The ratio should be less than or equal to 100")

    def represent(self):
        return (self.scores @ self.topk_pcs.T) + self.mean
        
        