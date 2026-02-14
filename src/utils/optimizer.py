# adaptive weight optimization
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class WeightOptimizer:
    
    def __init__(self):
        self.loss_history = []
        self.weights_history = []
        self.bayesian_history = {'X': [], 'y': []}
    
    def update_adaptive(self, weights, loss, metrics, learning_rate=0.1):

        num_queries = weights['num_queries']
        
        if num_queries == 0:
            print("   First query - saving baseline weights")
            weights['num_queries'] = 1
            return weights
        
        elif num_queries == 1:
            print("   Second query - random exploration")
            return self._update_simple(weights, loss, learning_rate)
        
        elif num_queries < 5:
            print("   Using Evolutionary Strategy (gradient-based)")
            return self._update_evolutionary(weights, loss)
        
        else:
            print("   Using Online Bayesian Optimization")
            return self._update_bayesian(weights, loss)
    
    def _update_simple(self, weights, loss, learning_rate):
        print(f"\nRandom exploration (lr={learning_rate})...")
        
        weights['num_queries'] += 1
        
        if weights.get('avg_loss') is None:
            weights['avg_loss'] = loss
        else:
            weights['avg_loss'] = 0.9 * weights['avg_loss'] + 0.1 * loss
        
        if loss > 0.05 or np.random.rand() < 0.3:
            noise_scale = learning_rate * loss
            old_weights = weights.copy()
            
            weights['alpha'] += np.random.uniform(-noise_scale, noise_scale)
            weights['beta'] += np.random.uniform(-noise_scale, noise_scale)
            weights['gamma'] += np.random.uniform(-noise_scale, noise_scale)
            weights['delta'] += np.random.uniform(-noise_scale, noise_scale)
            
            total = weights['alpha'] + weights['beta'] + weights['gamma'] + weights['delta']
            weights['alpha'] /= total
            weights['beta'] /= total
            weights['gamma'] /= total
            weights['delta'] /= total
            
            weights['alpha'] = np.clip(weights['alpha'], 0.1, 0.6)
            weights['beta'] = np.clip(weights['beta'], 0.1, 0.5)
            weights['gamma'] = np.clip(weights['gamma'], 0.1, 0.5)
            weights['delta'] = np.clip(weights['delta'], 0.05, 0.3)
            weights['tau'] = np.clip(weights['tau'] + np.random.uniform(-0.05, 0.05), 0.3, 
0.7)
            weights['lambda'] = np.clip(weights['lambda'] + np.random.uniform(-0.01, 0.01), 
0.01, 0.1)
            
            total = weights['alpha'] + weights['beta'] + weights['gamma'] + weights['delta']
            weights['alpha'] /= total
            weights['beta'] /= total
            weights['gamma'] /= total
            weights['delta'] /= total
            
            print(f"   α: {old_weights['alpha']:.3f} → {weights['alpha']:.3f}")
            print(f"   β: {old_weights['beta']:.3f} → {weights['beta']:.3f}")
        else:
            print(f"   Weights unchanged (loss: {loss:.3f})")
        
        return weights
    
    def _update_evolutionary(self, weights, loss):
        print(f"\nEvolutionary Strategy...")
        
        weights['num_queries'] += 1
        
        self.loss_history.append(loss)
        self.weights_history.append({
            'alpha': weights['alpha'],
            'beta': weights['beta'],
            'gamma': weights['gamma'],
            'delta': weights['delta'],
            'tau': weights['tau'],
            'lambda': weights['lambda']
        })
        
        if len(self.loss_history) < 2:
            print("   Not enough history (need 2+ queries)")
            return weights
        
        loss_current = self.loss_history[-1]
        loss_previous = self.loss_history[-2]
        weights_current = self.weights_history[-1]
        weights_previous = self.weights_history[-2]
        
        gradients = {}
        learning_rate = 0.1
        
        for key in ['alpha', 'beta', 'gamma', 'delta', 'tau', 'lambda']:
            delta_weight = weights_current[key] - weights_previous[key]
            if abs(delta_weight) > 1e-6:
                gradients[key] = (loss_current - loss_previous) / delta_weight
            else:
                gradients[key] = 0.0
        
        old_weights = weights.copy()
        
        weights['alpha'] -= learning_rate * gradients['alpha']
        weights['beta'] -= learning_rate * gradients['beta']
        weights['gamma'] -= learning_rate * gradients['gamma']
        weights['delta'] -= learning_rate * gradients['delta']
        weights['tau'] -= learning_rate * gradients['tau'] * 0.1
        weights['lambda'] -= learning_rate * gradients['lambda'] * 0.1
        
        weights['alpha'] = np.clip(weights['alpha'], 0.1, 0.6)
        weights['beta'] = np.clip(weights['beta'], 0.1, 0.5)
        weights['gamma'] = np.clip(weights['gamma'], 0.1, 0.5)
        weights['delta'] = np.clip(weights['delta'], 0.05, 0.3)
        weights['tau'] = np.clip(weights['tau'], 0.3, 0.7)
        weights['lambda'] = np.clip(weights['lambda'], 0.01, 0.1)
        
        total = weights['alpha'] + weights['beta'] + weights['gamma'] + weights['delta']
        weights['alpha'] /= total
        weights['beta'] /= total
        weights['gamma'] /= total
        weights['delta'] /= total
        
        print(f"   Loss: {loss_previous:.3f} → {loss_current:.3f}")
        print(f"   α: {old_weights['alpha']:.3f} → {weights['alpha']:.3f}")
        
        return weights
    
    def _update_bayesian(self, weights, loss):
        print(f"\nBayesian Optimization...")
        
        weights['num_queries'] += 1
        
        current_x = [
            weights['alpha'],
            weights['beta'],
            weights['gamma'],
            weights['delta'],
            weights['tau'],
            weights['lambda']
        ]
        
        self.bayesian_history['X'].append(current_x)
        self.bayesian_history['y'].append(loss)
        
        if len(self.bayesian_history['X']) < 3:
            print(f"   Not enough data (need 3+ queries, have {len(self.bayesian_history['X'])})")
            return weights
        
        if len(self.bayesian_history['X']) > 50:
            self.bayesian_history['X'] = self.bayesian_history['X'][-50:]
            self.bayesian_history['y'] = self.bayesian_history['y'][-50:]
        
        X = np.array(self.bayesian_history['X'])
        y = np.array(self.bayesian_history['y'])
        
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, 
n_restarts_optimizer=2)
        gp.fit(X, y)
        
        n_candidates = 100
        candidates = []
        
        for _ in range(n_candidates):
            alpha = np.random.uniform(0.1, 0.6)
            beta = np.random.uniform(0.1, 0.5)
            gamma = np.random.uniform(0.1, 0.5)
            delta = np.random.uniform(0.05, 0.3)
            
            total = alpha + beta + gamma + delta
            alpha /= total
            beta /= total
            gamma /= total
            delta /= total
            
            tau = np.random.uniform(0.3, 0.7)
            lam = np.random.uniform(0.01, 0.1)
            
            candidates.append([alpha, beta, gamma, delta, tau, lam])
        
        candidates = np.array(candidates)
        mu, sigma = gp.predict(candidates, return_std=True)
        
        kappa = 2.0
        lcb = mu - kappa * sigma
        best_idx = np.argmin(lcb)
        best_candidate = candidates[best_idx]
        
        old_weights = weights.copy()
        
        weights['alpha'] = float(best_candidate[0])
        weights['beta'] = float(best_candidate[1])
        weights['gamma'] = float(best_candidate[2])
        weights['delta'] = float(best_candidate[3])
        weights['tau'] = float(best_candidate[4])
        weights['lambda'] = float(best_candidate[5])
        
        print(f"   Expected loss: {mu[best_idx]:.3f} ± {sigma[best_idx]:.3f}")
        print(f"   α: {old_weights['alpha']:.3f} → {weights['alpha']:.3f}")
        
        return weights

