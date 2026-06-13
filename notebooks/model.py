import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Tiện ích chuyển đổi dữ liệu thành numpy array
def to_numpy(data):
    if hasattr(data, "toarray"):
        return data.toarray()
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    return np.asarray(data)

# =====================================================================
# LỚP 1: RIDGE REGRESSION FROM SCRATCH (Ridge_LogSales)
# Hồi quy tuyến tính có hiệu chỉnh L2 (tối ưu hóa closed-form)
# =====================================================================
class RidgeRegressionScratch(BaseEstimator, RegressorMixin):
    """
    Mô hình Ridge Regression tự viết từ đầu sử dụng công thức nghiệm Closed-Form:
    weights = (X^T X + alpha * I)^(-1) X^T y
    """
    def __init__(self, alpha=10.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        self.is_fitted_ = False

    def fit(self, X, y):
        X_np = to_numpy(X).astype(np.float32)
        y_np = to_numpy(y).astype(np.float32).reshape(-1, 1)
        n_samples, n_features = X_np.shape
        
        # Thêm cột bias nếu fit_intercept=True
        if self.fit_intercept:
            X_design = np.c_[np.ones(n_samples, dtype=np.float32), X_np]
            penalty = self.alpha * np.eye(n_features + 1, dtype=np.float32)
            penalty[0, 0] = 0.0 # Không phạt bias/intercept
        else:
            X_design = X_np
            penalty = self.alpha * np.eye(n_features, dtype=np.float32)
            
        # Tính toán closed-form
        A = X_design.T @ X_design + penalty
        weights = np.linalg.pinv(A) @ X_design.T @ y_np
        
        if self.fit_intercept:
            self.intercept_ = float(weights[0, 0])
            self.coef_ = weights[1:].flatten()
        else:
            self.intercept_ = 0.0
            self.coef_ = weights.flatten()
            
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Mô hình chưa được fit!")
        X_np = to_numpy(X).astype(np.float32)
        return X_np @ self.coef_ + self.intercept_


# =====================================================================
# LỚP 2: MLP REGRESSOR FROM SCRATCH (MLP_Scratch)
# Mạng nơ-ron có 1 lớp ẩn tự viết phần Lan truyền ngược (Backpropagation)
# =====================================================================
class MLPRegressorScratch(BaseEstimator, RegressorMixin):
    """
    Mạng nơ-ron có 1 lớp ẩn (1 hidden layer) tự viết dùng SGD.
    Cấu trúc: Input -> Hidden (ReLU) -> Output (Linear)
    """
    def __init__(self, hidden_dim=16, lr=0.01, epochs=20, batch_size=1024, random_state=42):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.loss_history_ = []
        self.is_fitted_ = False

    def _relu(self, Z):
        return np.maximum(0.0, Z)

    def _relu_deriv(self, Z):
        return (Z > 0.0).astype(np.float32)

    def fit(self, X, y):
        X_np = to_numpy(X).astype(np.float32)
        y_np = to_numpy(y).astype(np.float32).reshape(-1, 1)
        
        n_samples, n_features = X_np.shape
        rng = np.random.default_rng(self.random_state)
        
        # Khởi tạo He Initialization cho MLP
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_features), size=(n_features, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((1, self.hidden_dim), dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.hidden_dim), size=(self.hidden_dim, 1)).astype(np.float32)
        self.b2 = np.zeros((1, 1), dtype=np.float32)
        self.loss_history_ = []
        
        for epoch in range(self.epochs):
            indices = rng.permutation(n_samples)
            X_shuffled = X_np[indices]
            y_shuffled = y_np[indices]
            
            epoch_loss = 0.0
            num_batches = int(np.ceil(n_samples / self.batch_size))
            
            for b in range(num_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, n_samples)
                batch_len = end - start
                
                if batch_len == 0:
                    continue
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward Pass
                Z1 = X_batch @ self.W1 + self.b1
                A1 = self._relu(Z1)
                y_pred = A1 @ self.W2 + self.b2
                
                loss = np.mean((y_pred - y_batch) ** 2)
                epoch_loss += loss * batch_len
                
                # Backward Pass
                dZ2 = y_pred - y_batch # (batch_len, 1)
                dW2 = (A1.T @ dZ2) / batch_len
                db2 = np.mean(dZ2, axis=0, keepdims=True)
                
                dA1 = dZ2 @ self.W2.T # (batch_len, hidden_dim)
                dZ1 = dA1 * self._relu_deriv(Z1) # (batch_len, hidden_dim)
                dW1 = (X_batch.T @ dZ1) / batch_len
                db1 = np.mean(dZ1, axis=0, keepdims=True)
                
                # Gradient Descent update
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                
            self.loss_history_.append(epoch_loss / n_samples)
            
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Mô hình chưa được fit!")
        X_np = to_numpy(X).astype(np.float32)
        Z1 = X_np @ self.W1 + self.b1
        A1 = self._relu(Z1)
        pred = A1 @ self.W2 + self.b2
        return pred.flatten()


# =====================================================================
# LỚP 3: HISTOGRAM GRADIENT BOOSTING FROM SCRATCH (HistGradientBoosting_LogSales_160)
# Tự code vòng lặp Boosting và phần rời rạc hóa (discretization/binning) thuộc tính
# =====================================================================
class HistGradientBoostingRegressorScratch(BaseEstimator, RegressorMixin):
    """
    Mô hình Histogram-based Gradient Boosting Regressor tự viết từ đầu.
    - Phân giỏ thuộc tính liên tục thành 256 giỏ (percentile-based binning).
    - Vòng lặp boosting tuần tự để khớp các phần dư (pseudo-residuals).
    - Sử dụng DecisionTreeRegressor làm mô hình cơ sở xây dựng trên các đặc trưng đã phân giỏ.
    """
    def __init__(self, max_iter=160, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.1, random_state=42):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self.estimators_ = []
        self.init_value_ = 0.0
        self.bin_thresholds_ = []
        self.is_fitted_ = False

    def fit(self, X, y):
        X_np = to_numpy(X).astype(np.float32)
        y_np = to_numpy(y).astype(np.float32)
        n_samples, n_features = X_np.shape
        
        # 1. Rời rạc hóa đặc trưng liên tục (Binning)
        self.bin_thresholds_ = []
        X_binned = np.zeros_like(X_np, dtype=np.float32)
        
        for j in range(n_features):
            col = X_np[:, j]
            # Tạo 256 phân vị
            thresholds = np.unique(np.percentile(col, np.linspace(0, 100, 256)))
            self.bin_thresholds_.append(thresholds)
            # Gom giỏ
            X_binned[:, j] = np.digitize(col, thresholds[1:-1]).astype(np.float32)
            
        # 2. Khởi tạo F0 = mean(y)
        self.init_value_ = float(np.mean(y_np))
        F = np.full(n_samples, self.init_value_, dtype=np.float32)
        
        self.estimators_ = []
        
        # 3. Vòng lặp Boosting tuần tự
        for m in range(self.max_iter):
            # Tính pseudo-residuals (Phần dư)
            residuals = y_np - F
            
            # Huấn luyện cây quyết định khớp phần dư
            tree = DecisionTreeRegressor(
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=self.random_state + m,
                ccp_alpha=self.l2_regularization * 0.01 # Phạt L2 quy đổi
            )
            tree.fit(X_binned, residuals)
            
            # Cập nhật dự báo
            F += self.learning_rate * tree.predict(X_binned)
            self.estimators_.append(tree)
            
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Mô hình chưa được fit!")
        X_np = to_numpy(X).astype(np.float32)
        n_samples, n_features = X_np.shape
        
        # Gom giỏ dữ liệu test dựa trên ngưỡng học từ tập train
        X_binned = np.zeros_like(X_np, dtype=np.float32)
        for j in range(n_features):
            thresholds = self.bin_thresholds_[j]
            X_binned[:, j] = np.digitize(X_np[:, j], thresholds[1:-1]).astype(np.float32)
            
        # Dự báo tích lũy từ các cây
        F = np.full(n_samples, self.init_value_, dtype=np.float32)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X_binned)
            
        return F


# =====================================================================
# LỚP 4: LỚP TỔNG HỢP DOUBLE CHECK (ModelDoubleChecker)
# Đánh giá độ lệch MAE giữa mô hình tự viết (Scratch) và thư viện gốc (Sklearn)
# =====================================================================
class ModelDoubleChecker:
    """
    Lớp tổng hợp dùng để double check chất lượng dự báo giữa mô hình tự viết (Scratch)
    và mô hình thư viện chuẩn (Scikit-Learn).
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results_ = {}

    def check_models(self, X_train, y_train, X_val, y_val):
        X_tr = to_numpy(X_train).astype(np.float32)
        y_tr = to_numpy(y_train).astype(np.float32)
        X_v = to_numpy(X_val).astype(np.float32)
        y_v = to_numpy(y_val).astype(np.float32)
        
        # 1. So sánh Ridge
        ridge_scratch = RidgeRegressionScratch(alpha=10.0)
        ridge_sklearn = Ridge(alpha=10.0)
        
        ridge_scratch.fit(X_tr, y_tr)
        ridge_sklearn.fit(X_tr, y_tr)
        
        p_scratch = ridge_scratch.predict(X_v)
        p_sklearn = ridge_sklearn.predict(X_v)
        
        self.results_["Ridge"] = {
            "Scratch_MAE": float(mean_absolute_error(y_v, p_scratch)),
            "Sklearn_MAE": float(mean_absolute_error(y_v, p_sklearn)),
            "Absolute_Diff": float(abs(mean_absolute_error(y_v, p_scratch) - mean_absolute_error(y_v, p_sklearn)))
        }
        
        # 2. So sánh MLP (Huấn luyện 10 epochs để đối chiếu nhanh)
        mlp_scratch = MLPRegressorScratch(hidden_dim=16, lr=0.01, epochs=10, batch_size=1024, random_state=self.random_state)
        mlp_sklearn = MLPRegressor(hidden_layer_sizes=(16,), activation='relu', solver='sgd', 
                                   learning_rate='constant', learning_rate_init=0.01, max_iter=10, 
                                   batch_size=1024, random_state=self.random_state)
        
        mlp_scratch.fit(X_tr, y_tr)
        mlp_sklearn.fit(X_tr, y_tr)
        
        p_scratch = mlp_scratch.predict(X_v)
        p_sklearn = mlp_sklearn.predict(X_v)
        
        self.results_["MLP"] = {
            "Scratch_MAE": float(mean_absolute_error(y_v, p_scratch)),
            "Sklearn_MAE": float(mean_absolute_error(y_v, p_sklearn)),
            "Absolute_Diff": float(abs(mean_absolute_error(y_v, p_scratch) - mean_absolute_error(y_v, p_sklearn)))
        }
        
        # 3. So sánh HistGradientBoosting (Chạy 30 cây để kiểm chứng nhanh)
        hgb_scratch = HistGradientBoostingRegressorScratch(max_iter=30, learning_rate=0.1, random_state=self.random_state)
        hgb_sklearn = HistGradientBoostingRegressor(max_iter=30, learning_rate=0.1, random_state=self.random_state)
        
        hgb_scratch.fit(X_tr, y_tr)
        hgb_sklearn.fit(X_tr, y_tr)
        
        p_scratch = hgb_scratch.predict(X_v)
        p_sklearn = hgb_sklearn.predict(X_v)
        
        self.results_["HistGradientBoosting"] = {
            "Scratch_MAE": float(mean_absolute_error(y_v, p_scratch)),
            "Sklearn_MAE": float(mean_absolute_error(y_v, p_sklearn)),
            "Absolute_Diff": float(abs(mean_absolute_error(y_v, p_scratch) - mean_absolute_error(y_v, p_sklearn)))
        }
        
        return pd.DataFrame(self.results_).T


# =====================================================================
# WRAPPERS VÀ HÀM TIỆN ÍCH DÀNH CHO PIPELINES TRONG NOTEBOOK
# =====================================================================
class SklearnMLPRegressorWrapper(BaseEstimator, RegressorMixin):
    """Wrapper cho MLPRegressor để Scikit-Learn pipelines kiểm tra fit tự động."""
    def __init__(self, hidden_dim=16, lr=0.01, epochs=20, random_state=42):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim,),
            activation='relu',
            solver='sgd',
            learning_rate='constant',
            learning_rate_init=self.lr,
            max_iter=self.epochs,
            batch_size=1024,
            random_state=self.random_state
        )
        self.is_fitted_ = False

    def fit(self, X, y):
        X_np = to_numpy(X)
        y_np = to_numpy(y)
        self.model.fit(X_np, y_np)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X_np = to_numpy(X)
        return self.model.predict(X_np)


def get_models(one_hot_preprocessor, ordinal_preprocessor, random_state=42):
    """
    Trả về danh sách 7 cấu hình pipeline để huấn luyện và so sánh trong Notebook.
    """
    return {
        "Ridge_LogSales": Pipeline(
            steps=[("preprocess", one_hot_preprocessor), ("model", Ridge(alpha=10.0))]
        ),
        "Ridge_Scratch": Pipeline(
            steps=[("preprocess", one_hot_preprocessor), ("model", RidgeRegressionScratch(alpha=10.0))]
        ),
        "MLP_Sklearn": Pipeline(
            steps=[
                ("preprocess", one_hot_preprocessor),
                ("model", SklearnMLPRegressorWrapper(hidden_dim=16, lr=0.01, epochs=20, random_state=random_state))
            ]
        ),
        "MLP_Scratch": Pipeline(
            steps=[
                ("preprocess", one_hot_preprocessor),
                ("model", MLPRegressorScratch(hidden_dim=16, lr=0.01, epochs=20, batch_size=1024, random_state=random_state))
            ]
        ),
        "HistGradientBoosting_LogSales_80": Pipeline(
            steps=[
                ("preprocess", ordinal_preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=80,
                        learning_rate=0.06,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting_LogSales_160": Pipeline(
            steps=[
                ("preprocess", ordinal_preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=160,
                        learning_rate=0.04,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting_Scratch_160": Pipeline(
            steps=[
                ("preprocess", ordinal_preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressorScratch(
                        max_iter=160,
                        learning_rate=0.04,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
