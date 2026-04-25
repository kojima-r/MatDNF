from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, chi2

import numpy as np
from numpy.typing import NDArray

from mat_dnf.numpy.losses import (
    acc_classi,
    acc_dnf,
    logi_conseq,
    logi_equiv,
    pred_classi,
    pred_dnf,
)
from mat_dnf.numpy.models import MatDNF, train_mat_dnf
from mat_dnf.simplifications import remove_a_or_not_a_weak, simp_dnf
from mat_dnf.utils import (
    MeanLogger,
    n_parity_function,
    random_dnf,
    random_function,
    read_nth_dnf,
)


def _resolve_device(device_name: str):
    """Return (xp_module, rng) for the given device_name ('cpu' or 'gpu').

    CuPy is imported lazily so the module remains usable on systems without it
    when ``device_name == 'cpu'``.
    """
    if device_name == "gpu":
        import cupy as cp

        return cp, cp.random.default_rng()
    if device_name == "cpu":
        return np, np.random.default_rng()
    raise ValueError(
        f"Unknown device_name: {device_name!r} (expected 'cpu' or 'gpu')"
    )


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict_dnf_c(dnf, i1, xp=np):
    xx = (dnf @ xp.vstack([i1, 1 - i1])) == dnf.sum(axis=1)[:, None]
    return xx


class MatDNFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        h=1000,
        aa=4,
        er_max=0,
        alpha=0.1,
        max_itr=500,
        max_try=20,
        extra_itr=0,
        use_perturbation=True,
        use_sam=True,
        device_name="cpu",
    ):
        """
        Initializes the MatDNFClassifier.

        Args:
            h: Maximum number of conjunctions in a DNF.
            n: Number of variables.
            aa: Scaling factor for the random initialization.
            er_max: Maximum allowed error during training.
            alpha: Learning rate.
            max_itr: Maximum number of training iterations.
            max_try: Maximum number of attempts to find a better solution.
            extra_itr: Additional iterations after reaching zero error.
            use_perturbation: Whether to use perturbation during training.
            use_sam: Whether to use SAM optimizer.
            device_name: Device to use for training ('cpu' or 'gpu').
        """
        self.h = h
        self.aa = aa
        self.er_max = er_max
        self.alpha = alpha
        self.max_itr = max_itr
        self.max_try = max_try
        self.extra_itr = extra_itr
        self.use_perturbation = use_perturbation
        self.use_sam = use_sam
        self.device_name = device_name
        self.xp, self.rng = _resolve_device(device_name)
        self.model = None
        self.learned_dnf = None
        self.learned_dnf_weak = None
        self.n_feature = None
        self.feature_names = None
        self.target_name = None
        self.feature_indices = None

    def _prep(self, X, y=None):
        """
        Prepares input data for training or prediction.

        Args:
            X: Input features.
            y: Target variable (optional).

        Returns:
            xp: Array module (numpy or cupy).
            X: Prepared input features (transposed and on the active device).
            y: Prepared target variable (optional).
        """
        xp = self.xp
        if xp is np:
            X = X.T
        else:
            X = xp.asarray(X.T)
            if y is not None:
                y = xp.asarray(y)
        return xp, X, y

    def _post_process(self, out):
        """
        Post-processes the output data, transferring back to host memory if on GPU.
        """
        if self.xp is np:
            return out
        return self.xp.asnumpy(out)

    def _select_feature(self, X, feature_names=None):
        if self.feature_indices is not None:
            X = X[:, self.feature_indices]
            if feature_names is not None:
                feature_names = feature_names[self.feature_indices]
        return X, feature_names

    def transform(self, X, y, alpha=0.05, feature_names=None):

        selector = SelectKBest(
            score_func=chi2,
            k="all",
        )
        selector.fit(X, y)
        self.selector = selector
        # Get the p-values for each feature
        p_values = selector.pvalues_

        # Select features where p-value is less than alpha
        significant_feature_indices = np.where(p_values < alpha)[0]
        self.feature_indices = significant_feature_indices
        # Select significant features from X_train
        X_train_selected = X[:, self.feature_indices]
        feature_names_selected = None
        if feature_names is not None:
            feature_names_selected = feature_names[self.feature_indices]
        return X_train_selected, feature_names_selected

    def fit(
        self,
        X,
        y,
        feature_names=None,
        target_name=None,
        init_c=None,
        init_d_k=None,
        with_transform=False,
    ):
        """
        Fits the MatDNFClassifier model.

        Args:
            X: Training data features.
            y: Training data target variable.
            feature_names: Names of the input features.
            target_name: Name of the target variable.
            init_c: Initial value for the C-part of the model.
            init_d_k: Initial value for the D_k-part of the model.

        Returns:
            self: Fitted MatDNFClassifier instance.
        """
        n = X.shape[0]
        if feature_names is None:
            feature_names = ["x" + str(i) for i in range(n)]
        if with_transform:
            X, feature_names = self.transform(X, y, feature_names=feature_names)
        xp, X, y = self._prep(X, y)
        # X
        n = X.shape[0]
        self.n_feature = n
        self.feature_names = feature_names
        self.target_name = target_name

        if init_c is not None or init_d_k is not None:
            self.model = MatDNF(init_c, init_d_k, self.aa)
        else:
            self.model = MatDNF.create_random(
                h=self.h, n=n, aa=self.aa, xp=xp, rng=self.rng
            )

        self.model, self.v_k_th, learned_dnf, info = train_mat_dnf(
            model=self.model,
            i_in=X,
            i_out=y,
            er_max=self.er_max,
            alpha=self.alpha,
            max_itr=self.max_itr,
            max_try=self.max_try,
            extra_itr=self.extra_itr,
            fold=None,
            use_perturbation=self.use_perturbation,
            use_sam=self.use_sam,
            rng=self.rng,
        )
        c_th, d_k_th = info
        dnf_weak = remove_a_or_not_a_weak(
            self.model.c, c_th, d_k_th, eps=5.0e-1, thresh=0
        )
        dnf_weak_s = simp_dnf(dnf_weak)
        self.learned_dnf_weak = dnf_weak_s

        learned_dnf_s = simp_dnf(learned_dnf)
        self.learned_dnf = learned_dnf_s
        return self

    def predict(self, X, use_weak=False):
        """
        Predicts the class labels for the input data with DNF.

        Args:
            X: Input data features.

        Returns:
            predictions: Predicted class labels.
        """
        if self.learned_dnf is None:
            raise RuntimeError("Model has not been fitted yet.")
        X, _ = self._select_feature(X)
        xp, X, _ = self._prep(X)

        l2 = X.shape[1]
        if use_weak:
            predictions = pred_dnf(self.learned_dnf_weak, X, l2)
        else:
            predictions = pred_dnf(self.learned_dnf, X, l2)
        return self._post_process(predictions)

    def predict_conj(self, X, use_weak=False):
        """
        Predicts the truth values for each conjunction in the DNF.

        Args:
            X: Input data features.

        Returns:
            predictions: Truth values for each conjunction.
        """
        X, _ = self._select_feature(X)
        xp, X, _ = self._prep(X)
        if use_weak:
            predictions = predict_dnf_c(self.learned_dnf_weak, X, xp=xp).T
        else:
            predictions = predict_dnf_c(self.learned_dnf, X, xp=xp).T
        return self._post_process(predictions)

    def predict_proba(self, X):
        """
        Predicts the class probabilities for the input data.

        Args:
            X: Input data features.

        Returns:
            probabilities: Predicted class probabilities.
        """
        return sigmoid(self.predict_logproba(X))

    def predict_logproba(self, X):
        """
        Predicts the log probabilities for the input data.

        Args:
            X: Input data features.

        Returns:
            log_probabilities: Predicted log probabilities.
        """
        if self.learned_dnf is None:
            raise RuntimeError("Model has not been fitted yet.")
        X, _ = self._select_feature(X)
        xp, X, _ = self._prep(X)
        l2 = X.shape[1]
        predictions = pred_classi(self.model.d_k, X, l2, self.model.c)
        return self._post_process(predictions - self.v_k_th)

    def get_dependent_vars(self, use_weak=False):
        """
        Gets the indices of dependent variables based on the learned DNF.

        Returns:
            dependent_vars: Boolean array indicating dependent variables.
        """
        xp = self.xp
        if use_weak:
            x = xp.sum(self.learned_dnf_weak[:, :], axis=0) > 0
        else:
            x = xp.sum(self.learned_dnf[:, :], axis=0) > 0
        xx = (x[: self.n_feature] + x[self.n_feature :]) > 0
        return self._post_process(xx)

    def get_supported_score(self, X, use_weak=False):
        """
        Gets the support score for each conjunction based on the input data.

        Args:
            X: Input data features.

        Returns:
            supported_score: Support score for each conjunction.
        """
        return np.mean(self.predict_conj(X, use_weak=use_weak), axis=0)

    def get_supported_dnf(self, X, threshold=0, use_weak=False):
        """
        Gets the supported DNF based on the input data and a threshold.

        Args:
            X: Input data features.
            threshold: Threshold for support score.

        Returns:
            supported_dnf: Supported DNF.
        """
        if use_weak:
            supported_c = self.get_supported_score(X, use_weak=use_weak) > threshold
            m = self.learned_dnf_weak[supported_c, :]
        else:
            supported_c = self.get_supported_score(X) > threshold
            m = self.learned_dnf[supported_c, :]
        return m

    def get_supported_vars(self, X, threshold=0, use_weak=False):
        """
        Gets the supported variables based on the input data and a threshold.

        Args:
            X: Input data features.
            threshold: Threshold for support score.

        Returns:
            supported_vars: Boolean array indicating supported variables.
        """
        dnf = self.get_supported_dnf(X, threshold, use_weak=use_weak)
        xp = self.xp
        x = xp.sum(dnf, axis=0) > 0
        xx = (x[: self.n_feature] + x[self.n_feature :]) > 0
        return self._post_process(xx)

    def print_supported_dnf(self, X, threshold=0, use_weak=False):
        """
        Prints the supported DNF based on the input data and a threshold.

        Args:
            X: Input data features.
            threshold: Threshold for support score.
        """
        m = self.get_supported_dnf(X, threshold, use_weak=use_weak)
        self.print_dnf_mat(m)

    def score(self, X, y, metric="acc"):
        """
        Evaluates the model's performance using different metrics.

        Args:
            X: Input data features.
            y: Target variable.
            metric: Metric to use for evaluation ('acc', 'acc_dnf', 'cnsq', 'equiv', or 'all').

        Returns:
            score: Evaluation score(s).
        """
        X, _ = self._select_feature(X)
        xp, X, y = self._prep(X, y)
        l2 = X.shape[1]
        match metric:
            case "cnsq":
                cnsq, _ = logi_conseq(self.learned_dnf, y, X)
                return cnsq
            case "equiv":
                eqv, _ = logi_equiv(self.learned_dnf, y, X)
                return eqv
            case "cnsq_w":
                cnsq, _ = logi_conseq(self.learned_dnf_weak, y, X)
                return cnsq
            case "equiv_w":
                eqv, _ = logi_equiv(self.learned_dnf_weak, y, X)
                return eqv
            case "acc":
                acc_val = float(
                    acc_classi(self.model.d_k, self.v_k_th, X, y, l2, self.model.c)
                )
                return acc_val
            case "acc_dnf":
                acc_dnf_val = float(acc_dnf(self.learned_dnf, X, y, l2))
                return acc_dnf_val
            case "acc_dnf_w":
                acc_dnf_val = float(acc_dnf(self.learned_dnf_weak, X, y, l2))
                return acc_dnf_val
            case "all":
                acc_val = float(
                    acc_classi(self.model.d_k, self.v_k_th, X, y, l2, self.model.c)
                )
                print("acc\t", acc_val)
                cnsq, _ = logi_conseq(self.learned_dnf, y, X)
                eqv, _ = logi_equiv(self.learned_dnf, y, X)
                acc_dnf_val = float(acc_dnf(self.learned_dnf, X, y, l2))
                print("conseq\t", cnsq)
                print("equiv\t", eqv)
                print("acc_DNF\t", acc_dnf_val)
                cnsq_w, _ = logi_conseq(self.learned_dnf_weak, y, X)
                eqv_w, _ = logi_equiv(self.learned_dnf_weak, y, X)
                acc_dnf_val_w = float(acc_dnf(self.learned_dnf_weak, X, y, l2))
                print("conseq(w)\t", cnsq_w)
                print("equiv(w)\t", eqv_w)
                print("acc_DNF(w)\t", acc_dnf_val_w)

                return acc_val, cnsq, eqv, acc_dnf_val, cnsq_w, eqv_w, acc_dnf_val_w
            case _:
                raise ValueError("No metric:", metric)
        return None

    def print_dnf_mat(self, dnf_mat):
        """
        Prints a DNF matrix in a human-readable format.

        Args:
            dnf_mat: DNF matrix to print.
        """
        xp = self.xp
        n_clause = len(dnf_mat)
        dnf = [[] for _ in range(n_clause)]
        rows, cols = xp.where(dnf_mat > 0.5)
        if xp is not np:
            rows = xp.asnumpy(rows)
            cols = xp.asnumpy(cols)
        for i, j in zip(rows, cols):
            if j < self.n_feature:
                dnf[i].append(self.feature_names[j])
            else:
                dnf[i].append("-" + self.feature_names[j - self.n_feature])
        print("\n v  ".join([" & ".join(c) for c in dnf]))

    def print_dnf(self, use_weak=False):
        """
        Prints the learned DNF in a human-readable format.
        """
        if self.learned_dnf is None:
            raise RuntimeError("Model has not been fitted yet.")
        if use_weak:
            self.print_dnf_mat(self.learned_dnf_weak)
        else:
            self.print_dnf_mat(self.learned_dnf)

    @staticmethod
    def merge_param(model_list):
        """
        Merges the parameters of multiple MatDNFClassifier models.

        Args:
            model_list: List of MatDNFClassifier instances.

        Returns:
            c_all: Merged C-part of the models.
            d_k_all: Merged D_k-part of the models.
        """
        # c = (h, 2 * n)
        # d_k = (h,)
        xp = model_list[0].xp if model_list else np
        h_all = int(sum([m.model.c.shape[0] for m in model_list]))
        n_all = int(sum([m.model.c.shape[1] / 2 for m in model_list]))
        c_all = xp.zeros((h_all, 2 * n_all))
        d_k_all = xp.zeros(h_all)
        n_cnt = 0
        h_cnt = 0
        for m in model_list:
            c = m.model.c
            d_k = m.model.d_k
            h = c.shape[0]
            n = c.shape[1] // 2
            c_all[h_cnt : h_cnt + h, n_cnt : n_cnt + n] = c[:, :n]
            c_all[h_cnt : h_cnt + h, n_all + n_cnt : n_all + n_cnt + n] = c[:, n:]
            d_k_all[h_cnt : h_cnt + h] = d_k
            h_cnt += h
            n_cnt += n
        return c_all, d_k_all


class DNFBooleanNet(BaseEstimator):
    def __init__(
        self,
        h=1000,
        aa=4,
        er_max=0,
        alpha=0.1,
        max_itr=500,
        max_try=3,
        extra_itr=0,
        use_perturbation=True,
        use_sam=True,
        device_name="cpu",
    ):
        """
        Initializes the DNFBooleanNet.

        Args:
            h (int): Maximum number of conjunctions in a DNF.
            aa (int): Scaling factor for the random initialization.
            er_max (int): Maximum allowed error during training.
            alpha (float): Learning rate.
            max_itr (int): Maximum number of training iterations.
            max_try (int): Maximum number of attempts to find a better solution.
            extra_itr (int): Additional iterations after reaching zero error.
            use_perturbation (bool): Whether to use perturbation during training.
            use_sam (bool): Whether to use SAM optimizer.
            device_name (str): Device to use for training ('cpu' or 'gpu').
        """
        self.h = h
        self.aa = aa
        self.er_max = er_max
        self.alpha = alpha
        self.max_itr = max_itr
        self.max_try = max_try
        self.extra_itr = extra_itr
        self.use_perturbation = use_perturbation
        self.use_sam = use_sam
        self.device_name = device_name
        self.xp, self.rng = _resolve_device(device_name)
        self.learned_dnf_cls_ = {}  # Store learned DNF for each column
        self.feature_names = None

    def fit(self, X, y=None, feature_names=None, selfloop=False):
        """
        Fits the DNFBooleanNet model by training a MatDNFClassifier for each output variable.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray, optional): Target variables. If None, X is used as target variables.
            feature_names (list, optional): Names of the features.
            selfloop (bool): Whether to include the target variable as an input feature for its own DNF.

        Returns:
            self: Fitted DNFBooleanNet instance.
        """
        # Assuming X is a numpy array where each column is a variable
        # We will iterate through each column of X and treat it as the target variable Y
        num_variables = X.shape[1]
        if feature_names is None:
            feature_names = ["x" + str(i) for i in range(num_variables)]
        else:
            feature_names = feature_names
        self.feature_names = feature_names
        self.learned_dnf_cls_ = {}
        if y is None:
            y = X
        else:
            assert X.shape == y.shape, (
                "X and y could not be matched together with shapes: X="
                + str(X.shape)
                + " Y="
                + str(y.shape)
            )

        for i in range(num_variables):
            print(f"Training DNF for variable {i}...")
            # Treat the i-th column as the target variable
            Y_train_col = y[:, i]
            # Use the remaining columns as input features
            if selfloop:
                X_train_cols = X
                feature_names_cols = feature_names
            else:
                X_train_cols = np.delete(X, i, axis=1)
                feature_names_cols = np.delete(feature_names, i, axis=0)

            target_name = feature_names[i]
            n_features_for_col = X_train_cols.shape[1]

            # Initialize a new MatDNFClassifier instance for each column
            col_classifier = MatDNFClassifier(
                h=self.h,
                aa=self.aa,
                er_max=self.er_max,
                alpha=self.alpha,
                max_itr=self.max_itr,
                max_try=self.max_try,
                extra_itr=self.extra_itr,
                use_perturbation=self.use_perturbation,
                use_sam=self.use_sam,
                device_name=self.device_name,
            )

            # Train the classifier on the current column as target
            col_classifier.fit(
                X_train_cols,
                Y_train_col,
                feature_names=feature_names_cols,
                target_name=target_name,
            )

            # Store the learned DNF for this column
            # We'll store the learned_dnf attribute from the fitted classifier
            self.learned_dnf_cls_[i] = col_classifier
        return self

    def get_learned_dnf(
        self, variable_index: int, use_weak: bool = False
    ) -> NDArray:
        """
        Returns the learned DNF for a specific variable (column).

        Args:
            variable_index (int): The index of the variable.
            use_weak (bool): Whether to return the weakly simplified DNF.

        Returns:
            np.ndarray: The learned DNF matrix for the specified variable.

        Raises:
            ValueError: If no learned DNF is found for the variable index.
        """
        if variable_index in self.learned_dnf_cls_:
            if use_weak:
                return self.learned_dnf_cls_[variable_index].learned_dnf_weak
            else:
                return self.learned_dnf_cls_[variable_index].learned_dnf
        else:
            raise ValueError(
                f"No learned DNF found for variable index: {variable_index}"
            )

    def print_learned_dnfs(self, use_weak: bool = False):
        """
        Prints the learned DNF for each variable.

        Args:
            use_weak (bool): Whether to print the weakly simplified DNF.
        """
        for variable_key, dnf_cls in self.learned_dnf_cls_.items():
            print(f"Learned DNF for variable {variable_key}:")
            dnf_cls.print_dnf(use_weak=use_weak)
            print("-" * 20)
