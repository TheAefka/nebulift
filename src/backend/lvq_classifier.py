import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from backend.classify import DIFFUSE, COMPACT, UNCLASSIFIED

class LVQClassifier:
    """
    Online Learning Vector Quantisation classifier.
    One prototype per class, updated each time a label arrives.
    Falls back to nearest-prototype for unlabelled objects.
    """

    def __init__(self, features: list[str], learning_rate: float = 0.1):
        self.features      = features
        self.learning_rate = learning_rate
        self.scaler        = StandardScaler()
        self.fitted        = False

        # state specific to the current image
        self.labelled_ids  = {}     # {obj_id: class_label}
        self.prototypes    = {}     # {class_label: np.ndarray}
        
        # shared across images
        self.persistent_labels = [] # [(np.ndarray, class_label), ...]

        # undo/redo history
        self.history       = []     # [(dict, list), ...]
        self.history_index = -1     # pointer for undo/redo in history

    def reset(self):
        self.labelled_ids.clear()
        self.prototypes.clear()
        self.persistent_labels.clear()
        self.history.clear()
        self.history_index = -1
        self.fitted = False
    
    def reset_on_image_change(self, df: pd.DataFrame = None):
        self.labelled_ids.clear()
        self.prototypes.clear()
        self.history.clear()
        self.history_index = -1
        self.fitted = False
        if self.persistent_labels and df is not None:
            self.refit_from_persistent(df)

    def add_label(self, obj_id: int, label: int, df: pd.DataFrame):
        """
        Record a user label and update prototypes.
        """
        self.labelled_ids[obj_id] = label

        row = df.loc[obj_id, self.features].replace([np.inf, -np.inf], np.nan)
        if not row.isna().any():
            self.persistent_labels.append((row.values, label))
        self.push_history()
        self.refit(df)

    def remove_label(self, obj_id: int, df: pd.DataFrame):
        """
        Remove a previously added label.
        """
        if obj_id in self.labelled_ids:
            del self.labelled_ids[obj_id]
            self.push_history()
            self.refit(df)

    def is_ready(self) -> bool:
        """
        True once at least one label exists for each class.
        """
        classes_current = set(self.labelled_ids.values())
        classes_persistent = set(label for _, label in self.persistent_labels)
        combined_classes = classes_current | classes_persistent
        return COMPACT in combined_classes and DIFFUSE in combined_classes

    def refit_from_persistent(self, df: pd.DataFrame):
        """
        Refit the classifier using only the persistent labels.
        """
        return self.refit(df)
        # all_features = df[self.features].replace([np.inf, -np.inf], np.nan).fillna(0)
        # self.scaler.fit(all_features.values)
        # self.fitted = True

        # vectors = np.array([vec for vec, _ in self.persistent_labels])
        # labels  = np.array([lbl for _, lbl in self.persistent_labels])
        # scaled_vectors = self.scaler.transform(
        #     pd.DataFrame(vectors, columns=self.features).replace([np.inf, -np.inf], np.nan).fillna(0).values
        # )

        # for label in set(labels):
        #     mask = np.array([lbl == label for lbl in labels])
        #     self.prototypes[label] = scaled_vectors[mask].mean(axis=0)

    def classify(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify every row in df.
        """
        if not self.fitted:
            return pd.Series(UNCLASSIFIED, index=df.index)

        X = self.extract(df) # (N, F)
        results = np.full(len(df), UNCLASSIFIED, dtype=int)

        proto_classes = list(self.prototypes.keys())
        proto_vecs    = np.stack([self.prototypes[c] for c in proto_classes])  # (C, F)

        # squared distances to each prototype
        diffs = X[:, None, :] - proto_vecs[None, :, :]   # (N, C, F)
        dists = (diffs ** 2).sum(axis=2)                 # (N, C)
        nearest = np.argmin(dists, axis=1)               # (N,)

        for i, cls_idx in enumerate(nearest):
            results[i] = proto_classes[cls_idx]

        # overwritten labelled objects always keep their user label
        series = pd.Series(results, index=df.index)
        for object_id, label in self.labelled_ids.items():
            if object_id in series.index:
                series.loc[object_id] = label

        return series



    def refit(self, df: pd.DataFrame):
        """Re-initialise scaler and prototypes from all current labels."""
        if not self.labelled_ids:
            self.fitted = False
            return

        # refit scaler on all rows of the current image
        all_features = df[self.features].replace([np.inf, -np.inf], np.nan).fillna(0)
        self.scaler.fit(all_features.values)
        self.fitted = True
 
        vectors = np.array([vec for vec, _ in self.persistent_labels])
        labels  = np.array([lbl for _, lbl in self.persistent_labels])
        scaled  = self.scaler.transform(
            pd.DataFrame(vectors, columns=self.features)
              .replace([np.inf, -np.inf], np.nan).fillna(0).values
        )
 
        # initialise prototypes as class centroids
        self.prototypes = {
            label: scaled[labels == label].mean(axis=0)
            for label in set(labels.tolist())
        }


        # update prototypes based on labelled objects
        lr = self.learning_rate
        for x, label in zip(scaled, labels):
            best_cls, best_dist = None, np.inf
            for cls, proto in self.prototypes.items():
                d = np.sum((x - proto) ** 2)
                if d < best_dist:
                    best_dist, best_cls = d, cls
 
            if best_cls == label:
                self.prototypes[best_cls] += lr * (x - self.prototypes[best_cls])
            else:
                self.prototypes[best_cls] -= lr * (x - self.prototypes[best_cls])
                if label in self.prototypes:
                    self.prototypes[label] += lr * (x - self.prototypes[label])


    def extract(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].replace([np.inf, -np.inf], np.nan).fillna(0).values
        return self.scaler.transform(X)
    
    def push_history(self):
        del self.history[self.history_index + 1:]
        self.history.append((
            dict(self.labelled_ids),
            list(self.persistent_labels),
        ))

        self.history_index += 1
    

    # Undo/redo functionality
    # Small bug exists when undoing and redoing a noise label after compact
    # and diffuse labels have been added.

    def undo(self, df: pd.DataFrame):
        if self.history_index < 0:
            return
        self.history_index -= 1
        if self.history_index >= 0:
            ids, persistent = self.history[self.history_index]
            self.labelled_ids      = dict(ids)
            self.persistent_labels = list(persistent)
        else:
            self.labelled_ids.clear()
            self.persistent_labels.clear()
        self.refit(df)

        print(f"LVQ Undo: Reverted to {len(self.labelled_ids)} labels")
    
    def redo(self, df: pd.DataFrame):
        if self.history_index >= len(self.history) - 1:
            return
        self.history_index += 1
        ids, persistent = self.history[self.history_index]
        self.labelled_ids      = dict(ids)
        self.persistent_labels = list(persistent)
        self.refit(df)

        print(f"LVQ Redo: Restored to {len(self.labelled_ids)} labels")