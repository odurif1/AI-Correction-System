"""
Clustering and embedding analysis for student answers.

Uses embeddings and DBSCAN to group similar answers together.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from core.models import CopyDocument, AnswerCluster


class EmbeddingClustering:
    """
    Clusters student answers using embeddings and DBSCAN.

    Groups similar answers together to enable:
    - Consistent grading across similar responses
    - Pattern detection
    - Efficient retroactive rule application
    """

    def __init__(self, eps: float = 0.3, min_samples: int = 2):
        """
        Initialize the clustering engine.

        Args:
            eps: Maximum distance between samples in same cluster
            min_samples: Minimum samples to form a cluster
        """
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")

    def compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        from ai import create_ai_provider

        provider = create_ai_provider()
        return provider.get_embeddings(texts)

    def cluster_texts(
        self,
        texts: List[str],
        copy_ids: List[str],
        embeddings: List[List[float]] = None
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[float]]]:
        """
        Cluster texts by semantic similarity.

        Args:
            texts: List of text answers
            copy_ids: Corresponding copy IDs
            embeddings: Pre-computed embeddings (optional)

        Returns:
            (clusters, centroids) where:
                clusters: {cluster_id: [copy_ids]}
                centroids: {cluster_id: centroid_vector}
        """
        if embeddings is None:
            embeddings = self.compute_embeddings(texts)

        # Convert to numpy array
        X = np.array(embeddings)

        # Perform clustering
        labels = self.dbscan.fit_predict(X)

        # Group by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:  # Noise point
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(copy_ids[idx])

        # Compute centroids
        centroids = {}
        for cluster_id, member_indices in [
            (cid, [i for i, l in enumerate(labels) if l == cid])
            for cid in clusters.keys()
        ]:
            if member_indices:
                centroid = X[member_indices].mean(axis=0).tolist()
                centroids[cluster_id] = centroid

        return clusters, centroids

    def cluster_by_question(
        self,
        copies: List[CopyDocument],
        question_id: str
    ) -> List[AnswerCluster]:
        """
        Cluster answers for a specific question.

        Args:
            copies: List of copy documents
            question_id: Question to cluster

        Returns:
            List of AnswerCluster objects
        """
        # Extract answers for this question
        copy_ids = []
        answers = []

        for copy in copies:
            if question_id in copy.content_summary:
                copy_ids.append(copy.id)
                answers.append(copy.content_summary[question_id])

        if not answers:
            return []

        # Compute embeddings
        embeddings = self.compute_embeddings(answers)

        # Cluster
        clusters_dict, centroids = self.cluster_texts(
            answers,
            copy_ids,
            embeddings
        )

        # Create AnswerCluster objects
        result = []
        for cluster_id, member_ids in clusters_dict.items():
            # Find a representative answer
            member_indices = [copy_ids.index(cid) for cid in member_ids]
            representative_idx = member_indices[0]
            representative_answer = answers[representative_idx]

            cluster = AnswerCluster(
                cluster_id=cluster_id,
                question_id=question_id,
                copy_ids=member_ids,
                representative_description=representative_answer[:200],
                representative_answer=representative_answer,
                centroid_embedding=centroids[cluster_id]
            )
            result.append(cluster)

        return result

    def find_similar_copies(
        self,
        query_text: str,
        copies: List[CopyDocument],
        threshold: float = 0.85,
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        Find copies similar to a query text.

        Args:
            query_text: Query text
            copies: List of copy documents
            threshold: Minimum similarity score
            top_k: Maximum results (None = no limit)

        Returns:
            List of (copy_id, similarity_score) tuples
        """
        from ai import create_ai_provider

        provider = create_ai_provider()
        query_embedding = provider.get_embedding(query_text)

        # Get embeddings for all copies
        similarities = []

        for copy in copies:
            if copy.embedding is None:
                continue

            # Compute cosine similarity
            similarity = self._cosine_similarity(
                query_embedding,
                copy.embedding
            )

            if similarity >= threshold:
                similarities.append((copy.id, similarity))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            similarities = similarities[:top_k]

        return similarities

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def compute_cluster_statistics(
        self,
        cluster: AnswerCluster,
        grades: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Compute average grade and variance for a cluster.

        Args:
            cluster: AnswerCluster
            grades: Dict of {copy_id: grade}

        Returns:
            (mean, variance)
        """
        cluster_grades = [
            grades.get(cid, 0) for cid in cluster.copy_ids
            if cid in grades
        ]

        if not cluster_grades:
            return None, None

        mean = np.mean(cluster_grades)
        variance = np.var(cluster_grades) if len(cluster_grades) > 1 else 0.0

        return float(mean), float(variance)

    def detect_outliers(
        self,
        copies: List[CopyDocument],
        question_id: str
    ) -> List[str]:
        """
        Detect outlier answers (noise points in DBSCAN).

        Args:
            copies: List of copy documents
            question_id: Question to analyze

        Returns:
            List of outlier copy IDs
        """
        # Extract answers
        copy_ids = []
        answers = []

        for copy in copies:
            if question_id in copy.content_summary:
                copy_ids.append(copy.id)
                answers.append(copy.content_summary[question_id])

        if not answers:
            return []

        # Compute embeddings and cluster
        embeddings = self.compute_embeddings(answers)
        X = np.array(embeddings)
        labels = self.dbscan.fit_predict(X)

        # Find noise points (-1 label)
        outliers = [
            copy_ids[i] for i, label in enumerate(labels)
            if label == -1
        ]

        return outliers

    def get_optimal_eps(
        self,
        texts: List[str],
        sample_size: int = 100
    ) -> float:
        """
        Estimate optimal eps parameter using nearest neighbors.

        Args:
            texts: Sample texts to analyze
            sample_size: Max samples to use

        Returns:
            Estimated eps value
        """
        from sklearn.neighbors import NearestNeighbors

        # Sample if needed
        if len(texts) > sample_size:
            import random
            texts = random.sample(texts, sample_size)

        embeddings = self.compute_embeddings(texts)
        X = np.array(embeddings)

        # Find k-nearest neighbors distances
        neighbors = NearestNeighbors(n_neighbors=min(4, len(texts)))
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)

        # Use the 90th percentile of nearest neighbor distances
        k_distances = distances[:, -1]
        eps = np.percentile(k_distances, 90)

        return float(eps)
