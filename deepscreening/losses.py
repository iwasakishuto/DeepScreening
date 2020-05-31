"""[Define triplet-loss]"""

import tensorflow as tf

class TripletLoss():
    __name__ = 'triplet_loss'

    def __init__(self, margin=0.5, squared=False, strategy="batch_hard", K=1, label_index=0, distance="euclid"):
        self.margin = margin
        self.squared = squared
        self.strategy = strategy
        self.K = K
        self.label_index = label_index
        self.distance = distance
        """
        If you define loss's hyper parameters as `self.hogehoge`, you can adjust the relationship with other losses.
        The simplest adjustment method is to multiply by a constant K. (self.K)
        """

    def __call__(self, y_true, y_pred):
        """
        @param y_true: (Tensor) shape=(batch size, 2),
                       - triplet_label=y_true[:,0]
                       - softmax_label=y_true[:,1]
        @param y_pred: (Tensor) shape=(batch size, 4096),
        """
        triplet_label = y_true[:,self.label_index]

        if self.strategy=="batch_hard":
            ret = self.batch_hard_triplet_loss(triplet_label, y_pred)
        elif self.strategy=="batch_all":
            ret = self.batch_all_triplet_loss(triplet_label, y_pred)
        else:
            print("Can't understand the strategy '{}'.".format(self.strategy))

        """
        If you change the hyper parameter here, you can adjust the loss relationship with time flow.
        """
        return ret

    # Return a 2D mask where mask[a][p] is True if p is valid positive.
    def _get_anchor_positive_triplet_mask(self, labels):
        # Check that i and j are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)

        a_equal_p = tf.expand_dims(labels, 0) # shape = (1, batch_size)
        p_equal_a = tf.expand_dims(labels, 1) # shape = (batch_size, 1)

        # Check if labels[i] == labels[j]
        labels_equal = tf.equal(a_equal_p, p_equal_a) # Using broadcast

        # Combine the two masks
        mask = tf.logical_and(indices_not_equal, labels_equal)

        return mask


    # Return a 2D mask where mask[a][n] is True if n is valid negative.
    # If label is not equal, two sample must be distinct.
    def _get_anchor_negative_triplet_mask(self, labels):
        a_equal_n = tf.expand_dims(labels, 0) # shape = (1, batch_size)
        n_equal_a = tf.expand_dims(labels, 1) # shape = (batch_size, 1)

        # Check if labels[i] != labels[k]
        labels_equal = tf.equal(a_equal_n, n_equal_a) # Use broadcast

        mask = tf.logical_not(labels_equal)

        return mask


    # Return a 3D mask where mask[a][p][n] is True if the triplet (a, p, n) is valid.
    def _get_triplet_mask(self, labels):
        # Check that i, j and k are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        i_not_equal_j = tf.expand_dims(indices_not_equal, 2) # shape = (batch_size, batch_size, 1)
        i_not_equal_k = tf.expand_dims(indices_not_equal, 1) # shape = (batch_size, 1, batch_size)
        j_not_equal_k = tf.expand_dims(indices_not_equal, 0) # shape = (1, batch_size, batch_size)

        distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        i_equal_j = tf.expand_dims(label_equal, 2)
        i_equal_k = tf.expand_dims(label_equal, 1)

        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

        # Combine the two masks
        mask = tf.logical_and(distinct_indices, valid_labels)

        return mask


    # Triplet loss depends on the distance,
    # so first prepare the function which could compute the pairwise distance matrix efficiently
    def _pairwise_euclid_distances(self, embeddings, squared=False):
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings)) # dot product between all embeddings
        square_norm = tf.diag_part(dot_product) # extract only diagonal elements so that get respective norms.

        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
        distances = tf.maximum(distances, 0.0) # for computational errors

        if not squared:
            mask = tf.to_float(tf.equal(distances, 0.0)) # for avoid the gradient of sqrt is infinite,
            distances = distances + mask * 1e-16 # add a small epsilon where distances == 0.0
            distances = tf.sqrt(distances)
            distances = distances * (1.0 - mask) # Correct the epsilon added

        return distances

    def _pairwise_cosine_similarity(self, embeddings):
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings)) # dot product between all embeddings
        square_norm = tf.diag_part(dot_product) # extract only diagonal elements so that get respective norms.
        norm = tf.sqrt(square_norm)

        similalities = dot_product / (tf.expand_dims(norm, 0) * tf.expand_dims(norm, 1))
        distances = 1 - similalities

        return distances

    # Compute the all valid triplet loss and average them.
    def batch_all_triplet_loss(self, y_true, y_pred):
        # Get the pairwise distance matrix
        if self.distance == "euclid":
            pairwise_dist = self._pairwise_euclid_distances(y_pred, squared=self.squared)
        elif self.distance == "cosine":
            pairwise_dist = self._pairwise_cosine_similarity(y_pred)

        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2) # shape = (batch_size, batch_size, 1)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1) # shape = (batch_size, 1, batch_size)

        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin # Using broadcast.

        # Remove(Put zero) the invalid triplets.
        mask = self._get_triplet_mask(y_true)
        mask = tf.to_float(mask)
        triplet_loss = tf.multiply(mask, triplet_loss)

        # Remove negative losses (which are the easy triplets).
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # Count the number of positive triplets.
        valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16)) # val > 0 means positive triplets
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        rate_of_positive = num_positive_triplets / (num_valid_triplets + 1e-16) # Rate of positive triplets from valid.

        ave_triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

        return ave_triplet_loss


    def batch_hard_triplet_loss(self, y_true, y_pred):
        """
        Compute only the triplets (hardest positive, hardest negative, anchor) for each anchor.
        =======================
        @param y_true: shape=(b,)  'b' means batch size.
        @param y_pred: shape=(b,e) 'e' means embedding size.
        """
        # Get the pairwise distance matrix
        if self.distance == "euclid":
            pairwise_dist = self._pairwise_euclid_distances(y_pred, squared=self.squared)
        elif self.distance == "cosine":
            pairwise_dist = self._pairwise_cosine_similarity(y_pred)

        #=== hardest positive ===
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(y_true) # mask(same label and distinct). If i-th and j-th are valid positive pairs, mask[i][j] = True. shape=(b,b)
        mask_anchor_positive = tf.to_float(mask_anchor_positive) # turn True to 1, and False to 0, shape=(b,b)
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist) # multiply mask to set the invalid pair's value 0, shape=(b,b)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True) # for each data, leave only the farthest positive data ('s distance.). shape = (b, 1)

        #=== hardest negative ===
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(y_true) # mask(different label and distinct). If i-th and j-th are valid negative pairs, mask[i][j] = True. shape=(b,b)
        mask_anchor_negative = tf.to_float(mask_anchor_negative) # turn True to 1, and False to 0, shape=(b,b)
        """
        [Aim] Calcurate 'hardest negative' distance.
        [Problem] If you calcurate like positive,
        ```
        anchor_negative_dist  = tf.multiply(mask_anchor_negative, pairwise_dist) # The diagonal value must be 0 !!
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True) # ← All elements must be 0 !!!!
        ```
        [Solution]
        1. Add a penalty (maximum value for each) to invalid negative pairs
        2. Then, calcurate the minimum distance for each data.
        """
        max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True) # 1. Calcurate the penalty (maximum value) in each row. shape = (b, 1)
        mask_anchor_negative_invalid = 1.0-mask_anchor_negative # If i-th and j-th is invalid negative pairs, mask[i][j] = True. shape=(b,b)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * mask_anchor_negative_invalid # Add a penalty to invalid data. shape=(b, b)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True) # for each data, leave only the nearest negative data ('s distance.). shape = (b, 1)

        #=== Combine: So far, we get the farest positive, and nearest negative distances for each data. ===
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + self.margin, 0.0) # Find biggest d(a, p) and smallest d(a, n)
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss
