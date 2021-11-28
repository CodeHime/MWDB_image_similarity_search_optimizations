from decision_tree_funcs import *
from sklearn.metrics import confusion_matrix

class DecisionTree:
    def __init__(self, vectors_df, counter=0, min_support=0.8, min_samples=2, max_depth=5):
        self.print_tree = self.create_decision_tree(vectors_df, counter=counter, min_support=min_support,
                                                    min_samples=min_samples, max_depth=max_depth)

    def create_decision_tree(self, vectors_df, counter=0, min_support=0.6, min_samples=2, max_depth=5):
        count_df = \
        vectors_df.groupby(vectors_df.iloc[:, -1]).count().sort_values(by=vectors_df.columns[0], ascending=False)[
            vectors_df.columns[0]]
        support = count_df.iloc[0] / count_df.sum()
        # if all values have the same label or min samples or max depth is reached then exit creation
        if (len(vectors_df.iloc[:, -1].unique()) == 1) or (len(vectors_df) < min_samples) or (counter == max_depth) or (
                support > min_support):
            label_column = vectors_df.iloc[:, -1]
            unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

            if len(counts_unique_classes):
                index = counts_unique_classes.argmax()
                classification = unique_classes[index]
                return classification
            return -1

        # recursive part
        else:
            potential_splits = get_all_potential_splits(vectors_df)
            split_column, split_value = determine_best_split(vectors_df, potential_splits)
            data_below, data_above = split_col_data(vectors_df, split_column, split_value)

            # instantiate sub-tree
            feature_name = vectors_df.columns[split_column]
            question = "{} <= {}".format(feature_name, split_value)

            sub_tree = {question: []}
            # find answers (recursion)
            left_answer = self.create_decision_tree(data_below, counter + 1, min_samples, max_depth)
            right_answer = self.create_decision_tree(data_above, counter + 1, min_samples, max_depth)

            # If the answers are the same, then there is no point in asking the qestion.
            # This could happen when the data is classified even though it is not pure
            # yet (min_samples or max_depth base cases).
            if right_answer == -1 or left_answer == right_answer:
                sub_tree = left_answer
            elif left_answer == -1:
                sub_tree = right_answer
            else:
                sub_tree[question].append(left_answer)
                sub_tree[question].append(right_answer)

            return sub_tree

    def query_tree(self, xq):
        parse_tree = self.print_tree
        while type(parse_tree) == dict:
            cur_branch_comparison = list(parse_tree.keys())[0]
            index_col, compare, value = cur_branch_comparison.split()
            # print(index_col, compare, value)
            if eval(f"xq[{index_col}] {compare} {value}"):
                parse_tree = parse_tree[cur_branch_comparison][0]
            else:
                parse_tree = parse_tree[cur_branch_comparison][1]
        return parse_tree

    def get_prediction_summary(self, xq_df, labels):
        act_val = xq_df.iloc[:, -1]
        pred_val = []
        print(self.print_tree)
        print("xq_df", xq_df)
        for i in xq_df.index:
            pred_val.append(self.query_tree(xq_df.iloc[i, :]))
        conf_mat = confusion_matrix(act_val, pred_val, labels=labels)
        print("*"*10 + "  CONFUSION MATRIX " + "*"*10)
        print(conf_mat)
        # Calculate false +ves
        sum_col = conf_mat.sum(axis=0) - conf_mat.diagonal()
        # Calculate misses
        sum_row = conf_mat.sum(axis=1) - conf_mat.diagonal()
        print(f"False +ves: {sum(sum_col)} \nMisses: {sum(sum_row)}")
        print("*"*40)
