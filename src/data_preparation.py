import argparse
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def main(args):
    print('Carico i dataset...')
    train_df = load_dataframe(args.training_path)
    val_df = load_dataframe(args.validation_path)
    test_df = load_dataframe(args.testing_path)

    train_assoc_df = load_dataframe(args.training_assoc_path)
    val_assoc_df = load_dataframe(args.validation_assoc_path)
    test_assoc_df = load_dataframe(args.test_assoc_path)

    train_w2v_df = load_dataframe(args.training_word2vec_path)
    val_w2v_df = load_dataframe(args.validation_word2vec_path)
    test_w2v_df = load_dataframe(args.testing_word2vec_path)

    train_df, val_df, test_df = load_word2vec_feature(train_df, val_df, test_df, train_w2v_df, val_w2v_df, test_w2v_df)

    print('Unisco i dataset...')
    merged_df = merge_dataframes(train_df, val_df, test_df)
    print('La dimensione del dataset è: \n', merged_df.count())
    drop_columns(merged_df)
    rename_columns(merged_df)
    print('Le prime righe del dataset sono:')
    print(merged_df.head())
    assoc_df = get_all_assoc(train_assoc_df, val_assoc_df, test_assoc_df)

    item_feature = prepare_dataframe_feature_item(args.item_data_path, list(assoc_df['product_id'].unique()))
    print(item_feature)

    print('Le prime righe della tabella delle associazioni sono:')
    print(assoc_df.head())

    user_feature = prepare_dataframe_feature_user(args.user_data_path, assoc_df, merged_df[['review_id', 'split']])

    if not os.path.exists('../data/' + args.output_folder):
        print('Creo la cartella ../data/' + args.output_folder)
        os.makedirs('../data/' + args.output_folder)
    print(f"Salvo la tabella delle review in {args.output_folder}")
    save_dataframe(merged_df, '../data/' + args.output_folder + '/review_data.csv')
    save_dataframe(assoc_df, '../data/' + args.output_folder + '/associazioni.csv')
    save_dataframe(item_feature, '../data/' + args.output_folder + '/item_data.csv')
    save_dataframe(user_feature, '../data/' + args.output_folder + '/user_data.csv')


def drop_columns(df):
    list_column_to_remove = ['date', 'content', 'review_rating', 'reviewer_id', 'product_id', 'negative_review',
                             'positive_review', 'avg_review_lenght', 'max_review_per_day', 'total_reviews',
                             'product_rating', 'reviewer_deviation']
    for col in list_column_to_remove:
        if col in list(df.columns):
            df.drop(columns=col, inplace=True)


def rename_columns(df):
    bert_feature = ['emb_' + str(i) for i in range(768)]
    bert_columns = {c: 'BERT_' + str(c) for c in df.columns.to_list() if str(c) in bert_feature}
    behavioural_feature = ['positive_ratio', 'review_rating_scaled', 'max_review_per_day_scaled',
                           'avg_review_lenght_scaled', 'reviewer_deviation_scaled']
    behavioural_columns = {c: 'BEHAV_' + str(c) for c in df.columns.to_list() if str(c) in behavioural_feature}
    cossim_feature = ['review_cosine_mean', 'max_cosine_value']
    cossim_columns = {c: 'COSSIM_' + str(c) for c in df.columns.to_list() if str(c) in cossim_feature}
    df.rename(columns=bert_columns, inplace=True)
    df.rename(columns=behavioural_columns, inplace=True)
    df.rename(columns=cossim_columns, inplace=True)
    # return df


def merge_dataframes(train_df, val_df, test_df):
    train_df['split'] = 'TRAIN'
    val_df['split'] = 'VALIDATION'
    test_df['split'] = 'TEST'
    merged_df = pd.concat([train_df, val_df, test_df], axis=0).sort_values(by='review_id')
    return merged_df


def get_all_assoc(train, val, test):
    df = merge_dataframes(train, val, test)
    df = df[['review_id', 'reviewer_id', 'product_id']]
    df = df.sort_values(by='review_id')
    return df


def prepare_dataframe_feature_item(item_data_path, list_id):
    df = load_dataframe(item_data_path)
    df = df[df['product_id_custom'].isin(list_id)]
    categories_column = df['categories']
    categories_set = set()
    for row in categories_column.items():
        for term in str(row[1]).split(','):
            categories_set.add(term.strip())
    categories_list = list(categories_set)
    categories_list.sort()
    # categories_list = ['cat_['+x+']' for x in categories_list]
    df_features = df[['product_id_custom', 'categories']].copy(deep=True)
    for category in categories_list:
        df_features[category] = df_features['categories'].apply(
            lambda x: 1 if category in [str(c).strip() for c in str(x).split(',')] else 0)
    df_features.drop(columns='categories', inplace=True)
    df_features.rename(columns={'product_id_custom': 'product_id'}, inplace=True)
    col_to_delete = []
    for col in df_features.columns:
        if df_features[col].max() == df_features[col].min():
            col_to_delete.append(col)
    df_features.drop(columns=col_to_delete, inplace=True)
    rename_columns_dict = {col: 'ITEMCAT_' + str(col) for col in df_features.columns.to_list() if col != 'product_id'}
    df_features.rename(columns=rename_columns_dict, inplace=True)
    return df_features


def load_word2vec_feature(train_df, val_df, test_df, train_w2v_df, val_w2v_df, test_w2v_df):
    column_names = {c: 'W2V_' + str(c) for c in train_w2v_df.columns.to_list()}
    train_w2v_df.rename(columns=column_names, inplace=True)
    val_w2v_df.rename(columns=column_names, inplace=True)
    test_w2v_df.rename(columns=column_names, inplace=True)
    train_df = pd.concat([train_df, train_w2v_df], axis=1)
    val_df = pd.concat([val_df, val_w2v_df], axis=1)
    test_df = pd.concat([test_df, test_w2v_df], axis=1)
    return train_df, val_df, test_df


# non convince perchè non è possibile individuare degli utenti di test e degli utenti di training
def prepare_dataframe_feature_user(user_data_path, assoc, split):
    df = load_dataframe(user_data_path)

    df = df[['reviewer_id_custom', 'friendCount', 'reviewCount', 'firstCount', 'usefulCount', 'coolCount', 'funnyCount',
             'complimentCount', 'tipCount', 'fanCount']]

    #print("Totale utenti:", len(df))
    df_feature = assoc.copy(deep=True)
    df_feature = df_feature.merge(split, right_on='review_id', left_on='review_id')
    df_feature = df_feature.drop(columns=['review_id', 'product_id'])
    #print(df_feature.head())
    df_feature = df_feature.groupby(['reviewer_id'], as_index=False).agg({'split': ' '.join})
    df_feature['split'] = df_feature['split'].apply(lambda x: 'TRAIN' if 'TRAIN' in str(x) else 'TEST')
    #print(df_feature)
    # per alcuni utenti mancano le feature, per cui sostituisco i Nan con 0
    df_feature = df_feature.merge(df, left_on='reviewer_id', right_on='reviewer_id_custom', how='left')
    df_feature = df_feature.fillna(0)
    df_feature.drop(columns='reviewer_id_custom', inplace=True)
    scaler = MinMaxScaler()
    columns_to_scale = [col for col in df_feature.columns.to_list() if col not in {'reviewer_id','split'}]
    train_examples = df_feature[df_feature['split'] == 'TRAIN']
    #print(train_examples.count())
    scaler.fit(train_examples[columns_to_scale])
    df_feature[columns_to_scale] = scaler.transform(df_feature[columns_to_scale])
    df_feature.drop(columns='split', inplace=True)

    rename_columns_dict = {col: 'USER_'+str(col) for col in df_feature.columns.to_list() if col != 'reviewer_id'}
    df_feature.rename(columns = rename_columns_dict, inplace=True)
    print(df_feature)
    return df_feature



def load_dataframe(path):
    df = pd.read_csv(path)
    return df


def save_dataframe(df, path, index_col=None):
    df.to_csv(path, index=index_col)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-training_path', type=str,
                        help='Inserisci il percorso del csv contenenti le feature delle recensioni del training set ('
                             'review centric + reviewer centric)')
    parser.add_argument('-validation_path', type=str,
                        help='Inserisci il percorso del csv contenenti le feature delle recensioni del validation set '
                             '(review centric + reviewer centric)')
    parser.add_argument('-testing_path', type=str,
                        help='Inserisci il percorso del csv contenenti le feature delle recensioni del test set ('
                             'review centric + reviewer centric)')
    parser.add_argument('-output_folder', type=str, help='Nome della cartella in cui saranno salvati i dati')
    parser.add_argument('-training_assoc_path', type=str, help='file contenente review_id, reviewer_id e product_id del'
                                                               'training set')
    parser.add_argument('-test_assoc_path', type=str, help='file contenente review_id, reviewer_id e product_id del'
                                                           'test set')
    parser.add_argument('-validation_assoc_path', type=str,
                        help='file contenente review_id, reviewer_id e product_id del'
                             'validation set')
    parser.add_argument('-item_data_path', type=str, help='file contenente i dati degli item')
    parser.add_argument('-user_data_path', type=str, help='file contenente i dati degli user')
    parser.add_argument('-training_word2vec_path', type=str, help='file contenente le feature di word2vec di train')
    parser.add_argument('-testing_word2vec_path', type=str, help='file contenente le feature di word2vec di test')
    parser.add_argument('-validation_word2vec_path', type=str,
                        help='file contenente le feature di word2vec di validation')
    args = parser.parse_args()

    args = argparse.Namespace(
        training_path='../dataset/hotel/split_temporale_YR_NR/training_set_emb_5f_simnew.csv',
        testing_path='../dataset/hotel/split_temporale_YR_NR/test_set_emb_5f_simnew.csv',
        validation_path='../dataset/hotel/split_temporale_YR_NR/validation_set_emb_5f_simnew.csv',
        training_assoc_path='../dataset/hotel/split_temporale_YR_NR/training_set_5f.csv',
        test_assoc_path='../dataset/hotel/split_temporale_YR_NR/test_set_5f.csv',
        validation_assoc_path='../dataset/hotel/split_temporale_YR_NR/validation_set_5f.csv',
        item_data_path='../dataset/hotel/dataset_modificati/product_id_custom.csv',
        user_data_path='../dataset/hotel/dataset_modificati/reviewer_id_custom.csv',
        training_word2vec_path='../dataset/hotel/split_temporale_YR_NR/training_set_32_word2vec.csv',
        testing_word2vec_path='../dataset/hotel/split_temporale_YR_NR/test_set_32_word2vec.csv',
        validation_word2vec_path='../dataset/hotel/split_temporale_YR_NR/validation_set_32_word2vec.csv',
        output_folder='bingliuhotel_sim_w2v')
    main(args)
