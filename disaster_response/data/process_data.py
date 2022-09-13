import pandas as pd
import sys

from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str):
    '''
    input:
        messages_filepath: str - path to the messages csv file
        categories_filepath: str - path to the categories csv file
    output:
        df: pd.DataFrame - dataframe combining messages and categories
    '''
    messages = pd.read_csv(messages_filepath, index_col=False)
    categories = pd.read_csv(categories_filepath, index_col=False)
    df = messages.merge(categories, on="id", how="outer")

    return df


def clean_data(df):
    # split categories column into separate category columns
    categories = df["categories"].str.split(";", expand=True)
    categories.columns = [x.split("-")[0] for x in categories.iloc[0]]

    # convert category values to 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # replace categories column in df with new category columns
    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    df.drop(["id", "original"], axis=1, inplace=True)
    df.drop_duplicates(inplace=True)

    # remove categories that don't get any labels
    tmp = df.sum(numeric_only=True).reset_index()
    cat_no_label = tmp[tmp[0] == 0]["index"].tolist()
    df.drop(cat_no_label, axis=1, inplace=True)

    return df


def save_data(df, database_filename):
    '''
    saves df to database_filename
    '''
    engine = create_engine(database_filename)
    df.to_sql(
        "tbl_yyh_disaster_response_clean_data", 
        engine, 
        index=False, 
        if_exists="replace"
    )


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
