import pandas as pd

def main():
    pass

def pandaDemo():
    print("Panda Demo")
    # Set panda not to truncate
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_colwidth', -1)

    # Show Version
    # print(pd.__version__)

    # Prepare Data Set
    cities = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
    # print(cities)

    # Create Data Frame
    city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
    population = pd.Series([852469, 1015785, 485199])
    df = pd.DataFrame({ 'City name': city_names, 'Population': population })
    # print(df)

    # Get Data From CSV (Network)
    california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
    # print(california_housing_dataframe.describe())

    # Showing Only Head
    # print(california_housing_dataframe.head())

    # Make Graph
    # You can not on this environment
    # california_housing_dataframe.hist('housing_median_age')

    # DataFrame is just like an array
    cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
    # print(type(cities['City name']))
    # print(cities['City name'])

    # You can also access by index
    # print(type(cities['City name'][1]))
    # print(cities['City name'][1])

    # Also you can specify range
    # print(type(cities[0:2]))
    # print(cities[0:2])

    # You can also do this
    # print(population / 1000)

    # Or this
    # print(np.log(population))
    # Or this
    # print(population + 100)

    # Or this
    # print(population.apply(lambda val: val > 1000000))

    # This is how you modify data
    cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
    cities['Population density'] = cities['Population'] / cities['Area square miles']
    # print(cities)

    # You can do also complicated things like this
    cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: name.startswith('San'))
    # print(cities)

    # Examples of Indexing
    # print(city_names.index)
    # print(cities.index)
    # cities.reindex([2, 0, 1])

    # How to shuffle
    cities.reindex(np.random.permutation(cities.index))
    # print(cities)

if __name__ == '__main__':
    main()