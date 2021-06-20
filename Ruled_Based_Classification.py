


################################### RULE BASED CLASSIFICATION  ########################################
# A game company wants to create level-based new customer definitions (personas) by using some
# features ( Country, Source, Age, Sex) of its customers, and to create segments according to these new customer
# definitions and to estimate how much profit can be generated from  the new customers according to these segments.

# In this study, how to do rule-based classification and customer-based revenue calculation
# have been discussed step by step.

########################## Importing Libraries ##########################


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 20)

########################## Importing  The Data ##########################

def load_dataset(dataframe):
    return pd.read_csv(dataframe+".csv")

df = load_dataset("persona")
df.head()


########################### Describing The Data ################################

def check_df(dataframe, head=5):
    """
    This Function returns:
        - shape : The dimension of dataframe.
        - size : Number of elements in the dataframe.
        - type : The data type of each variable.
        - Column Names : The column labels of the DataFrame.
        - Head : The first "n" rows of the DataFrame.
        - Tail : The last "n" rows of the DataFrame.
        - Null Values : Checking if any "NA" Value is into DataFrame
        - quantile : The Basics Statistics

    Parameters
    ----------
    dataframe : dataframe
        Dataframe where the dataset is kept.
    head : int, optional
        The function which is used to get the first "n" rows.

    Returns
    -------

    Examples
    ------
        import pandas as pd
        df = pd.read_csv("titanic.csv")
        print(check_df(df,10))
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Size #####################")
    print(dataframe.size)
    print("##################### Type #####################")
    print(dataframe.dtypes)
    print("############### Column Names ####################")
    print(dataframe.columns)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("################## Null Values ##################")
    print(dataframe.isnull().values.any())
    print("################## Quantiles ####################")
    print(dataframe.quantile(q=[0, 0.25, 0.50, 0.75,1]))

check_df(df)

######################## Selection of Categorical and Numerical Variables ########################

 # Let's define a function to perform the selection of numeric and categorical variables in the data set in a parametric way.

def grab_col_names(dataframe, cat_th=5, car_th=20):
    """
    This function to perform the selection of numeric and categorical variables in the data set in a parametric way.
    Note: Variables with numeric data type but with categorical properties are included in categorical variables.

    Parameters
    ----------
    dataframe: dataframe
        The data set in which Variable types need to be parsed
    cat_th: int, optional
        The threshold value for number of distinct observations in numerical variables with categorical properties.
        cat_th is used to specify that if number of distinct observations in numerical variable is less than
        cat_th, this variables can be categorized as a categorical variable.

    car_th: int, optional
        The threshold value for categorical variables with  a wide range of cardinality.
        If the number of distinct observations in a categorical variables is greater than car_th, this
        variable can be categorized as a categorical variable.

    Returns
    -------
        cat_cols: list
            List of categorical variables.
        num_cols: list
            List of numerical variables.
        cat_but_car: list
            List of categorical variables with  a wide range of cardinality.

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        Sum of elements in lists the cat_cols,num_cols  and  cat_but_car give the total number of variables in dataframe.
    """

    # cat cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and
                   dataframe[col].nunique() < cat_th]

    cat_but_car = [col for col in df.columns if dataframe[col].dtypes == "O" and
                   dataframe[col].nunique() > car_th]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return {"Categorical_Data": cat_cols,
            "Numerical_Data": num_cols,
            "Categorical_But_Cardinal_Data": cat_but_car}

grab_col_names(df)

######################## General Exploration for Categorical Data ########################

def cat_summary(dataframe, plot=False):
    cat_cols = grab_col_names(dataframe)["Categorical_Data"]
    for col_name in cat_cols:
        print("############## Unique Observations of Categorical Data ###############")
        print("The unique number of "+ col_name+": "+ str(dataframe[col_name].nunique()))

        print("############## Frequency of Categorical Data ########################")
        print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                            "Ratio": dataframe[col_name].value_counts()/len(dataframe)}))
        if plot == True:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show()

cat_summary(df, plot=True)

######################## General Exploration for Numerical Data ########################

def num_summary(dataframe,  plot=False):
    numerical_col = [ 'PRICE','AGE']##grab_col_names(dataframe)["Numerical_Data"]
    quantiles = [0.25, 0.50, 0.75, 1]
    for col_name in numerical_col:
        print("########## Summary Statistics of " +  col_name + " ############")
        print(dataframe[numerical_col].describe(quantiles).T)

        if plot:
            sns.histplot(data=dataframe, x=col_name  )
            plt.xlabel(col_name)
            plt.title("The distribution of "+ col_name)
            plt.grid(True)
            plt.show(block=True)

num_summary(df, plot=True)

######################## Data Analysis  ########################

# Unique Values of Source:

df["SOURCE"].nunique()

# Frequency of Source:

print(df[["SOURCE"]].value_counts())

# Unique Values of Price:

df[["PRICE"]].nunique()

#  Number of product sales by sales price

df[["PRICE"]].value_counts()

# Number of product sales by country

df["COUNTRY"].value_counts(ascending=False,normalize=True)

# Total & average amount of sales by country

df.groupby("COUNTRY").agg({"PRICE":["mean", "sum"})

# Average amount of sales by source

df.groupby("SOURCE").agg({"PRICE":"mean"})

# Average amount of sales by source and country

df.pivot_table( values=['PRICE'],
                index=['COUNTRY'],
                columns=["SOURCE"],
                aggfunc=["mean"]).

######################## Defining Personas ########################

# Let's define new level-based customers (personas) by using Country, Source, Age and Sex.
    # But, firstly we need to convert age variable to categorical data.

bins = [df["AGE"].min(), 18, 23, 35, 45, df["AGE"].max()]
labels = [str(df["AGE"].min())+'_18', '19_23', '24_35', '36_45', '46_'+ str(df["AGE"].max())]

df["AGE_CAT"] = pd.cut(df["AGE"], bins, labels=labels)
df.groupby("AGE_CAT").agg({"AGE":["min", "max", "count"]})

# For creating personas, we group all the features in the dataset:
df_summary = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE_CAT"])[["PRICE"]].sum().reset_index()
df_summary["CUSTOMERS_LEVEL_BASED"] = pd.DataFrame(["_".join(row).upper() for row in df_summary.values[:,0:4]])

# Calculating average amount of personas:

df_persona = df_summary.groupby('CUSTOMERS_LEVEL_BASED')[['PRICE']].mean()
df_persona = df_persona.sort_values("PRICE",ascending=False)
df_persona.head()


######################## Creating Segments based on Personas ########################

 # When we list the price in descending order, we want to express the best segment as the A segment and to define 4 segments.

segment_labels = ["D","C","B","A"]
df_persona["SEGMENT"] = pd.qcut(df_persona["PRICE"], 4, labels=segment_labels)
df_persona.reset_index(inplace=True)
df_segment = df_persona.groupby('SEGMENT').mean("PRICE").reset_index().sort_values("SEGMENT",ascending=False)

# Demonstrating segments as bars on a chart, where the length of each bar varies based on the value of the customer profile
plot = sns.barplot(x="SEGMENT" ,y = "PRICE" , data=df_segment)
for bar in plot.patches:

    plot.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=8, xytext=(0, 8),
                   textcoords='offset points')




#df_persona.groupby("SEGMENT").agg({"PRICE":"mean"}).describe().T


######################## Prediction ########################

def AGE_CAT(age):
    if age <= 18:
        AGE_CAT = "15_18"
        return AGE_CAT
    elif (age > 18 and age <=23):
        AGE_CAT = "19_23"
        return AGE_CAT
    elif (age > 23 and age <= 35):
        AGE_CAT = "24_35"
        return AGE_CAT
    elif (age > 35 and age <= 45):
        AGE_CAT = "36_45"
        return AGE_CAT
    elif (age > 45 and age <= 66):
        AGE_CAT = "46_66"
        return AGE_CAT


def ruled_based_classification():
    COUNTRY = input("Enter a country name (USA/EUR/BRA/DEU/TUR/FRA):")
    SOURCE = input("Enter the operating system of phone (IOS/ANROID):")
    SEX = input("Enter the gender (FEMALE/MALE):")
    AGE = int(input("Enter the age:"))
    AGE_SEG = AGE_CAT(AGE)
    new_user = COUNTRY.upper() + '_' + SOURCE.upper()  + '_' + SEX.upper() + '_' + AGE_SEG
    print(new_user)
    print("Segment:" + df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user].loc[:,"SEGMENT"].values[0])
    print("Price:" + str(df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user].loc[:, "PRICE"].values[0]))

ruled_based_classification()



