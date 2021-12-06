# Import packages for use throughout the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


#Importing Data and Selecting Columns to Study, Drop N/A values
data = pd.read_csv("nba.csv")
df = data[['Player','Year','Draft pick', 'Height (No Shoes)','Weight','Body Fat','Agility','Sprint']]
df = df.dropna()



#Setting Up The Dashboard With Header and Image as well as the description
from PIL import Image
image = Image.open("nbalogo.jpeg")
st.image(image)
st.title('NBA Combine Data Dashboard')
st.markdown('Our dashboard can be used to see combine stats and determine what factors most influence speed and agility')





#Vizualizing the Data Through a Bar Chart Year by Year, with a slider to look at data from a certain year
years = list(df['Year'].drop_duplicates())
values = st.slider('Select a year', 2009, 2020)
st.write(df[df['Year'] == values])
df2 = pd.DataFrame(df[df['Year'] == values])
chart_data = pd.DataFrame(df2,columns=['Sprint',"Agility"])
st.bar_chart(chart_data)






#Vizualizing the Data For Each Variable through histograms
df.hist()
plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()







#Regression Analysis on Data and output on regression curve
def regress(x,y,nam1,nam2):
    N = len(x)
    xmean = x.mean()
    ymean = y.mean()

    B1n = ((x - xmean) * (y - ymean)).sum()
    B1d = ((x - xmean) ** 2).sum()
    B1 = B1n/B1d

    B0 = ymean - (B1 * xmean)

    reg = 'y = {} + {}B'.format(B0,round(B1,3))

    print("The Equation for the variable "+ nam1 +" In terms of: " + nam2 +" Is "+  reg)


    num = (N * (x*y).sum()) - (x.sum() * y.sum())
    den = np.sqrt((N * (x**2).sum() - x.sum()**2) * (N * (y**2).sum() - y.sum()**2))
    R = num / den


    plt.figure(figsize=(12,5))
    plt.scatter(x, y, s=300, linewidths=1, edgecolor='black')
    text = '''X Mean: {} Years
    Y Mean: ${}
    R: {}
    R^2: {}
    y = {} + {}X'''.format(round(x.mean(), 2),
                           round(y.mean(), 2),
                           round(R, 4),
                           round(R**2, 4),
                           round(B0, 3),
                           round(B1, 3))
    plt.text(x=1, y=100000, s=text, fontsize=12, bbox={'facecolor': 'grey', 'alpha': 0.2, 'pad': 10})
    plt.title('Regression Line')
    plt.xlabel(nam1, fontsize=15)
    plt.ylabel(nam2, fontsize=15)
    plt.plot(x, B0 + B1*x, c = 'r', linewidth=5, alpha=.5, solid_capstyle='round')
    plt.scatter(x=x.mean(), y=y.mean(), marker='*', s=10**2.5, c='r')
    plt.plot()






#Running the Regression Analysis function and Outputting Regression Curve
regress(df['Body Fat'],df['Sprint'],"Body Fat","Sprint")
regress(df['Body Fat'],df['Agility'],"Body Fat","Agility")
regress(df['Weight'],df['Sprint'],"Weight","Sprint")
regress(df['Weight'],df['Agility'],"Weight","Agility")
regress(df['Height (No Shoes)'],df['Agility'],"Height (No Shoes)","Agility")
regress(df['Height (No Shoes)'],df['Agility'],"Height (No Shoes)","Agility")


#Corellation Heatmap & Matrix for each variable
fig, ax = plt.subplots()
cor = df.corr()
sns.heatmap(cor, ax = ax)
st.write(fig)




# Streamlit commands to see the amount of players enetering the draft each year, with a slider to examine each individual year
st.header('Amount of NBA Draft Prospects By Year')
min_years = st.slider('Filter # GP:', 2009, int(df['Year'].max()))
filtered_data = df[df['Year'] == min_years]
st.markdown(f'Number of players who joined the league in {min_years}: {len(filtered_data)}')





