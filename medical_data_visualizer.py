import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Solve bmi, logical operator, create 'overweight' column
df.loc[df['weight']/(df['height']/100)**2 > 25, 'overweight'] = 1
df.loc[df['weight']/(df['height']/100)**2 <= 25, 'overweight'] = 0

# Convert 'overweight' column to integers
df['overweight'] = df['overweight'].astype(int)
# (df['overweight'] == 0).sum() /// Count how many have the value of 0 in overweight column
# (df['overweight'] == 1).sum() /// Count how many have the value of 1 in overweight column

# Normalize 'cholesterol' 'gluc' values
df[df[['cholesterol', 'gluc']] == 1] = 0
df[df[['cholesterol', 'gluc']] > 1] = 1
# df[['cholesterol', 'gluc']] /// View the normalized 'cholesterol' and 'gluc' columns
# df[df[['cholesterol', 'gluc']] == 0].count() /// Count how many are 0 or 1

# Draw Catplot
def draw_cat_plot():
    df_cat = sorted(['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = pd.melt(df, id_vars='cardio', value_vars=df_cat)

    # Draw using catplot and store it in fig variable
    fig = sns.catplot(x='variable', col='cardio', hue='value', kind='count', data=df_cat).set_axis_labels('variable', 'total')


    plt.show()


def draw_heat_map():
    
    # Clean the data
    df_heat = df.loc[(df['ap_lo'] <= df['ap_hi']) &
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975))]

    #Calculate the correlation matrix
    corr = round(df_heat.corr(), 2)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize = (8, 8))

    # Draw the heatmap with 'sns.heatmap()'
    ax = sns.heatmap(corr, vmin=0, vmax=0.25, annot=True, fmt='.2f', annot_kws=None, linewidths=0, square=True, mask=mask, cbar_kws={'shrink': 70})


    plt.show()

draw_cat_plot()
draw_heat_map()


