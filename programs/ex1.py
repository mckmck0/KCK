import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function reading files and converting them to dictionary of dataframes
def read_files(files: list, dfs: list):
    df_dictionary = {}
    for file, name in zip(files, dfs):
        df = pd.read_csv(file)
        df_dictionary[name] = df

    return df_dictionary


# Adding column of average run times
def add_means(data: dict, names: list):
    for name in names:
        df = data[name]
        columns = list(df.columns.values)[2:]
        df['average_run'] = df[columns].mean(axis=1)


# Plotting function
def plot(data: dict, names: list):
    # Dummy dictionaries for easier plotting
    efforts = {}
    generations = {}
    win_percent = {}
    for name in names:
        efforts[name] = data[name].effort
        generations[name] = data[name].generation
        win_percent[name] = data[name].average_run

    # Creating subplots
    fig = plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax11 = plt.twiny(ax1)

    # Line and scatter plots
    # Labeling the axis
    ax1.set_xlabel(r"Rozegranych gier ($\times$1000)")
    ax11.set_xlabel(r"Pokolenie")
    ax1.set_ylabel(r"Odsetek wygranych gier [%]")
    # Setting axis limits and tick intervals
    ax1.set_xlim([0, 500])
    ax1.set_xticks(np.arange(0, 501, 100))
    ax11.set_xlim([0, 200])
    ax11.set_xticks(np.arange(0, 201, 40))
    ax1.set_ylim([60, 100])
    ax1.set_yticks(np.arange(60, 101, 5))
    ax11.set_ylim([60, 100])
    ax11.set_yticks(np.arange(60, 101, 5))
    signs = ['o', 'v', '*', 's', 'D']   # Signs used for scatter plot
    merged = []
    # Plotting loop
    for name, sign in zip(names, signs):
        y1 = [100 * x for x in win_percent[name]]
        x1 = [x / 1000 for x in efforts[name]]
        x2 = generations[name]
        line, = ax1.plot(x1, y1, label=name)
        mark, = ax11.plot(x2, y1, sign, markevery=25, markeredgecolor='black')
        # Merging line and scatter plots into tuple to create the legend
        merge = (line, mark)
        merged.append(merge)

    ax1.legend(merged, names)   # Merged legend

    # Box plot
    # Converting float values into decimals (%)
    for name in names:
        data[name] = round(data[name].mul(100), 3)
    # Dropping not necessary columns, extracting last row and converting output to list
    df = {key: list(np.concatenate(df.drop(['average_run', 'generation', 'effort'],
                                           axis=1).tail(1).values.tolist()).flat) for key, df in data.items()}
    # Order of boxplots
    column_order = ['1-Evol-RS', '1-Coev-RS', '2-Coev-RS', '1-Coev', '2-Coev']
    ax2.yaxis.tick_right()
    ax2.set_ylim([60, 100])
    ax2.set_yticks(np.arange(60, 101, 5))
    # Plotting loop
    for position, column in enumerate(column_order):
        ax2.boxplot(df[column], 'blue', widths=0.5, positions=[position], showmeans=True)

    ax2.set_xticks(ticks=[0, 1, 2, 3, 4], labels=column_order, rotation=30)
    plt.savefig('plot1.pdf', dpi=300)   # Saving plot to .pdf file
    plt.show()                          # Show the plot

    return


# Main
if __name__ == '__main__':
    csv_files = glob.glob('../data/lab1_data/*.csv')
    df_names = ['1-Coev', '1-Coev-RS', '1-Evol-RS', '2-Coev', '2-Coev-RS']
    print(os.getcwd())
    print(csv_files)
    df_dicts = read_files(csv_files, df_names)
    add_means(df_dicts, df_names)
    # print(df_dicts)
    plot(df_dicts, df_names)