import matplotlib as plt


def plot_learning_curve(mean_df):
    # Assuming df contains 'Rewards' and 'Iteration' columns
    plt.figure(figsize=(10, 6))

    # Plot rewards for each player
    plt.plot(mean_df['Iteration'], mean_df['Reward1'],
             alpha=0.5, label='Player 1')
    plt.plot(mean_df['Iteration'], mean_df['Reward2'],
             alpha=0.5, label='Player 2')

    # Customize the plot
    plt.title('Learning Curves')
    plt.xlabel('Iteration')
    plt.ylabel('Rewards')
    plt.legend()
    plt.grid(True)
    plt.show()
